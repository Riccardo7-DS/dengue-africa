import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation, binary_closing
from tqdm import tqdm
from utils import compute_ndvi

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Cloud mask preprocessing
# ─────────────────────────────────────────────

@dataclass
class CloudMaskResult:
    """Outputs of generate_cloud_mask, one per spatial tile."""
    cleaned_mask: np.ndarray       # bool   (H, W) — True = cloudy
    soft_score: np.ndarray         # float32 (H, W) in [0, 1] — loss weight, 1 = confident clear
    meta_mask_channel: np.ndarray  # float32 (H, W) in [0, 1] — extra GFM input channel


def generate_cloud_mask(
    mod09qa_bits: np.ndarray,
    mod35_confidence: np.ndarray,
    blue_band: np.ndarray,
) -> CloudMaskResult:
    """
    Build a cleaned binary cloud mask, a per-pixel soft confidence score,
    and a meta-mask channel for GFM input from three co-registered arrays.

    Parameters
    ----------
    mod09qa_bits : np.ndarray (H, W), uint16
        ``state_1km`` QA bits from MOD09GA.  Bits 0-1 encode cloud state:
        0b00 = clear, 0b01 = cloudy, 0b10 = mixed, 0b11 = not set.
        Pixels with value 1 (cloudy) or 2 (mixed) are treated as cloud.
    mod35_confidence : np.ndarray (H, W), uint8
        MOD35 4-level unobstructed-FOV quality flag stored in zarr as
        ``cloud_probability``:
        0 = confident cloudy, 1 = probably cloudy,
        2 = probably clear,   3 = confident clear.
    blue_band : np.ndarray (H, W), float32
        Band 3 (blue, ~459–479 nm) surface reflectance in physical units
        (i.e. raw DN / 10000).  Used as a spectral veto and soft penalty.

    Returns
    -------
    CloudMaskResult
        cleaned_mask      — bool (H, W), True = cloudy
        soft_score        — float32 (H, W) in [0, 1], use directly as pixel
                            weight in a masked reconstruction loss
        meta_mask_channel — float32 (H, W) in [0, 1], concatenate as an
                            extra channel alongside the imagery fed to the GFM
    """
    # ------------------------------------------------------------------
    # Raw binary cloud flags from each product
    # ------------------------------------------------------------------

    # MOD09GA state_1km — bits 0-1: 1=cloudy, 2=mixed both indicate cloud.
    cloud_bits  = (mod09qa_bits & 0b11).astype(np.uint8)
    mod09_cloud = np.isin(cloud_bits, [1, 2])  # bool (H, W)

    # MOD35 — confidence 0 (confident cloudy) or 1 (probably cloudy) → cloud.
    mod35_cloud = mod35_confidence <= 1  # bool (H, W)

    # ------------------------------------------------------------------
    # Step 2 — Blue-band veto for disagreement pixels
    # Applied BEFORE dilation so corrected cloud info propagates outward.
    #
    # At cloud edges the two products often disagree (one says clear, the
    # other cloudy).  A blue-band threshold of 0.15 provides an independent
    # physical check: surface reflectance above this level in the blue is
    # inconsistent with bare-soil/vegetation and strongly indicates cloud
    # contamination.
    # ------------------------------------------------------------------
    disagree  = mod09_cloud ^ mod35_cloud  # True where masks differ
    blue_veto = disagree & (blue_band > 0.15)

    # Bake the veto into both raw masks so the subsequent dilation step
    # propagates from a more complete starting set of cloudy pixels.
    mod09_cloud = mod09_cloud | blue_veto
    mod35_cloud = mod35_cloud | blue_veto

    # ------------------------------------------------------------------
    # Step 1 — Dilation → union → morphological closing
    # ------------------------------------------------------------------
    kernel = np.ones((5, 5), dtype=bool)  # 5×5 square structuring element

    # Independently dilate each mask to buffer cloud edges by ~2.5 px and
    # absorb the ~1-2 pixel misregistration typical between MOD35 and MOD09.
    mod09_dilated = binary_dilation(mod09_cloud, structure=kernel)
    mod35_dilated = binary_dilation(mod35_cloud, structure=kernel)

    # Union: a pixel is marked cloudy if EITHER product (after dilation) says so.
    combined = mod09_dilated | mod35_dilated

    # Morphological closing (dilation then erosion with the same kernel):
    # fills small interior holes that arise where the two products disagree
    # in the cloud interior, without expanding the outer boundary further.
    cleaned_mask = binary_closing(combined, structure=kernel)  # bool (H, W)

    # ------------------------------------------------------------------
    # Step 3 — Soft cloud confidence score ∈ [0, 1]
    # Three complementary signals are blended; 1 = confident clear.
    # ------------------------------------------------------------------

    # Signal 1 (w=0.5): MOD35 4-level confidence normalised to [0, 1].
    mod35_score = mod35_confidence.astype(np.float32) / 3.0

    # Signal 2 (w=0.3): inverted MOD09 binary flag (pre-veto).
    # Uses the veto-corrected flag so the veto is also reflected in the score.
    mod09_score = (~mod09_cloud).astype(np.float32)

    # Signal 3 (w=0.2): blue-band spectral clarity score.
    # Clip to [0, 0.3], rescale to [0, 1], then invert so low blue → score near 1.
    blue_clipped = np.clip(blue_band, 0.0, 0.3)
    blue_penalty = 1.0 - (blue_clipped / 0.3)  # 1 = clear, 0 = very blue

    soft_score = (
        0.5 * mod35_score +
        0.3 * mod09_score +
        0.2 * blue_penalty
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4 — Meta-mask channel
    # Retain the soft score for clear pixels; zero out cloudy pixels so
    # the model sees a clean "confidence of clearness" signal.
    # ------------------------------------------------------------------
    meta_mask_channel = np.where(cleaned_mask, 0.0, soft_score).astype(np.float32)

    return CloudMaskResult(
        cleaned_mask=cleaned_mask,
        soft_score=soft_score,
        meta_mask_channel=meta_mask_channel,
    )


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class PatchConfig:
    patch_size: int = 256
    stride: int = 256
    min_clear_fraction: float = 0.05


BAND_CONFIG = {
    "mod09_250": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "qc": "QC_250m",
    },
    "mod09_500": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "blue": "sur_refl_b03",
        "state": "state_1km",
    },
}


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def upsample_500_to_250(mask_500):
    return np.repeat(np.repeat(mask_500, 2, axis=0), 2, axis=1)


def resize_to_multiple_of_patch(x, y, patch_size=14):
    _, _, H, W = x.shape
    H_new = (H // patch_size) * patch_size
    W_new = (W // patch_size) * patch_size
    if H_new != H or W_new != W:
        x = x[:, :, :H_new, :W_new]
        if y is not None:
            y = y[:, :H_new, :W_new]
    return x, y


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

def extract_zarr_store(path, band_name, samples:int=None, patches=True):
    store = zarr.open(path, mode="r")
    if patches:
        patches = store["patches"]
    else:
        patches = store

    band = patches[band_name]
    dates = sorted(band.keys())
    grid_ids = sorted(band[dates[0]].keys())
    if samples is not None:
        grid_ids = grid_ids[:samples]

    return {
        "patches": patches,
        "dates": dates,
        "grid_ids": grid_ids,
    }

def extract_modis_cube(path, product: str, samples: int = None):
    store = zarr.open(path, mode="r")
    patches = store["patches"]
    cfg = BAND_CONFIG[product]
    dates = sorted(patches[cfg["red"]].keys())
    grid_ids = sorted(patches[cfg["red"]][dates[0]].keys())
    if samples is not None:
        grid_ids = grid_ids[:samples]
    return {
        "store": store,
        "patches": patches,
        "cfg": cfg,
        "dates": dates,
        "grid_ids": grid_ids,
        "product": product,
    }


# ─────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────

class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.adapter = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.ReLU(),
            nn.Conv2d(bottleneck, dim, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        return x + self.adapter(self.norm(x))


class CloudAdapterModel(nn.Module):
    def __init__(self, backbone, in_channels=3):
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.embed_dim

        # Patch embed: replace only once, here
        old = backbone.patch_embed.proj
        backbone.patch_embed.proj = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
        )

        self.adapter = Adapter(feat_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),  # binary → BCE
        )

    def forward(self, x):
        B, C, H, W = x.shape
        h, w = H // 14, W // 14

        feats = self.backbone.forward_features(x)["x_norm_patchtokens"]
        feats = feats.permute(0, 2, 1).reshape(B, -1, h, w)
        feats = self.adapter(feats)

        logits = self.decoder(feats)  # (B, 1, h, w)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits.squeeze(1)  # (B, H, W)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class MODISZarrPatchDataset(Dataset):
    """
    Parameters
    ----------
    cube_250 : dict
        Output of extract_modis_cube for MOD09GQ (250 m).
    cube_500 : dict
        Output of extract_modis_cube for MOD09GA (500 m).
    use_blue : bool
        Append the blue band (sur_refl_b03, upsampled 2× to 250 m) as an
        extra input channel after red / nir / ndvi.
    use_meta_mask : bool
        Run generate_cloud_mask to produce the meta-mask channel and the
        per-pixel soft confidence score.
        • The meta-mask channel is appended to the input tensor x.
        • soft_score is returned as a third element of each sample so the
          training loop can use it as a per-pixel loss weight instead of the
          hard nan-mask.
        Requires the blue band (loaded automatically even if use_blue=False).
    cube_mod35 : dict | None
        Optional output of extract_mod35_cube.  When provided its
        cloud_probability array is used as the mod35_confidence input to
        generate_cloud_mask.  When None a fully-confident-clear fallback
        (all 3s) is used so the mask degrades gracefully to mod09 + blue.
    """

    def __init__(
        self,
        cube_250,
        cube_500,
        min_clear_fraction=0.05,
        use_blue=False,
        use_meta_mask=False,
        cube_mod35=None,
        start_date=None,
        end_date=None,
    ):
        self.cube_250 = cube_250
        self.cube_500 = cube_500
        self.min_clear_fraction = min_clear_fraction
        self.use_blue = use_blue
        self.use_meta_mask = use_meta_mask
        self.cube_mod35 = cube_mod35

        all_dates = cube_250["dates"]
        self.dates = [
            d for d in all_dates
            if (start_date is None or d >= start_date)
            and (end_date is None or d <= end_date)
        ]
        if not self.dates:
            raise ValueError(
                f"No dates remain after filtering ({start_date} – {end_date}). "
                f"Available range: {all_dates[0]} – {all_dates[-1]}"
            )
        logger.info(f"Dataset: {len(self.dates)} dates ({self.dates[0]} – {self.dates[-1]})")
        self.grid_ids = cube_250["grid_ids"]
        self.index = [
            (t, g)
            for t in range(len(self.dates))
            for g in range(len(self.grid_ids))
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        t, g = self.index[idx]
        date = self.dates[t]
        grid_id = self.grid_ids[g]

        try:
            p250 = self.cube_250["patches"]
            cfg250 = self.cube_250["cfg"]

            red = p250[cfg250["red"]][date][grid_id][:].astype(np.float32) / 10000.0
            nir = p250[cfg250["nir"]][date][grid_id][:].astype(np.float32) / 10000.0
            ndvi = compute_ndvi(nir, red)

            p500 = self.cube_500["patches"]
            cfg500 = self.cube_500["cfg"]

            # state_1km — decode raw QA bits → binary cloud mask (baseline label)
            state_500 = p500[cfg500["state"]][date][grid_id][:].astype(np.uint16)
            cloud_bits = state_500 & 0b11
            mask_500 = np.isin(cloud_bits, [1, 2]).astype(np.float32)
            mask_250 = upsample_500_to_250(mask_500)  # (H, W) float32

            clear_fraction = 1.0 - mask_250.mean()
            if clear_fraction < self.min_clear_fraction:
                return None

            # ------------------------------------------------------------------
            # Blue band (sur_refl_b03) — needed for both use_blue and
            # use_meta_mask so load once whenever either flag is set.
            # ------------------------------------------------------------------
            need_blue = self.use_blue or self.use_meta_mask
            blue_250 = None
            if need_blue:
                blue_500 = p500[cfg500["blue"]][date][grid_id][:].astype(np.float32) / 10000.0
                blue_250 = upsample_500_to_250(blue_500)  # (H, W) float32

            # ------------------------------------------------------------------
            # Meta-cloud mask via generate_cloud_mask
            # ------------------------------------------------------------------
            soft_score = None
            meta_channel = None
            if self.use_meta_mask:
                if self.cube_mod35 is not None:
                    p35 = self.cube_mod35["patches"]
                    band35 = self.cube_mod35["band"]
                    mod35_raw = p35[band35][date][grid_id][:].astype(np.uint8)
                    # MOD35 is stored at the same patch resolution as MOD09GQ
                    # (resampled during download); upsample only if 500 m sized.
                    if mod35_raw.shape != blue_250.shape:
                        mod35_raw = upsample_500_to_250(mod35_raw)
                    mod35_conf = mod35_raw
                else:
                    # Fallback: assume fully confident clear so only mod09 + blue
                    # contribute to the mask.
                    mod35_conf = np.full(blue_250.shape, 3, dtype=np.uint8)

                result = generate_cloud_mask(
                    mod09qa_bits=upsample_500_to_250(state_500),
                    mod35_confidence=mod35_conf,
                    blue_band=blue_250,
                )
                meta_channel = result.meta_mask_channel  # (H, W) float32
                soft_score = result.soft_score            # (H, W) float32

            # ------------------------------------------------------------------
            # Build input tensor x: [red, nir, ndvi, (blue), (meta_mask)]
            # ------------------------------------------------------------------
            channels = [red, nir, ndvi]
            if self.use_blue:
                channels.append(blue_250)
            if self.use_meta_mask:
                channels.append(meta_channel)
            x = np.stack(channels, axis=0)  # (C, H, W)

            x_t = torch.from_numpy(x)
            y_t = torch.from_numpy(mask_250)

            if self.use_meta_mask:
                return x_t, y_t, torch.from_numpy(soft_score)
            return x_t, y_t

        except KeyError:
            return None


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────


class CloudAdapterPipeline:
    def __init__(
        self,
        model_name=("facebookresearch/dinov2", "dinov2_vits14"),
        resampling=Resampling.nearest,
    ):
        self.model_name = model_name
        self._resampling_method = resampling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    # ── data ──────────────────────────────────

    def extract_modis_cube(self, path, product: str, samples: int = None):
        store = zarr.open(path, mode="r")
        patches = store["patches"]
        cfg = BAND_CONFIG[product]

        dates = sorted(patches[cfg["red"]].keys())
        grid_ids = sorted(patches[cfg["red"]][dates[0]].keys())

        if samples is not None:
            grid_ids = grid_ids[:samples]

        return {
            "store": store,
            "patches": patches,
            "cfg": cfg,
            "dates": dates,
            "grid_ids": grid_ids,
            "product": product,
        }

    def extract_mod35_cube(self, path: str, band: str = "cloud_probability", samples: int = None):
        """
        Open a MOD35 zarr store and return a cube dict compatible with
        MODISZarrPatchDataset(cube_mod35=...).

        Parameters
        ----------
        path : str
            Path to the MOD35 zarr store.
        band : str
            Variable name inside patches/ that holds the 4-level confidence
            flag (0 = confident cloudy … 3 = confident clear).
            Defaults to "cloud_probability"; use "Cloud_Mask" if that is what
            was written during download.
        samples : int | None
            Limit to the first N grid_ids (useful for quick tests).
        """
        store = zarr.open(path, mode="r")
        patches = store["patches"]
        dates = sorted(patches[band].keys())
        grid_ids = sorted(patches[band][dates[0]].keys())
        if samples is not None:
            grid_ids = grid_ids[:samples]
        return {
            "store": store,
            "patches": patches,
            "band": band,
            "dates": dates,
            "grid_ids": grid_ids,
        }

    # ── model ─────────────────────────────────

    def _load_fm(self, in_channels=3):
        """Load DINOv2 backbone and wrap in CloudAdapterModel."""
        backbone = torch.hub.load(*self.model_name)
        # patch embed replacement happens inside CloudAdapterModel.__init__
        self.model = CloudAdapterModel(backbone, in_channels=in_channels)

    # ── checkpointing ─────────────────────────

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": loss,
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        """Load weights for resuming training or running inference."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
        return ckpt["epoch"]

    # ── training ──────────────────────────────

    def train(self, epochs=10, checkpoint_dir="checkpoints", checkpoint_every=1):
        logger.info("Starting training...")
        best_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in tqdm(self.loader, desc=f"Epoch {epoch}"):
                if batch is None:
                    continue

                if len(batch) == 3:
                    # use_meta_mask=True: dataset returns (x, y, soft_score)
                    x, y, soft_score = batch
                    x, y = x.to(self.device), y.to(self.device)
                    soft_score = soft_score.to(self.device)
                    x, y = resize_to_multiple_of_patch(x, y, patch_size=14)
                    soft_score = soft_score[:, :y.shape[-2], :y.shape[-1]]
                    weight = soft_score
                else:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    x, y = resize_to_multiple_of_patch(x, y, patch_size=14)
                    weight = np.nan

                self.optimizer.zero_grad()
                logits = self.model(x)                        # (B, H, W)
                loss = self.criterion(logits, y, weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch} | Loss {avg_loss:.4f}")

            # ── periodic checkpoint ──
            if (epoch + 1) % checkpoint_every == 0:
                self.save_checkpoint(
                    f"{checkpoint_dir}/epoch_{epoch:03d}.pt",
                    epoch=epoch,
                    loss=avg_loss,
                )

            # ── best checkpoint ──
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(
                    f"{checkpoint_dir}/best.pt",
                    epoch=epoch,
                    loss=avg_loss,
                )

    # ── inference ─────────────────────────────

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x: np.ndarray (3, H, W)  — red, nir, ndvi
        returns: cloud probability mask (H, W) in [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)  # (1, 3, H, W)
            t, _ = resize_to_multiple_of_patch(t, None, patch_size=14)
            logits = self.model(t)           # (1, H, W)
            probs = torch.sigmoid(logits)    # binary probability
        return probs[0].cpu().numpy()        # (H, W)

    def predict_from_file(self, mod09gq_path: str) -> np.ndarray:
        """Convenience: load a GQ tiff and return cloud probability map."""
        import rasterio
        with rasterio.open(mod09gq_path) as ds:
            red  = ds.read(1).astype(np.float32) / 10000.0
            nir  = ds.read(2).astype(np.float32) / 10000.0
            ndvi = compute_ndvi(nir, red)
        x = np.stack([red, nir, ndvi], axis=0)
        return self.predict(x)

    # ── main entry point ──────────────────────

    def pipeline(
        self,
        path_mod09gq,
        path_mod09ga,
        epochs=10,
        batch_size=16,
        sample=False,
        checkpoint_dir="checkpoints",
        resume_from=None,
        use_blue=False,
        use_meta_mask=False,
        path_mod35=None,
        mod35_band="cloud_probability",
        start_date=None,
        end_date=None,
    ):
        """
        Parameters
        ----------
        use_blue : bool
            Add the blue band (sur_refl_b03) as an extra input channel.
        use_meta_mask : bool
            Add the meta-cloud channel produced by generate_cloud_mask as an
            extra input channel, and use the per-pixel soft_score as the
            per-pixel loss weight instead of the hard nan-mask.
            Implicitly loads the blue band even when use_blue=False.
        path_mod35 : str | None
            Path to a MOD35 zarr store.  When set and use_meta_mask=True, the
            MOD35 4-level confidence is fed to generate_cloud_mask.  When None
            the mask is computed from mod09 state + blue only.
        mod35_band : str
            Variable name inside the MOD35 patches group (default
            "cloud_probability").
        start_date : str | None
            Inclusive lower bound for training dates, "YYYY-MM-DD".
        end_date : str | None
            Inclusive upper bound for training dates, "YYYY-MM-DD".
        """
        from models import masked_bce
        samples = 10 if sample else None

        logger.info("Extracting MODIS cubes...")
        cube_250 = self.extract_modis_cube(path_mod09gq, "mod09_250", samples=samples)
        cube_500 = self.extract_modis_cube(path_mod09ga, "mod09_500", samples=samples)

        cube_mod35 = None
        if use_meta_mask and path_mod35 is not None:
            logger.info("Extracting MOD35 cube...")
            cube_mod35 = self.extract_mod35_cube(path_mod35, band=mod35_band, samples=samples)

        logger.info("Building dataset...")
        dataset = MODISZarrPatchDataset(
            cube_250,
            cube_500,
            min_clear_fraction=0.05,
            use_blue=use_blue,
            use_meta_mask=use_meta_mask,
            cube_mod35=cube_mod35,
            start_date=start_date,
            end_date=end_date,
        )

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=safe_collate,
            persistent_workers=True,
        )

        # Resolve the number of input channels dynamically
        in_channels = 3                        # red, nir, ndvi
        if use_blue:
            in_channels += 1                   # + blue
        if use_meta_mask:
            in_channels += 1                   # + meta_mask_channel
        self.use_meta_mask = use_meta_mask

        logger.info(f"Loading GFM {self.model_name} (in_channels={in_channels})...")
        self._load_fm(in_channels=in_channels)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = masked_bce

        if resume_from is not None:
            logger.info(f"Resuming from {resume_from}")
            self.load_checkpoint(resume_from)

        self.train(epochs=epochs, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    import argparse
    from definitions import DATA_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run on a small sample for testing")
    parser.add_argument("--use_blue", action="store_true", help="Add blue band (sur_refl_b03) as input channel")
    parser.add_argument("--use_meta_mask", action="store_true", help="Add meta-cloud channel and use soft_score loss weights")
    parser.add_argument("--path_mod35", type=str, default=None, help="Path to MOD35 zarr (optional, used with --use_meta_mask)")
    parser.add_argument("--start_date", type=str, default=None, help="Inclusive start date for training, YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default=None, help="Inclusive end date for training, YYYY-MM-DD")
    args = parser.parse_args()

    path_modis_500 = DATA_PATH / "modis" / "MOD09GA_dataset.zarr"
    path_modis_250 = DATA_PATH / "modis" / "MOD09GQ_dataset.zarr"

    cloud_pipe = CloudAdapterPipeline(
        model_name=("facebookresearch/dinov2", "dinov2_vits14"))

    cloud_pipe.pipeline(
        path_modis_250,
        path_modis_500,
        batch_size=64,
        sample=args.sample,
        use_blue=args.use_blue,
        use_meta_mask=args.use_meta_mask,
        path_mod35=args.path_mod35,
        start_date=args.start_date,
        end_date=args.end_date,
    )