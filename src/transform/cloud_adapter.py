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
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        "state": "state_1km",
    },
}


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def compute_ndvi(nir, red, eps=1e-6):
    return (nir - red) / (nir + red + eps)


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
    def __init__(self, cube_250, cube_500, min_clear_fraction=0.05):
        self.cube_250 = cube_250
        self.cube_500 = cube_500
        self.min_clear_fraction = min_clear_fraction

        self.dates = cube_250["dates"]
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
            x = np.stack([red, nir, ndvi], axis=0)  # (3, H, W)

            p500 = self.cube_500["patches"]
            cfg500 = self.cube_500["cfg"]

            # decode raw state bits → binary cloud mask
            state_500 = p500[cfg500["state"]][date][grid_id][:].astype(np.uint16)
            cloud_bits = state_500 & 0b11
            mask_500 = np.isin(cloud_bits, [1, 2]).astype(np.float32)
            mask_250 = upsample_500_to_250(mask_500)  # (H, W) float32

            clear_fraction = 1.0 - mask_250.mean()
            if clear_fraction < self.min_clear_fraction:
                return None

            return (
                torch.from_numpy(x),
                torch.from_numpy(mask_250),  # float32 for BCE
            )

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

                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                x, y = resize_to_multiple_of_patch(x, y, patch_size=14)

                self.optimizer.zero_grad()
                logits = self.model(x)                        # (B, H, W)
                loss = self.criterion(logits, y, np.nan)
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
    ):  
        from models import masked_bce
        logger.info("Extracting MODIS cubes...")
        cube_250 = self.extract_modis_cube(path_mod09gq, "mod09_250", samples=10 if sample else None)
        cube_500 = self.extract_modis_cube(path_mod09ga, "mod09_500", samples=10 if sample else None)

        logger.info("Building dataset...")
        dataset = MODISZarrPatchDataset(cube_250, cube_500, min_clear_fraction=0.05)

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=safe_collate,
            persistent_workers=True,
        )

        logger.info(f"Loading GFM {self.model_name}...")
        self._load_fm(in_channels=3)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run on a small sample for testing")
    args = parser.parse_args()

    import xarray as xr
    from definitions import DATA_PATH

    path_modis_500 = DATA_PATH / "modis" / "MOD09GA_dataset.zarr"
    path_modis_250 = DATA_PATH / "modis" / "MOD09GQ_dataset.zarr"

    cloud_pipe = CloudAdapterPipeline(
        model_name=("facebookresearch/dinov2", "dinov2_vits14"))
    
    cloud_pipe.pipeline(
        path_modis_250, 
        path_modis_500,
        batch_size=64,
        sample = args.sample
        )