import logging
import random
from matplotlib import dates
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
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.transforms.functional as TF

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
    denom = nir + red
    denom = np.where(np.abs(denom) < eps, eps, denom)
    ndvi = (nir - red) / denom
    
    return ndvi

def normalize_modis(x, reverse=False):
    if not reverse:
        x[:, 0] = (x[:, 0] - 0.5) / 0.5
        x[:, 1] = (x[:, 1] - 0.5) / 0.5
        x[:, 2] = np.clip(x[:, 2], -1, 1)
        x = np.nan_to_num(x, nan=-1.0, posinf=1.0, neginf=-1.0)
        
    else:
        x[:, 0] = x[:, 0] * 0.5 + 0.5
        x[:, 1] = x[:, 1] * 0.5 + 0.5

    return x

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


def extract_image(path, product,  t, g, normalize=False):
    store = zarr.open(path, mode="r")
    patches = store["patches"]
    cfg = BAND_CONFIG[product]
    dates = sorted(patches[cfg["red"]].keys())
    grid_ids = sorted(patches[cfg["red"]][dates[0]].keys())
    date = dates[t]
    grid_id = grid_ids[g]
    red = patches[cfg["red"]][date][grid_id][:].astype(np.float32) / 10000.0
    nir = patches[cfg["nir"]][date][grid_id][:].astype(np.float32) / 10000.0
    ndvi = compute_ndvi(nir, red)
    x = np.stack([red, nir, ndvi], axis=0)  #
    if normalize:
        x = normalize_modis(x)
    
    return x


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


class DecoderBlock(nn.Module):
    """Conv → BN → ReLU × 2, with optional channel projection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CloudAdapterModel(nn.Module):
    def __init__(self, backbone, in_channels=3):
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.embed_dim

        # ── patch embed replacement ────────────────
        old = backbone.patch_embed.proj
        backbone.patch_embed.proj = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
        )

        self.adapter = Adapter(feat_dim)

        # ── decoder ───────────────────────────────
        # DINOv2 ViT-S/14 → feat_dim=384, tokens are H/14 × W/14
        # We upsample in 3 steps: ×2 → ×2 → ×2 (gets us to H/1.75, close enough, 
        # final interpolate covers the remainder cleanly)
        self.decoder = nn.Sequential(
            # Step 1: project backbone dim down
            nn.Conv2d(feat_dim, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DecoderBlock(256, 128),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DecoderBlock(128, 64),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DecoderBlock(64, 32),
        )

        # Final 1×1 logit head — no BN before logits
        self.head = nn.Conv2d(32, 1, 1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for m in [self.decoder, self.up1, self.up2, self.up3, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h, w = H // 14, W // 14

        feats_dict = self.backbone.forward_features(x)

        if torch.isnan(feats_dict["x_norm_patchtokens"]).any():
            logger.warning("NaNs after backbone!")
            return torch.zeros(B, H, W, device=x.device)

        feats = feats_dict["x_norm_patchtokens"]                  # (B, h*w, feat_dim)
        feats = feats.permute(0, 2, 1).reshape(B, -1, h, w)       # (B, feat_dim, h, w)
        feats = self.adapter(feats)

        x = self.decoder(feats)   # (B, 256, h, w)
        x = self.up1(x)           # (B, 128, h*2, w*2)
        x = self.up2(x)           # (B, 64,  h*4, w*4)
        x = self.up3(x)           # (B, 32,  h*8, w*8)

        logits = self.head(x)     # (B, 1, h*8, w*8)

        # Final resize to exact input resolution
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        return logits.squeeze(1)  # (B, H, W)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class MODISZarrPatchDataset(Dataset):
    def __init__(self, cube_250, cube_500, min_clear_fraction=0.05, dates=None, augment=False):
        self.cube_250 = cube_250
        self.cube_500 = cube_500
        self.min_clear_fraction = min_clear_fraction
        self.augment = augment

        self.dates = dates if dates is not None else cube_250["dates"]
        logger.info(f"Using {len(self.dates)} dates for dataset")
        self.grid_ids = cube_250["grid_ids"]
        self.index = [
            (t, g)
            for t in range(len(self.dates))
            for g in range(len(self.grid_ids))
        ]

    def __len__(self):
        return len(self.index)

    def _augment(self, x, y):
        # horizontal flip
        if random.random() > 0.5:
            x = TF.hflip(x)
            y = TF.hflip(y)

        # vertical flip
        if random.random() > 0.5:
            x = TF.vflip(x)
            y = TF.vflip(y)

        # 90° rotation — exact, no interpolation artifacts
        k = random.randint(0, 3)
        x = torch.rot90(x, k, dims=[-2, -1])
        y = torch.rot90(y, k, dims=[-2, -1])

        return x, y

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
            x = normalize_modis(x)

            p500 = self.cube_500["patches"]
            cfg500 = self.cube_500["cfg"]

            # decode raw state bits → binary cloud mask
            state_500 = p500[cfg500["state"]][date][grid_id][:].astype(np.uint16)
            cloud_bits = state_500 & 0b11
            mask_500 = np.isin(cloud_bits, [1, 2]).astype(np.float32)
            mask_250 = upsample_500_to_250(mask_500)  # (H, W) float32

            # clear_fraction = 1.0 - mask_250.mean()
            # if clear_fraction < self.min_clear_fraction:
            #     return None
            x = torch.from_numpy(x)
            y = torch.from_numpy(mask_250)

            if self.augment:
                x, y = self._augment(x, y)

            return x, y

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
        self._load_fm(in_channels=3)
        self.dataset = None  # Will be set in pipeline() method

    def _start_logging(self, checkpoint_dir):
        run_id = datetime.now().strftime("%Y%m%d_%H%M")
        self.output_dir = Path(checkpoint_dir) / f"run_{run_id}"

        # Create organized folder structure
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints" / "periodic").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints" / "best").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Store for checkpointing
        self.checkpoint_dir = self.output_dir / "checkpoints"


    # ── model ─────────────────────────────────

    def _load_fm(self, in_channels=3):
        """Load DINOv2 backbone and wrap in CloudAdapterModel."""
        backbone = torch.hub.load(*self.model_name)
        # patch embed replacement happens inside CloudAdapterModel.__init__
        self.model = CloudAdapterModel(backbone, in_channels=in_channels).to(self.device)

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

    def save_checkpoint_to_folder(self, epoch: int, loss: float, checkpoint_type: str = "periodic"):
        """
        Save checkpoint to organized folder structure.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            checkpoint_type: Type of checkpoint ("periodic" or "best")
        """
        if checkpoint_type == "periodic":
            ckpt_path = self.checkpoint_dir / "periodic" / f"epoch_{epoch:03d}.pt"
        else:
            ckpt_path = self.checkpoint_dir / "best" / f"best_model.pt"
        
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": loss,
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved → {ckpt_path}")
        return ckpt_path

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

    def train(self, epochs=10, checkpoint_every=1):
        logger.info("Starting training...")
        best_loss = float("inf")
        
        # FIX: Add early stopping
        patience = 10
        patience_counter = 0

        # Track losses for learning curves
        train_losses = []
        val_losses = []

        # Track validation metrics
        val_metrics_history = {"iou": [], "precision": [], "recall": []}

        for epoch in range(epochs):
            # ── Training phase ──
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # FIX: Keep backbone frozen throughout training
            # Unfreezing too early causes catastrophic forgetting
            # if epoch == 3:
            #     for p in self.model.backbone.parameters():
            #         p.requires_grad = True

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]"):
                if batch is None:
                    continue

                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                if torch.isnan(x).any():
                    logger.warning("NaNs in input!")
                    continue

                x, y = resize_to_multiple_of_patch(x, y, patch_size=14)
                x = torch.nan_to_num(x, nan=-1.0) 

                self.optimizer.zero_grad()
                logits = self.model(x)  # (B, H, W)
                if torch.isnan(logits).any():
                    logger.warning("NaNs in logits!")
                    continue

                loss = self.criterion(logits, y, np.nan, pos_weight=self.pos_weight)
                if torch.isnan(loss):
                    logger.warning("NaN loss!")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # ── Validation phase ──
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            metrics_accum = {"iou": 0.0, "precision": 0.0, "recall": 0.0}

            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                    if batch is None:
                        continue

                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    x, y = resize_to_multiple_of_patch(x, y, patch_size=14)
                    x = torch.nan_to_num(x, nan=-1.0) 

                    logits = self.model(x)
                    loss = self.criterion(logits, y, np.nan, pos_weight=self.pos_weight)

                    val_loss += loss.item()
                    val_batches += 1

                    # Compute IoU / precision / recall
                    batch_metrics = self.compute_metrics(logits, y)
                    for k in metrics_accum:
                        metrics_accum[k] += batch_metrics[k]

            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)

            avg_metrics = {k: metrics_accum[k] / max(val_batches, 1) for k in metrics_accum}
            for k in avg_metrics:
                val_metrics_history[k].append(avg_metrics[k])

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"IoU: {avg_metrics['iou']:.4f} | "
                f"Precision: {avg_metrics['precision']:.4f} | "
                f"Recall: {avg_metrics['recall']:.4f}"
            )

            # ── Plot learning curves ──
            self.plot_learning_curve(
                train_losses,
                val_losses,
                val_metrics_history=val_metrics_history,
                save_path=f"{self.output_dir}/plots/learning_curves.png",
            )

            # ── Periodic checkpoint ──
            if (epoch + 1) % checkpoint_every == 0:
                self.save_checkpoint_to_folder(
                    epoch=epoch,
                    loss=avg_val_loss,
                    checkpoint_type="periodic"
                )

            # ── Best checkpoint with early stopping ──
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                self.save_checkpoint_to_folder(
                    epoch=epoch,
                    loss=avg_val_loss,
                    checkpoint_type="best"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # ── Plot sample predictions ──
            if n_batches > 0 and (epoch + 1) % max(1, checkpoint_every) == 0:
                random_idx = np.random.randint(0, len(x))
                x_sample = normalize_modis(x[random_idx], reverse=True)  # (3, H, W)
                y_sample = y[random_idx]  # (H, W)
                y_pred_sample = torch.sigmoid(logits[random_idx])  # (H, W)

                self.plot_sample(
                    x_sample,
                    y_sample,
                    y_pred_sample,
                    save_path=f"{self.output_dir}/samples/epoch_{epoch:03d}.png"
                )
        # ── inference ─────────────────────────────

    def compute_metrics(self, preds, labels, threshold=0.5, null_val=float('nan')):
        if preds.dim() == 4:
            preds = preds.squeeze(1)
        probs = torch.sigmoid(preds)
        pred_mask = (probs > threshold).float()

        if torch.isnan(labels).any():
            mask = ~torch.isnan(labels)
        else:
            mask = torch.ones_like(labels).bool()

        pred_mask = pred_mask[mask]
        labels = labels[mask]

        tp = (pred_mask * labels).sum()
        fp = (pred_mask * (1 - labels)).sum()
        fn = ((1 - pred_mask) * labels).sum()

        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        return {"iou": iou.item(), "precision": precision.item(), "recall": recall.item()}

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
        return probs.squeeze(0).cpu().numpy()  # (H, W)

    def predict_from_file(self, mod09gq_path: str) -> np.ndarray:
        """Convenience: load a GQ tiff and return cloud probability map."""
        import rasterio
        with rasterio.open(mod09gq_path) as ds:
            red  = ds.read(1).astype(np.float32) / 10000.0
            nir  = ds.read(2).astype(np.float32) / 10000.0
            ndvi = compute_ndvi(nir, red)
        x = np.stack([red, nir, ndvi], axis=0)
        x = normalize_modis(x)
        return self.predict(x)

    def predict_zarr(self, cube_250, t, g):
        """
        Predict cloud mask for a given time/grid from Zarr
        """
        p250 = cube_250["patches"]
        cfg = cube_250["cfg"]

        date = cube_250["dates"][t]
        grid_id = cube_250["grid_ids"][g]

        red = p250[cfg["red"]][date][grid_id][:].astype(np.float32) / 10000.0
        nir = p250[cfg["nir"]][date][grid_id][:].astype(np.float32) / 10000.0
        ndvi = compute_ndvi(nir, red)
        
        x = np.stack([red, nir, ndvi], axis=0)
        x = normalize_modis(x)

        return self.predict(x)

    def plot_learning_curve(self, train_losses, val_losses, save_path, val_metrics_history=None):
        """
        Plot training and validation loss curves, and optionally IoU/precision/recall.

        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            save_path: Path to save the plot
            val_metrics_history: Optional dict with keys 'iou', 'precision', 'recall' containing lists per epoch
        """
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 6))

        # Plot losses
        plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
        if val_losses:
            plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)

        # Plot metrics if provided
        if val_metrics_history is not None:
            if "iou" in val_metrics_history:
                plt.plot(epochs, val_metrics_history["iou"], 'g-^', label='Validation IoU', linewidth=2)
            if "precision" in val_metrics_history:
                plt.plot(epochs, val_metrics_history["precision"], 'm-<', label='Validation Precision', linewidth=2)
            if "recall" in val_metrics_history:
                plt.plot(epochs, val_metrics_history["recall"], 'c->', label='Validation Recall', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Training and Validation Loss / Metrics Over Epochs', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to {save_path}")
        plt.close()

    def plot_sample(self, x, y_true, y_pred, save_path):
        """
        x: (3, H, W)
        y_true: (H, W)
        y_pred: (H, W)
        """
        ndvi = x[-1].cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Show input NDVI and add colorbar
        im0 = axs[0].imshow(ndvi, cmap="Greens")
        axs[0].set_title("Input NDVI")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)  # fraction and pad control size/spacing

        # Ground truth
        axs[1].imshow(y_true, cmap="gray")
        axs[1].set_title("Ground Truth")

        # Prediction
        axs[2].imshow(y_pred, cmap="viridis")
        axs[2].set_title("Prediction")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

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
        num_workers=4,
    ):  
        from models import masked_bce
        
        logger.info("Extracting MODIS cubes...")
        cube_250 = extract_modis_cube(path_mod09gq, "mod09_250", samples=200 if sample else None)
        cube_500 = extract_modis_cube(path_mod09ga, "mod09_500", samples=200 if sample else None)

        logger.info("Building dataset...")

        split_idx = int(0.8 * len(cube_250["dates"]))
        train_dates = cube_250["dates"][:split_idx]
        val_dates = cube_250["dates"][split_idx:]

        logger.info("Creating training dataset")
        train_dataset = MODISZarrPatchDataset(cube_250, cube_500, min_clear_fraction=0.05, dates=train_dates, augment=True)
        logger.info("Creating validation dataset")
        val_dataset = MODISZarrPatchDataset(cube_250, cube_500, min_clear_fraction=0.05, dates=val_dates)

        persistent_workers = True if num_workers > 0 else False

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=safe_collate,
            persistent_workers=persistent_workers,
        )
        
        # Create validation loader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=safe_collate,
            persistent_workers=persistent_workers,
        )

        logger.info(f"Loading GFM {self.model_name}...")

        self.optimizer = torch.optim.AdamW([
            {"params": self.model.adapter.parameters(), "lr": 1e-4},
            {"params": self.model.decoder.parameters(), "lr": 1e-4},
            {"params": self.model.backbone.parameters(), "lr": 1e-6},  # FIX: Reduced from 1e-5
        ])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = masked_bce

        if resume_from is not None:
            logger.info(f"Resuming from {resume_from}")
            self.load_checkpoint(resume_from)

        self._start_logging(checkpoint_dir)
        self.pos_weight = torch.tensor(2.0, device=self.device)  # FIX: Reduced from 5.0
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        self.train(epochs=epochs)


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run on a small sample for testing")
    parser.add_argument("--num_workers", type=int, help="Number of DataLoader workers", default=os.getenv("NUM_WORKERS", 4) )
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=os.getenv("BATCH_SIZE", 16) )
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=os.getenv("EPOCHS", 10) )
    args = parser.parse_args()

    from utils import init_logging
    from definitions import DATA_PATH
    logger = init_logging("cloud_adapter.log")

    path_modis_500 = DATA_PATH / "modis/data/modis" / "MOD09GA_dataset.zarr"
    path_modis_250 = DATA_PATH / "modis/data/modis" / "MOD09GQ_dataset.zarr"

    cloud_pipe = CloudAdapterPipeline(
        model_name=("facebookresearch/dinov2", "dinov2_vits14"))
    
    cloud_pipe.pipeline(
        path_modis_250, 
        path_modis_500, 
        batch_size=args.batch_size,
        sample=args.sample,
        epochs=args.epochs,
        checkpoint_dir="checkpoints",
        num_workers=args.num_workers,
        )