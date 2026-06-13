import os
import math
import argparse
import gc
from torch.utils.data import DataLoader
import torch.distributed as dist

import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import xarray as xr

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from models import (
    VIIRSData,
    ERA5Daily,
    StaticLayer,
    XarraySpatioTemporalDataset,
    DengueDataset,
    collate_skip_none
)
from definitions import DATA_PATH, ROOT_DIR, SM_DATA_PATH, LC_DATA_PATH
from models.model_utils import (
    EarlyStopping,
    MaskedAdmin2Loss,
    nan_checks_replace,
    standardize_tensor,
    export_batches,
    SoilMoistureData,
    LandCoverData,
    GPWv4Population,
)
from models.transformer import DenguePredictor
from modis_majortom.utils import latin_box, init_logging


# ------------------------------------------------------------------
# VAE MODEL
# ------------------------------------------------------------------


class InputUpsampler(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(size=(128, 128), mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class OutputDownsampler(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((86, 86)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class VAEWithTrainableResize(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.pre = InputUpsampler(in_channels=1, hidden_channels=16, out_channels=1)
        self.vae = vae
        self.post = OutputDownsampler(in_channels=1, hidden_channels=16, out_channels=1)

    def forward(self, x):
        x_up = self.pre(x)
        enc = self.vae.encode(x_up)
        posterior = enc.latent_dist
        z = posterior.rsample() if hasattr(posterior, "rsample") else posterior.sample()
        recon_up = self.vae.decode(z).sample
        recon = self.post(recon_up)
        return recon, posterior.mean, posterior.logvar


# ------------------------------------------------------------------
# SHARED UTILS
# ------------------------------------------------------------------


def plot_learning_curves(train_losses, val_losses, plot_dir, logger):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = plot_dir / "learning_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Learning curves saved to {plot_path}")
    plt.close()


def save_prediction_snapshot(model, snapshot_batch, device, num_zones, plot_dir,
                              epoch, cfg, logger):
    """
    Save a 3-panel prediction map for one fixed reference batch.
    Called every `snapshot_every` epochs so you can watch predictions evolve.
    Panels: pixel log-rate | zone-level predicted rate | zone-level actual cases.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_poisson = cfg.get("loss_fn", "mse") == "poisson"
    add_sm = cfg.get("add_sm", False)
    add_lc = cfg.get("add_lc", False)

    _items = [t.to(device) for t in snapshot_batch]
    x_high, x_med, x_static, x_cond = _items[0], _items[1], _items[2], _items[3]
    y_batch = _items[-1]
    x_sm = x_lc = None
    if add_sm and add_lc:
        x_sm, x_lc = _items[4], _items[5]
    elif add_sm:
        x_sm = _items[4]
    elif add_lc:
        x_lc = _items[4]

    x_high   = torch.nan_to_num(x_high,   nan=0.0)
    x_med    = torch.nan_to_num(x_med,    nan=0.0)
    x_static = torch.nan_to_num(x_static, nan=0.0)
    if x_sm is not None:
        x_sm = torch.nan_to_num(x_sm, nan=0.0)

    m = model.module if hasattr(model, "module") else model
    m.eval()
    with torch.no_grad():
        pred = m(x_high, x_med, x_static, x_cond, x_sm, x_lc)   # [B, 1, H, W]

    pred_cpu = pred[0, 0].cpu().float()              # [H, W]
    zone_map = x_cond[0].squeeze().long().cpu()      # [H, W]
    H, W     = zone_map.shape
    valid_zm = (zone_map >= 0) & (zone_map < num_zones)
    zm_flat  = zone_map[valid_zm].view(-1)
    pv_flat  = pred_cpu[valid_zm].view(-1)

    # Zone-level predicted rates via logsumexp (consistent with loss)
    zone_max = torch.full((num_zones,), float('-inf'))
    with torch.no_grad():
        zone_max.scatter_reduce_(0, zm_flat, pv_flat, reduce='amax', include_self=True)
    exp_sum = torch.zeros(num_zones)
    exp_sum.scatter_add_(0, zm_flat, torch.exp(pv_flat - zone_max[zm_flat]))
    zone_pred_rates = torch.exp(zone_max + torch.log(exp_sum + 1e-8))

    # Zone-level actual cases
    if is_poisson:
        zone_actual = y_batch[0].cpu().float()       # [num_zones]
    else:
        y_2d      = y_batch[0, 0].cpu().float()
        z_sums    = torch.zeros(num_zones)
        z_cnt     = torch.zeros(num_zones)
        valid_y   = valid_zm & torch.isfinite(y_2d)
        if valid_y.any():
            zm2 = zone_map[valid_y].view(-1)
            yv2 = y_2d[valid_y].view(-1)
            z_sums.scatter_add_(0, zm2, yv2)
            z_cnt.scatter_add_(0, zm2, torch.ones_like(yv2))
        zone_actual = torch.where(z_cnt > 0, z_sums / z_cnt,
                                  torch.full_like(z_sums, float('nan')))

    def zone_to_img(values):
        img = np.full((H, W), np.nan, dtype=np.float32)
        zm_np = zone_map.numpy()
        mask  = (zm_np >= 0) & (zm_np < num_zones)
        img[mask] = values.numpy()[zm_np[mask]]
        return img

    pred_img        = pred_cpu.numpy()
    zone_pred_img   = zone_to_img(zone_pred_rates)
    zone_actual_img = zone_to_img(zone_actual)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Epoch {epoch:03d} — prediction snapshot", fontsize=12)

    ax = axes[0]
    im = ax.imshow(pred_img, cmap="YlOrRd", origin="upper")
    ax.set_title("Pixel log-rate (model output)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1]
    vp = np.nanpercentile(zone_pred_img, 99) if np.any(np.isfinite(zone_pred_img)) else 1.0
    im = ax.imshow(zone_pred_img, cmap="YlOrRd", origin="upper", vmin=0, vmax=vp)
    ax.set_title("Predicted rate (zone level)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[2]
    va = np.nanpercentile(zone_actual_img, 99) if np.any(np.isfinite(zone_actual_img)) else 1.0
    im = ax.imshow(zone_actual_img, cmap="Blues", origin="upper", vmin=0, vmax=va)
    ax.set_title("Actual cases (zone level)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    plt.tight_layout()
    snap_dir = plot_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    out_path = snap_dir / f"snapshot_epoch_{epoch:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Prediction snapshot saved to {out_path}")


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    return start_epoch, ckpt


def ensure_bchw_1x86x86(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input is not a torch.Tensor, got: {type(x)}")

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        if x.shape[0] == 1:
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)
    elif x.ndim == 4:
        pass
    elif x.ndim == 5:
        if x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.shape[2] == 1:
            x = x.squeeze(2)
        else:
            raise ValueError(f"Incompatible 5D input shape: {tuple(x.shape)}")
    else:
        raise ValueError(f"Unsupported number of dimensions: {tuple(x.shape)}")

    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor after reshape, got: {tuple(x.shape)}")

    if x.shape[1] != 1:
        raise ValueError(
            f"Expected a single channel, but got shape {tuple(x.shape)}. "
            f"C={x.shape[1]} instead of 1."
        )

    if x.shape[2] != 86 or x.shape[3] != 86:
        raise ValueError(f"Expected H=W=86, but got shape {tuple(x.shape)}")

    return x


def normalize_to_neg_one_one(x, data_min, data_max):
    denom = torch.clamp(torch.tensor(data_max - data_min, device=x.device), min=1e-6)
    return 2.0 * (x - data_min) / denom - 1.0


def denormalize_from_neg_one_one(x, data_min, data_max):
    data_min = torch.as_tensor(data_min, device=x.device, dtype=x.dtype)
    data_max = torch.as_tensor(data_max, device=x.device, dtype=x.dtype)
    return 0.5 * (x + 1.0) * (data_max - data_min) + data_min


def compute_global_minmax(loader, device):
    global_min = None
    global_max = None

    for batch in tqdm(loader, desc="Compute global min/max", leave=False):
        if batch is None:
            continue

        _, _, _, _, y_batch = [b.to(device, non_blocking=True) for b in batch]
        y_batch = ensure_bchw_1x86x86(y_batch)

        valid = y_batch[~torch.isnan(y_batch)]
        if valid.numel() == 0:
            continue

        batch_min = valid.min()
        batch_max = valid.max()

        if global_min is None:
            global_min = batch_min
            global_max = batch_max
        else:
            global_min = torch.minimum(global_min, batch_min)
            global_max = torch.maximum(global_max, batch_max)

    return global_min, global_max


def masked_mse_loss(pred, target, valid_mask):
    if pred.shape != target.shape or pred.shape != valid_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred.shape}, target={target.shape}, mask={valid_mask.shape}"
        )

    diff2 = (pred - target) ** 2
    diff2 = diff2[valid_mask]

    if diff2.numel() == 0:
        return pred.sum() * 0.0

    return diff2.mean()


# ------------------------------------------------------------------
# MODEL FACTORY
# ------------------------------------------------------------------


def build_model(cfg, num_zones):
    """Return (model, compute_loss_fn) for the requested cfg["model"]."""

    if cfg["model"] == "vae":
        vae = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            latent_channels=cfg["latent_channels"],
            down_block_types=("DownEncoderBlock2D",) * 3,
            up_block_types=("UpDecoderBlock2D",) * 3,
            block_out_channels=tuple(cfg["block_out_channels"]),
            layers_per_block=cfg["layers_per_block"],
            norm_num_groups=cfg["norm_num_groups"],
        )
        model = VAEWithTrainableResize(vae)

        beta_kl = cfg["beta_kl"]
        use_kl = cfg["use_kl"]

        def compute_loss_fn(model, x_high, x_med, x_static, x_cond, y_batch, valid_mask):
            recon, posterior_mean, posterior_logvar = model(y_batch)
            if recon.shape != y_batch.shape:
                raise ValueError(f"Shape mismatch: recon {recon.shape} vs y_batch {y_batch.shape}")
            recon_loss = masked_mse_loss(recon, y_batch, valid_mask)
            kl_loss = -0.5 * torch.mean(
                1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp()
            )
            loss = recon_loss + beta_kl * kl_loss if use_kl else recon_loss
            return loss, {"recon": recon_loss.item(), "kl": kl_loss.item()}

    elif cfg["model"] == "transformer":
        model = DenguePredictor(
            med_in_ch=cfg["med_in_ch"],
            num_zones=num_zones,
            swin_model=cfg.get("swin_model", "swin_tiny_patch4_window7_224.ms_in1k"),
            use_titok=cfg.get("use_titok", False),
            titok_backbone=cfg.get("titok_backbone", "vit_base_patch16_224.mae"),
            titok_num_latent_tokens=cfg.get("titok_num_latent_tokens", 32),
            add_sm=cfg.get("add_sm", False),
            add_lc=cfg.get("add_lc", False),
        )

        if cfg.get("loss_fn", "mse") == "poisson":
            _criterion = MaskedAdmin2Loss(
                loss_fn=nn.PoissonNLLLoss(log_input=True, reduction='mean'),
                num_zones=num_zones,
                device=cfg.get("device", "cpu"),
            )

            def compute_loss_fn(model, x_high, x_med, x_static, x_cond, y_batch, valid_mask, x_sm=None, x_lc=None, pop_map=None):
                x_high   = torch.nan_to_num(x_high,   nan=0.0)
                x_med    = torch.nan_to_num(x_med,    nan=0.0)
                x_static = torch.nan_to_num(x_static, nan=0.0)
                if x_sm is not None:
                    x_sm = torch.nan_to_num(x_sm, nan=0.0)
                pred = model(x_high, x_med, x_static, x_cond, x_sm, x_lc)
                # x_cond is [B, 1, 1, H, W] or [B, 1, H, W] — flatten to [B, H, W]
                zone_map = x_cond.view(x_cond.shape[0], x_cond.shape[-2], x_cond.shape[-1]).long()
                loss = _criterion.zone_loss(pred, y_batch, zone_map, pop_map=pop_map)
                # Bias metrics: integer-rounded predicted counts vs actual, no gradient
                with torch.no_grad():
                    log_rates, zone_counts = _criterion.zone_aggregate(pred, zone_map, pop_map)
                    zone_pred_rates = torch.exp(log_rates)
                    valid = (zone_counts > 0) & torch.isfinite(y_batch)
                    if valid.any():
                        pred_int = zone_pred_rates[valid].round()
                        actual   = y_batch[valid]
                        mae  = (pred_int - actual).abs().mean().item()
                        bias = (pred_int - actual).mean().item()
                    else:
                        mae = bias = float("nan")
                return loss, {"recon": loss.item(), "kl": 0.0, "mae": mae, "bias": bias}
        else:
            def compute_loss_fn(model, x_high, x_med, x_static, x_cond, y_batch, valid_mask, x_sm=None, x_lc=None, pop_map=None):
                # Replace NaN in inputs (cloud/nodata pixels, out-of-region ERA5) with 0
                # before forward pass — NaN propagates through Swin attention and BatchNorm.
                x_high   = torch.nan_to_num(x_high,   nan=0.0)
                x_med    = torch.nan_to_num(x_med,    nan=0.0)
                x_static = torch.nan_to_num(x_static, nan=0.0)
                if x_sm is not None:
                    x_sm = torch.nan_to_num(x_sm, nan=0.0)
                pred = model(x_high, x_med, x_static, x_cond, x_sm, x_lc)
                if pred.shape != y_batch.shape:
                    raise ValueError(f"Shape mismatch: pred {pred.shape} vs y_batch {y_batch.shape}")
                loss = masked_mse_loss(pred, y_batch, valid_mask)
                return loss, {"recon": loss.item(), "kl": 0.0}

    else:
        raise ValueError(f"Unknown model '{cfg['model']}'. Choose 'vae' or 'transformer'.")

    return model, compute_loss_fn


# ------------------------------------------------------------------
# TRAIN / EVAL
# ------------------------------------------------------------------


def train_one_epoch(model, compute_loss_fn, train_loader, optimizer, device,
                    data_min, data_max, scaler=None, cfg=None, logger=None, is_ddp=False):
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    n_batches = 0
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1) if cfg else 1))
    is_poisson = (cfg.get("loss_fn", "mse") == "poisson") if cfg else False

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0 and logger is not None:
            logger.info("[train_one_epoch] First batch received from DataLoader")
        start = time.time()

        # Synchronize skip decision across DDP ranks.
        skip = (batch is None)
        x_high = x_med = x_static = x_cond = x_sm = x_lc = pop_map = y_batch = valid_mask = None
        add_sm  = (cfg.get("add_sm",  False) if cfg else False)
        add_lc  = (cfg.get("add_lc",  False) if cfg else False)
        add_pop = (cfg.get("add_pop", False) if cfg else False)
        if not skip:
            items = [b.to(device, non_blocking=True) for b in batch]
            # pop_map is always second-to-last when add_pop is True
            if add_pop:
                pop_map = items[-2]
                items = items[:-2] + [items[-1]]
            if add_sm and add_lc:
                x_high, x_med, x_static, x_cond, x_sm, x_lc, y_batch = items
            elif add_sm:
                x_high, x_med, x_static, x_cond, x_sm, y_batch = items
            elif add_lc:
                x_high, x_med, x_static, x_cond, x_lc, y_batch = items
            else:
                x_high, x_med, x_static, x_cond, y_batch = items
            if is_poisson:
                # y_batch is [B, num_zones] — zone-level counts from DengueDataset
                valid_mask = None
                if not torch.isfinite(y_batch).any():
                    skip = True
            else:
                y_batch = ensure_bchw_1x86x86(y_batch)
                valid_mask = (~torch.isnan(y_batch)).bool()
                if not valid_mask.any():
                    skip = True
        if is_ddp:
            skip_t = torch.tensor(int(skip), dtype=torch.int32, device=device)
            dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
            skip = skip_t.item() > 0
        if skip:
            continue

        if not is_poisson:
            y_batch = nan_checks_replace([y_batch], replace_nan=-2.0)[0]
            y_batch = normalize_to_neg_one_one(y_batch, data_min, data_max)

        amp_device = cfg.get("device", "cuda") if cfg else "cuda"
        amp_enabled = cfg.get("amp", True) if cfg else True
        with torch.amp.autocast(enabled=amp_enabled, device_type=amp_device):
            loss, metrics = compute_loss_fn(
                model, x_high, x_med, x_static, x_cond, y_batch, valid_mask,
                x_sm=x_sm, x_lc=x_lc, pop_map=pop_map,
            )

            if torch.isnan(loss):
                if logger:
                    logger.warning("[train] Loss is NaN, skipping this batch")
                continue

            scaled_loss = loss / grad_accum_steps
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = y_batch.size(0)
        n_batches += bs
        running_loss += loss.item() * bs
        running_recon += metrics["recon"] * bs
        running_kl += metrics["kl"] * bs

        del x_high, x_med, x_static, x_cond, x_sm, x_lc, pop_map, y_batch, loss, valid_mask

        cleanup_every = int(cfg.get("memory_cleanup_interval", 0) if cfg else 0)
        if cleanup_every > 0 and (batch_idx + 1) % cleanup_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rank_id = dist.get_rank() if dist.is_initialized() else 0
        print(f"Rank {rank_id} batch time: {time.time() - start:.3f}s")

    # Handle trailing gradients from incomplete accumulation window.
    if n_batches > 0 and (n_batches % grad_accum_steps) != 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if is_ddp:
        world_size = dist.get_world_size()
        count_tensor = torch.tensor(float(n_batches), device=device)
        dist.all_reduce(count_tensor)
        if count_tensor.item() == 0:
            return {"loss": float("nan"), "recon": float("nan"), "kl": float("nan")}
        loss_t = torch.tensor(running_loss / max(n_batches, 1), device=device)
        recon_t = torch.tensor(running_recon / max(n_batches, 1), device=device)
        kl_t = torch.tensor(running_kl / max(n_batches, 1), device=device)
        dist.all_reduce(loss_t)
        dist.all_reduce(recon_t)
        dist.all_reduce(kl_t)
        return {
            "loss": loss_t.item() / world_size,
            "recon": recon_t.item() / world_size,
            "kl": kl_t.item() / world_size,
        }

    if n_batches == 0:
        return {"loss": float("nan"), "recon": float("nan"), "kl": float("nan")}

    return {
        "loss": running_loss / n_batches,
        "recon": running_recon / n_batches,
        "kl": running_kl / n_batches,
    }


@torch.inference_mode()
def evaluate_one_epoch(model, compute_loss_fn, loader, device, data_min, data_max,
                       desc="Eval", cfg=None, logger=None, is_ddp=False):
    model.eval()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    running_mae = 0.0
    running_bias = 0.0
    n_batches = 0
    n_mae_batches = 0
    is_poisson = (cfg.get("loss_fn", "mse") == "poisson") if cfg else False

    add_sm  = (cfg.get("add_sm",  False) if cfg else False)
    add_lc  = (cfg.get("add_lc",  False) if cfg else False)
    add_pop = (cfg.get("add_pop", False) if cfg else False)
    for batch_idx, batch in enumerate(loader):
        skip = (batch is None)
        x_high = x_med = x_static = x_cond = x_sm = x_lc = pop_map = y_batch = valid_mask = None
        if not skip:
            items = [b.to(device, non_blocking=True) for b in batch]
            if add_pop:
                pop_map = items[-2]
                items = items[:-2] + [items[-1]]
            if add_sm and add_lc:
                x_high, x_med, x_static, x_cond, x_sm, x_lc, y_batch = items
            elif add_sm:
                x_high, x_med, x_static, x_cond, x_sm, y_batch = items
            elif add_lc:
                x_high, x_med, x_static, x_cond, x_lc, y_batch = items
            else:
                x_high, x_med, x_static, x_cond, y_batch = items
            if is_poisson:
                valid_mask = None
                if not torch.isfinite(y_batch).any():
                    skip = True
            else:
                y_batch = ensure_bchw_1x86x86(y_batch)
                valid_mask = (~torch.isnan(y_batch)).bool()
                if not valid_mask.any():
                    skip = True
        if is_ddp:
            skip_t = torch.tensor(int(skip), dtype=torch.int32, device=device)
            dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
            skip = skip_t.item() > 0
        if skip:
            continue

        if not is_poisson:
            y_batch = nan_checks_replace([y_batch], replace_nan=-2.0)[0]
            y_batch = normalize_to_neg_one_one(y_batch, data_min, data_max)

        amp_device = cfg.get("device", "cuda") if cfg else "cuda"
        amp_enabled = cfg.get("amp", True) if cfg else True
        with torch.amp.autocast(enabled=amp_enabled, device_type=amp_device):
            loss, metrics = compute_loss_fn(
                model, x_high, x_med, x_static, x_cond, y_batch, valid_mask,
                x_sm=x_sm, x_lc=x_lc, pop_map=pop_map,
            )

            if torch.isnan(loss):
                if logger:
                    logger.warning(f"[{desc}] Loss is NaN, skipping this batch")
                continue

        bs = y_batch.size(0)
        n_batches += bs
        running_loss += loss.item() * bs
        running_recon += metrics["recon"] * bs
        running_kl += metrics["kl"] * bs
        if is_poisson and not (math.isnan(metrics.get("mae", float("nan")))):
            running_mae  += metrics["mae"]  * bs
            running_bias += metrics["bias"] * bs
            n_mae_batches += bs

        del x_high, x_med, x_static, x_cond, x_sm, x_lc, pop_map, y_batch, loss, valid_mask

        cleanup_every = int(cfg.get("memory_cleanup_interval", 0) if cfg else 0)
        if cleanup_every > 0 and (batch_idx + 1) % cleanup_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if is_ddp:
        world_size = dist.get_world_size()
        count_tensor = torch.tensor(float(n_batches), device=device)
        dist.all_reduce(count_tensor)
        if count_tensor.item() == 0:
            return {"loss": float("nan"), "recon": float("nan"), "kl": float("nan"), "mae": float("nan"), "bias": float("nan")}
        loss_t  = torch.tensor(running_loss  / max(n_batches, 1), device=device)
        recon_t = torch.tensor(running_recon / max(n_batches, 1), device=device)
        kl_t    = torch.tensor(running_kl    / max(n_batches, 1), device=device)
        mae_t   = torch.tensor(running_mae   / max(n_mae_batches, 1), device=device)
        bias_t  = torch.tensor(running_bias  / max(n_mae_batches, 1), device=device)
        for t in (loss_t, recon_t, kl_t, mae_t, bias_t):
            dist.all_reduce(t)
        return {
            "loss":  loss_t.item()  / world_size,
            "recon": recon_t.item() / world_size,
            "kl":    kl_t.item()    / world_size,
            "mae":   mae_t.item()   / world_size if n_mae_batches > 0 else float("nan"),
            "bias":  bias_t.item()  / world_size if n_mae_batches > 0 else float("nan"),
        }

    if n_batches == 0:
        return {"loss": float("nan"), "recon": float("nan"), "kl": float("nan"), "mae": float("nan"), "bias": float("nan")}

    return {
        "loss":  running_loss  / n_batches,
        "recon": running_recon / n_batches,
        "kl":    running_kl    / n_batches,
        "mae":   running_mae   / n_mae_batches  if n_mae_batches > 0 else float("nan"),
        "bias":  running_bias  / n_mae_batches  if n_mae_batches > 0 else float("nan"),
    }


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------


def main(config: dict | None = None):
    default_config = {
        # --- shared ---
        "model": "vae",
        "batch_size": 1,
        "train_split": 0.8,
        "num_workers": 4,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "patience": 5,
        "amp": True,
        "grad_accum_steps": 1,
        "memory_cleanup_interval": 100,
        "save_dir": ROOT_DIR / "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "loss_fn": "mse",
        "ddp": False,
        "snapshot_every": 5,
        # --- VAE-specific ---
        "beta_kl": 0.01,
        "use_kl": False,
        "latent_channels": 4,
        "block_out_channels": (64, 128, 256),
        "layers_per_block": 2,
        "norm_num_groups": 32,
        # --- transformer-specific ---
        "med_in_ch": 18,
        "swin_model": "swin_tiny_patch4_window7_224.ms_in1k",
        "use_titok": False,
        "titok_backbone": "vit_base_patch16_224.mae",
        "titok_num_latent_tokens": 32,
        # --- soil moisture ---
        "add_sm": False,
        "sm_data_path": None,
        # --- land cover ---
        "add_lc": False,
        "lc_data_path": None,
        # --- GPWv4 population exposure (Poisson loss only) ---
        "gpw_path": None,
        "add_pop": False,
    }
    if config:
        default_config.update(config)
    cfg = default_config

    from torch.utils.data.distributed import DistributedSampler

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = cfg.get("ddp", False) and world_size > 1

    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(hours=2),
        )
        rank = dist.get_rank()
        print(f"Initialized DDP: world_size={world_size}, local_rank={local_rank}, global_rank={rank}")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg["device"])
        local_rank = 0
        rank = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(cfg.get("save_dir", ROOT_DIR / "checkpoints"))
    run_dir = base_dir / f"run_{timestamp}"

    for d in [run_dir / "checkpoints", run_dir / "logs", run_dir / "plots"]:
        d.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    plot_dir = run_dir / "plots"

    if rank == 0:
        logger = init_logging(log_file=log_dir / "training.log", verbose=False)
        logger.info(f"Starting {cfg['model'].upper()} training run at {timestamp}")
        logger.info(f"Run directory: {run_dir}")
        if is_ddp:
            logger.info(f"Using DDP: device={device}, world_size={world_size}, local_rank={local_rank}")
        else:
            logger.info(f"Using single GPU/CPU: device={device}")
    else:
        logger = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    viirs_data_path = DATA_PATH / "modis" / "VIIRS_nightlight"
    era5_path = DATA_PATH / "ERA5" / "Latin_america"
    risk_raster_path = DATA_PATH / "riskmaps" / "DEN_riskmap_wmean_masked.tif"
    admin_path = DATA_PATH / "dengue_cases"

    start_date = "2012-01-01"
    end_date = "2023-12-31"

    # Soil moisture is only available 2016-2022 — restrict window when enabled
    if cfg.get("add_sm", False):
        start_date = max(start_date, "2016-01-01")
        end_date   = min(end_date,   "2022-12-31")

    ds_cases = (
        xr.open_mfdataset(os.path.join(admin_path, "*.nc"))
        .sel(time=slice(start_date, end_date))
        .chunk(chunks={"time": 1})
    )
    num_zones = len(np.unique(ds_cases["FAO_GAUL_code"].values))

    # Materialise to pure-numpy before any fork() to avoid dask/HDF5 deadlocks.
    ds_cases.load()
    _dc_vars = {v: (ds_cases[v].dims, np.asarray(ds_cases[v].values)) for v in ds_cases.data_vars}
    _dc_coords = {c: np.asarray(ds_cases.coords[c].values) for c in ds_cases.coords}
    ds_cases = xr.Dataset(_dc_vars, _dc_coords)

    y = XarraySpatioTemporalDataset(ds_cases, variables=["dengue_total"], T_max=1)
    x_spatial = XarraySpatioTemporalDataset(ds_cases, variables=["FAO_GAUL_code"], T_max=1)
    shared_cache_dir = base_dir / "dataset_cache"
    shared_cache_dir.mkdir(parents=True, exist_ok=True)

    # ERA5 weekly cache — rank 0 precomputes, others wait.
    if is_ddp and rank != 0:
        dist.barrier()

    if rank == 0:
        weekly_zarr = shared_cache_dir / "era5_weekly.zarr"
        if not weekly_zarr.exists():
            logger.info("ERA5 weekly cache not found — precomputing (runs once, ~20–40 min)...")
            _era5_tmp = ERA5Daily(era5_path, T_max=63, min_date=start_date, max_date=end_date)
            _era5_tmp.precompute_weekly_cache(shared_cache_dir)

    if is_ddp and rank == 0:
        dist.barrier()

    era5 = ERA5Daily(
        era5_path, T_max=63, min_date=start_date, max_date=end_date,
        weekly_cache_dir=shared_cache_dir,
    )

    viirs = VIIRSData(viirs_data_path, min_date=start_date, max_date=end_date)
    static = StaticLayer(risk_raster_path, nodata=-3.3999999521443642e+38)

    soil_moisture = None
    if cfg.get("add_sm", False):
        sm_path = cfg.get("sm_data_path")
        if not sm_path:
            raise ValueError("--sm_data_path is required when --add_sm is set")
        soil_moisture = SoilMoistureData(
            sm_path, min_date=start_date, max_date=end_date,
            cache_dir=shared_cache_dir,
        )
        if rank == 0:
            logger.info(f"SoilMoistureData loaded from {sm_path}")

    land_cover = None
    if cfg.get("add_lc", False):
        lc_path = cfg.get("lc_data_path")
        if not lc_path:
            raise ValueError("--lc_data_path is required when --add_lc is set")
        land_cover = LandCoverData(lc_path)
        if rank == 0:
            logger.info(f"LandCoverData loaded from {lc_path}")

    population = None
    gpw_path = cfg.get("gpw_path", None)
    if gpw_path is not None and cfg.get("loss_fn", "mse") == "poisson":
        population = GPWv4Population(gpw_path, target_size=(86, 86))
        cfg["add_pop"] = True
        if rank == 0:
            logger.info(f"GPWv4Population loaded from {gpw_path} (exposure-only, not a model input)")

    # DengueDataset patch cache — rank 0 computes, others wait.
    if is_ddp and rank != 0:
        dist.barrier()

    full_dataset = DengueDataset(
        viirs, era5, static, x_spatial, y,
        bbox=latin_box(),
        skip_era5_bounds=True,
        cache_dir=shared_cache_dir,
        num_zones=num_zones,
        loss_fn=cfg.get("loss_fn", "mse"),
        soil_moisture=soil_moisture,
        land_cover=land_cover,
        population=population,
    )

    if is_ddp and rank == 0:
        dist.barrier()

    if rank == 0:
        logger.info(f"Full dataset size: {len(full_dataset)}")

    train_size = int(cfg["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    def _worker_init_fn(worker_id):
        torch.set_num_threads(1)
        os.environ['GDAL_CACHEMAX'] = '256'
        try:
            from osgeo import gdal
            gdal.SetCacheMax(256 * 1024 * 1024)
        except Exception:
            pass
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return
        ds = worker_info.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset
        for attr in ('viirs', 'static'):
            old = getattr(ds, attr, None)
            if old is None or not hasattr(old, 'paths'):
                continue
            try:
                setattr(ds, attr, old.__class__(paths=old.paths, crs=old.crs, res=old.res))
            except Exception:
                pass

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=(not is_ddp),
        collate_fn=collate_skip_none,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn if cfg["num_workers"] > 0 else None,
    )
    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_skip_none,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn if cfg["num_workers"] > 0 else None,
    )
    if cfg["num_workers"] > 0:
        train_loader_kwargs["persistent_workers"] = False
        val_loader_kwargs["persistent_workers"] = False

    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_loader_kwargs["sampler"] = train_sampler
        val_loader_kwargs["sampler"] = val_sampler
        del train_loader_kwargs["shuffle"]

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    if rank == 0:
        logger.info(f"DataLoader: num_workers={cfg['num_workers']}, batch_size={cfg['batch_size']}")

    data_min = float(ds_cases['dengue_total'].min())
    data_max = float(ds_cases['dengue_total'].max())
    if rank == 0:
        logger.info(f"Global min: {data_min:.6f}, max: {data_max:.6f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model, compute_loss_fn = build_model(cfg, num_zones)

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model.to(device), device_ids=[local_rank])
    else:
        model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler(enabled=cfg["amp"])

    es_config = SimpleNamespace(patience=cfg["patience"], min_patience=0)
    early_stopper = EarlyStopping(es_config, verbose=True)

    # ------------------------------------------------------------------
    # Optional checkpoint resume
    # ------------------------------------------------------------------
    start_epoch = 1
    ckpt_epoch = cfg.get("checkpoint", 0)
    if ckpt_epoch > 0:
        # If an explicit run dir was given, use it; otherwise scan save_dir for the
        # most recent run that contains checkpoint_epoch_{ckpt_epoch}.
        ckpt_run = cfg.get("checkpoint_run")
        if ckpt_run:
            ckpt_run = Path(ckpt_run)
        else:
            base_dir = Path(cfg.get("save_dir", ROOT_DIR / "checkpoints"))
            candidates = sorted(
                [
                    d for d in base_dir.iterdir()
                    if d.is_dir() and d.name.startswith("run_")
                    and (d / "checkpoints" / f"checkpoint_epoch_{ckpt_epoch}").exists()
                ],
                key=lambda d: d.name,
                reverse=True,   # most recent first (timestamp in name)
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No run in {base_dir} contains checkpoint_epoch_{ckpt_epoch}"
                )
            ckpt_run = candidates[0]
            if rank == 0:
                logger.info(f"Auto-selected checkpoint run: {ckpt_run}")

        ckpt_dir = ckpt_run / "checkpoints" / f"checkpoint_epoch_{ckpt_epoch}"
        metadata_path = ckpt_dir / f"metadata_epoch_{ckpt_epoch}.pth"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        raw_model = model.module if hasattr(model, "module") else model
        _sd = torch.load(metadata["components"]["model_state"], map_location="cpu", weights_only=True)
        if all(k.startswith("module.") for k in _sd):
            _sd = {k[len("module."):]: v for k, v in _sd.items()}
        raw_model.load_state_dict(_sd)
        optimizer.load_state_dict(
            torch.load(metadata["components"]["optimizer_state"], map_location=device, weights_only=True)
        )
        start_epoch = ckpt_epoch + 1
        if rank == 0:
            logger.info(f"Resumed from {ckpt_dir} — starting at epoch {start_epoch}")

    train_losses = []
    val_losses = []

    # Grab one fixed val batch for periodic prediction snapshots (rank 0 only).
    # Also saved to disk so visualize_predictions.py can run without reloading the dataset.
    snapshot_batch = None
    if rank == 0 and cfg.get("snapshot_every", 0) > 0:
        for _b in val_loader:
            if _b is not None:
                snapshot_batch = [t.cpu() for t in _b]
                torch.save(snapshot_batch, run_dir / "reference_batch.pt")
                break

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if rank == 0:
        logger.info(f"Starting training for {cfg['num_epochs']} epochs...")

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model=model,
            compute_loss_fn=compute_loss_fn,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            data_min=data_min,
            data_max=data_max,
            scaler=scaler,
            cfg=cfg,
            logger=logger,
            is_ddp=is_ddp,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            compute_loss_fn=compute_loss_fn,
            loader=val_loader,
            device=device,
            data_min=data_min,
            data_max=data_max,
            desc="Validation",
            cfg=cfg,
            logger=logger,
            is_ddp=is_ddp,
        )

        scheduler.step(val_metrics["loss"])

        if rank == 0:
            val_mae  = val_metrics.get("mae",  float("nan"))
            val_bias = val_metrics.get("bias", float("nan"))
            bias_str = (
                f" | Val MAE: {val_mae:.2f}, Bias: {val_bias:+.2f}"
                if not (math.isnan(val_mae) or math.isnan(val_bias)) else ""
            )
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_metrics['loss']:.6f} "
                f"(Recon: {train_metrics['recon']:.6f}, KL: {train_metrics['kl']:.6f}) | "
                f"Val Loss: {val_metrics['loss']:.6f} "
                f"(Recon: {val_metrics['recon']:.6f}, KL: {val_metrics['kl']:.6f})"
                f"{bias_str}"
            )

            train_losses.append(train_metrics["loss"])
            val_losses.append(val_metrics["loss"])
            plot_learning_curves(train_losses, val_losses, plot_dir, logger)

            snapshot_every = cfg.get("snapshot_every", 0)
            if snapshot_every > 0 and epoch % snapshot_every == 0 and snapshot_batch is not None:
                save_prediction_snapshot(
                    model, snapshot_batch, device, num_zones,
                    plot_dir, epoch, cfg, logger,
                )

            _raw_m = model.module if hasattr(model, "module") else model
            model_dict = {
                "epoch": epoch,
                "model_state": _raw_m.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            early_stopper(val_metrics["loss"], model_dict, epoch, str(checkpoint_dir))

            if early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    if is_ddp:
        dist.destroy_process_group()
        if rank == 0:
            logger.info("DDP process group destroyed")

    if rank == 0:
        logger.info("Training finished.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train dengue model (VAE or Transformer).")

    # --- shared ---
    parser.add_argument("--model", type=str, choices=["vae", "transformer"], default="vae")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cleanup_every", type=int, default=100)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--loss_fn", type=str, choices=["mse", "poisson"], default="mse")

    # --- VAE-specific ---
    parser.add_argument("--beta_kl", type=float, default=0.01)
    parser.add_argument("--use_kl", action="store_true")
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--layers_per_block", type=int, default=2)
    parser.add_argument("--norm_num_groups", type=int, default=32)

    # --- transformer-specific ---
    parser.add_argument("--med_in_ch", type=int, default=18,
                        help="ERA5 channel count after weekly aggregation")
    parser.add_argument("--swin_model", type=str, default="swin_tiny_patch4_window7_224.ms_in1k")
    parser.add_argument("--use_titok", action="store_true")
    parser.add_argument("--titok_backbone", type=str, default="vit_base_patch16_224.mae")
    parser.add_argument("--titok_num_latent_tokens", type=int, default=32)

    # --- soil moisture ---
    parser.add_argument("--add_sm", action="store_true",
                        help="Add soil moisture branch; restricts training data to 2016-2022")
    parser.add_argument("--sm_data_path", type=str, default=SM_DATA_PATH,
                        help="Root directory of SM_[A|D]_YYYYMMDD.tif files (required with --add_sm)")

    # --- land cover ---
    parser.add_argument("--add_lc", action="store_true",
                        help="Add MODIS MCD12Q1 land cover as extra static channel")
    parser.add_argument("--lc_data_path", type=str, default=LC_DATA_PATH,
                        help="Root directory of annual MCD12Q1 GeoTIFF files (required with --add_lc)")

    # --- GPWv4 population exposure (Poisson loss only) ---
    parser.add_argument("--gpw_path", type=str, default=None,
                        help="Directory containing GPWv4 NetCDF files "
                             "(GPWv4_latin_america_YYYY.nc). "
                             "Only used with --loss_fn poisson; ignored otherwise.")

    # --- resume ---
    parser.add_argument("--checkpoint", type=int, default=0,
                        help="Epoch to resume from (0 = start fresh). "
                             "Requires --checkpoint_run.")
    parser.add_argument("--checkpoint_run", type=str, default=None,
                        help="Path to the run directory to resume from, e.g. "
                             "checkpoints/run_20260613_150752. "
                             "Required when --checkpoint > 0.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "train_split": args.train_split,
        "num_workers": args.num_workers,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "beta_kl": args.beta_kl,
        "use_kl": args.use_kl,
        "latent_channels": args.latent_channels,
        "layers_per_block": args.layers_per_block,
        "norm_num_groups": 32,
        "memory_cleanup_interval": args.cleanup_every,
        "grad_accum_steps": args.grad_accum_steps,
        "amp": True,
        "save_dir": ROOT_DIR / "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "loss_fn": args.loss_fn,
        "ddp": args.ddp,
        "med_in_ch": args.med_in_ch,
        "swin_model": args.swin_model,
        "use_titok": args.use_titok,
        "titok_backbone": args.titok_backbone,
        "titok_num_latent_tokens": args.titok_num_latent_tokens,
        "add_sm": args.add_sm,
        "sm_data_path": args.sm_data_path,
        "add_lc": args.add_lc,
        "lc_data_path": args.lc_data_path,
        "gpw_path": args.gpw_path,
        "checkpoint": args.checkpoint,
        "checkpoint_run": args.checkpoint_run,
    }

    main(config=config)
