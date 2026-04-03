import os
import argparse
import gc
from torch.utils.data import DataLoader
import torch.distributed as dist

import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

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
from definitions import DATA_PATH, ROOT_DIR
from models.model_utils import (
    EarlyStopping,
    nan_checks_replace,
    standardize_tensor,
    export_batches
)
from utils import latin_box, init_logging


# ------------------------------------------------------------------
# MODEL
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
        # Return tensors only for DataParallel/DDP compatibility
        # posterior is DiagonalGaussianDistribution, extract mean and logvar
        posterior_mean = posterior.mean
        posterior_logvar = posterior.logvar
        return recon, posterior_mean, posterior_logvar


# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------


def plot_learning_curves(train_losses, val_losses, plot_dir, logger):
    """Plot and save learning curves."""
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
# TRAIN
# ------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, device, data_min, data_max, 
                   beta_kl=0.01, use_kl=False, scaler=None, cfg=None, logger=None, is_ddp=False):
    """Train for one epoch with optional gradient accumulation and memory cleanup."""
    
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    n_batches = 0
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1) if cfg else 1))
    
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        start = time.time()
        if batch is None:
            continue

        x_high, x_med, x_static, x_cond, y_batch = [
            b.to(device, non_blocking=True) for b in batch
        ]

        y_batch = ensure_bchw_1x86x86(y_batch)
        valid_mask = (~torch.isnan(y_batch)).bool()
        if not valid_mask.any():
            continue

        # Preprocess
        y_batch = nan_checks_replace([y_batch], replace_nan=-2.0)[0]
        y_batch = normalize_to_neg_one_one(y_batch, data_min, data_max)

        with torch.amp.autocast(enabled=cfg.get("amp", True) if cfg else True, device_type=cfg.get("device", "cuda") if cfg else "cuda"):
            recon, posterior_mean, posterior_logvar = model(y_batch)

            if recon.shape != y_batch.shape:
                raise ValueError(f"Shape mismatch: recon {recon.shape} vs y_batch {y_batch.shape}")

            recon_loss = masked_mse_loss(recon, y_batch, valid_mask)
            # Compute KL divergence manually: KL(N(mean, logvar) || N(0,1))
            kl_loss = -0.5 * torch.mean(1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp())
            loss = recon_loss + beta_kl * kl_loss if use_kl else recon_loss

            if torch.isnan(loss):
                if logger:
                    logger.warning("[train] Loss is NaN, skipping this batch")
                continue

            scaled_loss = loss / grad_accum_steps
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if is_ddp:
            torch.distributed.barrier()

        should_step = ((batch_idx + 1) % grad_accum_steps == 0)
        if should_step:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = y_batch.size(0)
        n_batches += bs
        running_loss += loss.item() * bs
        running_recon += recon_loss.item() * bs
        running_kl += kl_loss.item() * bs

        # Explicit cleanup
        del x_high, x_med, x_static, x_cond, y_batch, recon, posterior_mean, posterior_logvar, loss, valid_mask
        
        cleanup_every = int(cfg.get("memory_cleanup_interval", 0) if cfg else 0)
        if cleanup_every > 0 and (batch_idx + 1) % cleanup_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"Rank {dist.get_rank()} batch time: {time.time() - start}")

    # Handle trailing gradients
    if n_batches > 0 and (n_batches % grad_accum_steps) != 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if n_batches == 0:
        return {"loss": float("nan"), "recon": float("nan"), "kl": float("nan")}
    
    # Aggregate across DDP replicas
    if is_ddp:
        world_size = torch.distributed.get_world_size()
        loss_tensor = torch.tensor(running_loss / n_batches, device=device)
        recon_tensor = torch.tensor(running_recon / n_batches, device=device)
        kl_tensor = torch.tensor(running_kl / n_batches, device=device)
        count_tensor = torch.tensor(n_batches, device=device)
        
        torch.distributed.all_reduce(loss_tensor)
        torch.distributed.all_reduce(recon_tensor)
        torch.distributed.all_reduce(kl_tensor)
        torch.distributed.all_reduce(count_tensor)
        
        return {
            "loss": loss_tensor.item() / world_size,
            "recon": recon_tensor.item() / world_size,
            "kl": kl_tensor.item() / world_size,
        }
    
    return {
        "loss": running_loss / n_batches,
        "recon": running_recon / n_batches,
        "kl": running_kl / n_batches,
    }

@torch.inference_mode()
def evaluate_one_epoch(model, loader, device, data_min, data_max, 
                      beta_kl=0.01, use_kl=False, desc="Eval", cfg=None, logger=None, is_ddp=False):
    """Evaluate for one epoch with memory cleanup."""
    model.eval()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue

        x_high, x_med, x_static, x_cond, y_batch = [
            b.to(device, non_blocking=True) for b in batch
        ]

        y_batch = ensure_bchw_1x86x86(y_batch)
        valid_mask = (~torch.isnan(y_batch)).bool()
        if not valid_mask.any():
            continue

        # Preprocess
        y_batch = nan_checks_replace([y_batch], replace_nan=-2.0)[0]
        y_batch = normalize_to_neg_one_one(y_batch, data_min, data_max)

        with torch.amp.autocast(enabled=cfg.get("amp", True) if cfg else True, device_type=cfg.get("device", "cuda") if cfg else "cuda"):
            recon, posterior_mean, posterior_logvar = model(y_batch)

            if recon.shape != y_batch.shape:
                raise ValueError(f"Shape mismatch: recon {recon.shape} vs y_batch {y_batch.shape}")

            recon_loss = masked_mse_loss(recon, y_batch, valid_mask)
            # Compute KL divergence manually: KL(N(mean, logvar) || N(0,1))
            kl_loss = -0.5 * torch.mean(1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp())
            loss = recon_loss + beta_kl * kl_loss if use_kl else recon_loss

            if torch.isnan(loss):
                if logger:
                    logger.warning(f"[{desc}] Loss is NaN, skipping this batch")
                continue

        bs = y_batch.size(0)
        n_batches += bs
        running_loss += loss.item() * bs
        running_recon += recon_loss.item() * bs
        running_kl += kl_loss.item() * bs

        # Explicit cleanup
        del x_high, x_med, x_static, x_cond, y_batch, recon, posterior_mean, posterior_logvar, loss, valid_mask
        
        cleanup_every = int(cfg.get("memory_cleanup_interval", 0) if cfg else 0)
        if cleanup_every > 0 and (batch_idx + 1) % cleanup_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if n_batches == 0:
        return {"loss": float("nan"), "recon": float("nan"), "kl": float("nan")}
    
    # Aggregate across DDP replicas
    if is_ddp:
        world_size = torch.distributed.get_world_size()
        loss_tensor = torch.tensor(running_loss / n_batches, device=device)
        recon_tensor = torch.tensor(running_recon / n_batches, device=device)
        kl_tensor = torch.tensor(running_kl / n_batches, device=device)
        count_tensor = torch.tensor(n_batches, device=device)
        
        torch.distributed.all_reduce(loss_tensor)
        torch.distributed.all_reduce(recon_tensor)
        torch.distributed.all_reduce(kl_tensor)
        torch.distributed.all_reduce(count_tensor)
        
        return {
            "loss": loss_tensor.item() / world_size,
            "recon": recon_tensor.item() / world_size,
            "kl": kl_tensor.item() / world_size,
        }
    
    return {
        "loss": running_loss / n_batches,
        "recon": running_recon / n_batches,
        "kl": running_kl / n_batches,
    }


def main(config: dict | None = None):
    """Main training pipeline following pipeline.py pattern."""
    
    # Merge config
    default_config = {
        "batch_size": 1,
        "train_split": 0.8,
        "num_workers": 4,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "patience": 5,
        "beta_kl": 0.01,
        "latent_channels": 4,
        "block_out_channels": (64, 128, 256),
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "amp": True,
        "grad_accum_steps": 1,
        "memory_cleanup_interval": 100,
        "save_dir": ROOT_DIR / "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_kl": False,
    }
    if config:
        default_config.update(config)
    cfg = default_config

    # Determine rank and DDP setup BEFORE logging
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = args.ddp and world_size > 1
    
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        rank = dist.get_rank()
        print(f"Initialized DDP: world_size={world_size}, local_rank={local_rank}, global_rank={rank}")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
    else:
        device = torch.device(cfg["device"])
        local_rank = 0
        rank = 0

    # Setup logging - ONLY on rank 0 (or when not using DDP)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(config.get("save_dir", ROOT_DIR / "checkpoints"))
    run_dir = base_dir / f"run_{timestamp}"
    
    # Only rank 0 creates directories and sets up logging
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = run_dir / "checkpoints"
        log_dir = run_dir / "logs"
        plot_dir = run_dir / "plots"
        checkpoint_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        plot_dir.mkdir(exist_ok=True)
        logger = init_logging(log_file=log_dir / "training.log", verbose=False)
        logger.info(f"Starting VAE training run at {timestamp}")
        logger.info(f"Run directory: {run_dir}")
    else:
        # Create placeholder objects for non-rank-0 processes
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = run_dir / "checkpoints"
        log_dir = run_dir / "logs"
        plot_dir = run_dir / "plots"
        checkpoint_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        plot_dir.mkdir(exist_ok=True)
        logger = None
    
    if rank == 0:
        if is_ddp:
            logger.info(f"Using DDP: device={device}, world_size={world_size}, local_rank={local_rank}")
        else:
            logger.info(f"Using single GPU: device={device}")

    # ------------------------------------------------------------------
    # Dataset construction (reuse pipeline.py pattern)
    # ------------------------------------------------------------------
    viirs_data_path = DATA_PATH / "modis" / "VIIRS_nightlight"
    era5_path = DATA_PATH / "ERA5" / "Latin_america"
    risk_raster_path = DATA_PATH / "riskmaps_public main data" / "DEN_riskmap_wmean_masked.tif"
    admin_path = DATA_PATH / "dengue_cases"

    start_date = "2012-01-01"
    end_date = "2023-12-31"

    ds_cases = xr.open_mfdataset(os.path.join(admin_path, "*.nc")).sel(time=slice(start_date, end_date)).chunk(chunks={"time": 1, "x": 86, "y": 86})
    num_zones = len(np.unique(ds_cases["FAO_GAUL_code"].values))

    y = XarraySpatioTemporalDataset(ds_cases, variables=["dengue_total"], T_max=1)
    x_spatial = XarraySpatioTemporalDataset(ds_cases, variables=["FAO_GAUL_code"], T_max=1)
    era5 = ERA5Daily(era5_path, T_max=63, min_date=start_date, max_date=end_date)
    viirs = VIIRSData(viirs_data_path, min_date=start_date, max_date=end_date)
    static = StaticLayer(risk_raster_path, nodata=-3.3999999521443642e+38)

    shared_cache_dir = base_dir / "dataset_cache"
    full_dataset = DengueDataset(
        viirs, era5, static, x_spatial, y,
        bbox=latin_box(),
        skip_era5_bounds=True,
        cache_dir=shared_cache_dir,
        num_zones=num_zones,
        loss_fn=cfg.get("loss_fn", "mse"),
    )

    if rank == 0:
        logger.info(f"Full dataset size: {len(full_dataset)}")

    # Train/val split
    train_size = int(cfg["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    def _worker_init_fn(_):
        torch.set_num_threads(1)

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=(not is_ddp),  # Use sampler for DDP
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

    # Add DistributedSampler for DDP
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
        train_loader_kwargs["sampler"] = train_sampler
        val_loader_kwargs["sampler"] = val_sampler
        del train_loader_kwargs["shuffle"]  # Can't use shuffle with sampler

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    if rank == 0:
        logger.info(f"DataLoader config: num_workers={cfg['num_workers']}, batch_size={cfg['batch_size']}")

    # Compute global min/max
    if rank == 0:
        logger.info("Computing global min/max on training set...")
    # data_min, data_max = compute_global_minmax(train_loader, device=device)
    data_min, data_max = float(ds_cases['dengue_total'].min().values), float(ds_cases['dengue_total'].max().values)
    if rank == 0:
        logger.info(f"Global training min: {data_min:.6f}, max: {data_max:.6f}")

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler
    # ------------------------------------------------------------------
    vae = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        latent_channels=cfg["latent_channels"],
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=cfg["block_out_channels"],
        layers_per_block=cfg["layers_per_block"],
        norm_num_groups=cfg["norm_num_groups"],
    )

    model = VAEWithTrainableResize(vae)

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

    train_losses = []
    val_losses = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if rank == 0:
        logger.info(f"Starting training for {cfg['num_epochs']} epochs...")
    
    for epoch in range(1, cfg["num_epochs"] + 1):
        # Update sampler epoch for DDP
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            data_min=data_min,
            data_max=data_max,
            beta_kl=cfg["beta_kl"],
            use_kl=cfg["use_kl"],
            scaler=scaler,
            cfg=cfg,
            logger=logger,
            is_ddp=is_ddp,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            data_min=data_min,
            data_max=data_max,
            beta_kl=cfg["beta_kl"],
            use_kl=cfg["use_kl"],
            desc="Validation",
            cfg=cfg,
            logger=logger,
            is_ddp=is_ddp,
        )

        scheduler.step(val_metrics["loss"])

        # Only log from rank 0
        if rank == 0:
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_metrics['loss']:.6f} "
                f"(Recon: {train_metrics['recon']:.6f}, KL: {train_metrics['kl']:.6f}) | "
                f"Val Loss: {val_metrics['loss']:.6f} "
                f"(Recon: {val_metrics['recon']:.6f}, KL: {val_metrics['kl']:.6f})"
            )

            train_losses.append(train_metrics["loss"])
            val_losses.append(val_metrics["loss"])
            plot_learning_curves(train_losses, val_losses, plot_dir, logger)

            # Early stopping
            model_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            early_stopper(val_metrics["loss"], model_dict, epoch, str(checkpoint_dir))

            if early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Cleanup DDP
    if is_ddp:
        dist.destroy_process_group()
        if rank == 0:
            logger.info("DDP process group destroyed")

    if rank == 0:
        logger.info("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on dengue dataset.")
    parser.add_argument("--ddp", action="store_true", help="training with Distributed Data Parallel")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DDP")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--beta_kl", type=float, default=0.01, help="KL divergence weight")
    parser.add_argument("--latent_channels", type=int, default=4, help="VAE latent channels")
    parser.add_argument("--layers_per_block", type=int, default=2, help="Layers per VAE block")
    parser.add_argument("--norm_num_groups", type=int, default=32, help="GroupNorm groups")
    parser.add_argument("--use_kl", action="store_true", help="Include KL loss term")
    parser.add_argument("--cleanup_every", type=int, default=100, help="Memory cleanup every N batches")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--loss_fn", type=str, choices=["mse", "poisson"], default="mse", help="Loss function to use for training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {
        "batch_size": args.batch_size,
        "train_split": args.train_split,
        "num_workers": args.num_workers,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "beta_kl": args.beta_kl,
        "latent_channels": args.latent_channels,
        "layers_per_block": args.layers_per_block,
        "norm_num_groups": 32,
        "use_kl": args.use_kl,
        "memory_cleanup_interval": args.cleanup_every,
        "grad_accum_steps": args.grad_accum_steps,
        "amp": True,
        "save_dir": ROOT_DIR / "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    main(config=config)