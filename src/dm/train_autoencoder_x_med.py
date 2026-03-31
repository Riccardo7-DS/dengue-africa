import os
import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import xarray as xr

from models import (
    VIIRSData,
    ERA5Daily,
    StaticLayer,
    XarraySpatioTemporalDataset,
    DengueDataset,
    collate_skip_none,
)
from definitions import DATA_PATH, ROOT_DIR
from models.model_utils import EarlyStopping
from utils import latin_box

from utils_training import (
    plot_learning_curves,
    setup_run_dirs_and_logger,
    setup_device_and_distributed,
    wrap_model_for_parallel,
    build_train_val_loaders,
    standardize_tensor,
    compute_channelwise_mean_std_btchw,
)


# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------

class Autoencoder_x_med(nn.Module):
    """
    Input:  [B, T, C, 43, 43]
    Latent: [B, T, latent_dim]
    Output: [B, T, C, 43, 43]
    """

    def __init__(self, in_channels: int, latent_dim: int = None, hidden_channels: int = 32):
        super().__init__()

        if latent_dim is None:
            latent_dim = in_channels

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),  # 43 -> 22
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),  # 22 -> 11
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),  # 11 -> 6
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_latent = nn.Linear(hidden_channels, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_channels * 6 * 6)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),  # 6 -> 12
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),

            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),

            nn.ConvTranspose2d(hidden_channels, in_channels, kernel_size=4, stride=2, padding=1),      # 24 -> 48
        )

        self.final_resize = nn.AdaptiveAvgPool2d((43, 43))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.encoder_cnn(x)                    # [B*T, hidden, 6, 6]
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B*T, hidden]
        z = self.to_latent(x)                      # [B*T, latent_dim]
        z = z.reshape(B, T, self.latent_dim)       # [B, T, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        x = z.reshape(B * T, D)
        x = self.from_latent(x)
        x = x.reshape(B * T, self.hidden_channels, 6, 6)
        x = self.decoder_cnn(x)                    # [B*T, C, 48, 48]
        x = self.final_resize(x)                   # [B*T, C, 43, 43]
        x = x.reshape(B, T, self.in_channels, 43, 43)
        return x

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------

def ensure_btchw_43(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input is not a torch.Tensor, got: {type(x)}")

    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor [B,T,C,H,W], got shape {tuple(x.shape)}")

    if x.shape[-2] != 43 or x.shape[-1] != 43:
        raise ValueError(f"Expected H=W=43, got shape {tuple(x.shape)}")

    return x


# ------------------------------------------------------------------
# LOSS
# ------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type="mse"):
        super().__init__()
        self.loss_type = loss_type.lower()

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

        if self.loss_type == "mse":
            loss = torch.mean((pred - target) ** 2)
        elif self.loss_type == "l1":
            loss = torch.mean(torch.abs(pred - target))
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        return loss


# ------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    xmed_mean,
    xmed_std,
    scaler=None,
    cfg=None,
    logger=None,
    is_ddp=False,
):
    model.train()
    running_loss = 0.0
    n_samples = 0
    num_steps = 0

    criterion = ReconstructionLoss(loss_type=cfg.get("recon_loss", "mse"))
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1) if cfg else 1))

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        x_high, x_med, x_static, x_cond, y_batch = [b.to(device, non_blocking=True) for b in batch]

        x_med = ensure_btchw_43(x_med)
        x_in = standardize_tensor(x_med.float(), xmed_mean, xmed_std)

        with torch.amp.autocast(
            enabled=cfg.get("amp", True) if cfg else True,
            device_type=cfg.get("device", "cuda") if cfg else "cuda"
        ):
            recon, z = model(x_in)
            loss = criterion(recon, x_in)

            if torch.isnan(loss):
                if logger:
                    logger.warning("[train] Loss is NaN, skipping batch")
                continue

            scaled_loss = loss / grad_accum_steps

        if scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        num_steps += 1
        should_step = ((num_steps % grad_accum_steps) == 0)

        if should_step:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = x_in.size(0)
        n_samples += bs
        running_loss += loss.item() * bs

        if batch_idx == 0 and logger is not None:
            logger.info(
                f"x_med shape: {tuple(x_in.shape)} | "
                f"latent z shape: {tuple(z.shape)} | "
                f"recon shape: {tuple(recon.shape)}"
            )

        del x_high, x_med, x_static, x_cond, y_batch, x_in, recon, z, loss

        cleanup_every = int(cfg.get("memory_cleanup_interval", 0) if cfg else 0)
        if cleanup_every > 0 and (batch_idx + 1) % cleanup_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if num_steps > 0 and (num_steps % grad_accum_steps) != 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if n_samples == 0:
        return {"loss": float("nan")}

    if is_ddp:
        loss_tensor = torch.tensor(running_loss, device=device, dtype=torch.float64)
        count_tensor = torch.tensor(n_samples, device=device, dtype=torch.float64)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
        running_loss = loss_tensor.item()
        n_samples = int(count_tensor.item())

    return {"loss": running_loss / n_samples}


@torch.inference_mode()
def evaluate_one_epoch(
    model,
    loader,
    device,
    xmed_mean,
    xmed_std,
    cfg=None,
    logger=None,
    is_ddp=False,
):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    criterion = ReconstructionLoss(loss_type=cfg.get("recon_loss", "mse"))

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue

        x_high, x_med, x_static, x_cond, y_batch = [b.to(device, non_blocking=True) for b in batch]

        x_med = ensure_btchw_43(x_med)
        x_in = standardize_tensor(x_med.float(), xmed_mean, xmed_std)

        with torch.amp.autocast(
            enabled=cfg.get("amp", True) if cfg else True,
            device_type=cfg.get("device", "cuda") if cfg else "cuda"
        ):
            recon, z = model(x_in)
            loss = criterion(recon, x_in)

            if torch.isnan(loss):
                if logger:
                    logger.warning("[val] Loss is NaN, skipping batch")
                continue

        bs = x_in.size(0)
        n_samples += bs
        running_loss += loss.item() * bs

        del x_high, x_med, x_static, x_cond, y_batch, x_in, recon, z, loss

        cleanup_every = int(cfg.get("memory_cleanup_interval", 0) if cfg else 0)
        if cleanup_every > 0 and (batch_idx + 1) % cleanup_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if n_samples == 0:
        return {"loss": float("nan")}

    if is_ddp:
        loss_tensor = torch.tensor(running_loss, device=device, dtype=torch.float64)
        count_tensor = torch.tensor(n_samples, device=device, dtype=torch.float64)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
        running_loss = loss_tensor.item()
        n_samples = int(count_tensor.item())

    return {"loss": running_loss / n_samples}


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main(config: dict | None = None, args=None):
    run_dir, checkpoint_dir, log_dir, plot_dir, logger = setup_run_dirs_and_logger(
        config=config,
        default_save_dir=ROOT_DIR / "checkpoints",
        run_prefix="run",
    )

    logger.info("Starting autoencoder training run")
    logger.info(f"Run directory: {run_dir}")

    default_config = {
        "batch_size": 1,
        "train_split": 0.8,
        "num_workers": 4,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "patience": 5,
        "hidden_channels": 32,
        "latent_dim": None,
        "amp": True,
        "grad_accum_steps": 1,
        "memory_cleanup_interval": 100,
        "save_dir": ROOT_DIR / "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "recon_loss": "mse",
    }
    if config:
        default_config.update(config)
    cfg = default_config

    is_ddp, device, world_size = setup_device_and_distributed(args, cfg, logger)

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------
    viirs_data_path = DATA_PATH / "modis" / "VIIRS_nightlight"
    era5_path = DATA_PATH / "ERA5" / "Latin_america"
    risk_raster_path = DATA_PATH / "riskmaps_public main data" / "DEN_riskmap_wmean_masked.tif"
    admin_path = DATA_PATH / "dengue_cases"

    start_date = "2012-01-01"
    end_date = "2023-12-31"

    ds_cases = xr.open_mfdataset(os.path.join(admin_path, "*.nc")).sel(time=slice(start_date, end_date))
    num_zones = len(np.unique(ds_cases["FAO_GAUL_code"].values))

    y = XarraySpatioTemporalDataset(ds_cases, variables=["dengue_total"], T_max=1)
    x_spatial = XarraySpatioTemporalDataset(ds_cases, variables=["FAO_GAUL_code"], T_max=1)
    era5 = ERA5Daily(era5_path, T_max=63, min_date=start_date, max_date=end_date)
    viirs = VIIRSData(viirs_data_path, min_date=start_date, max_date=end_date)
    static = StaticLayer(risk_raster_path, nodata=-3.3999999521443642e+38)

    shared_cache_dir = Path(cfg["save_dir"]) / "dataset_cache"
    full_dataset = DengueDataset(
        viirs, era5, static, x_spatial, y,
        bbox=latin_box(),
        skip_era5_bounds=True,
        cache_dir=shared_cache_dir,
        num_zones=num_zones,
        loss_fn=cfg.get("loss_fn", "mse"),
    )

    logger.info(f"Full dataset size: {len(full_dataset)}")

    _, _, train_loader, val_loader, train_sampler, _ = build_train_val_loaders(
        full_dataset=full_dataset,
        train_split=cfg["train_split"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        collate_fn=collate_skip_none,
        is_ddp=is_ddp,
        world_size=world_size,
        local_rank=args.local_rank,
    )

    logger.info(f"DataLoader config: num_workers={cfg['num_workers']}, batch_size={cfg['batch_size']}")

    # infer C from first valid batch
    sample_batch = next(iter(train_loader))
    x_med_sample = sample_batch[1]
    if x_med_sample is None:
        raise RuntimeError("x_med sample is None")
    x_med_sample = ensure_btchw_43(x_med_sample)
    in_channels = x_med_sample.shape[2]
    latent_dim = cfg["latent_dim"] if cfg["latent_dim"] is not None else in_channels
    logger.info(f"x_med channels: {in_channels} | latent_dim: {latent_dim}")

    # Compute channel-wise mean/std on training set
    logger.info("Computing channel-wise mean/std on training set for x_med...")
    xmed_mean, xmed_std = compute_channelwise_mean_std_btchw(
        train_loader,
        device=device,
        tensor_index=1,
        ensure_fn=ensure_btchw_43,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler
    # ------------------------------------------------------------------
    model = Autoencoder_x_med(
        in_channels=in_channels,
        latent_dim=latent_dim,
        hidden_channels=cfg["hidden_channels"],
    ).to(device)

    model = wrap_model_for_parallel(model, args, device)

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
    logger.info(f"Starting training for {cfg['num_epochs']} epochs...")

    for epoch in range(1, cfg["num_epochs"] + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            xmed_mean=xmed_mean,
            xmed_std=xmed_std,
            scaler=scaler,
            cfg=cfg,
            logger=logger,
            is_ddp=is_ddp,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            xmed_mean=xmed_mean,
            xmed_std=xmed_std,
            cfg=cfg,
            logger=logger,
            is_ddp=is_ddp,
        )

        scheduler.step(val_metrics["loss"])

        if not is_ddp or args.local_rank == 0:
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_metrics['loss']:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f}"
            )

            train_losses.append(train_metrics["loss"])
            val_losses.append(val_metrics["loss"])
            plot_learning_curves(train_losses, val_losses, plot_dir, logger)

            model_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "xmed_mean": xmed_mean.detach().cpu(),
                "xmed_std": xmed_std.detach().cpu(),
            }
            early_stopper(val_metrics["loss"], model_dict, epoch, str(checkpoint_dir))

            if early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
        logger.info("DDP process group destroyed")

    logger.info("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train frame autoencoder on x_med.")
    parser.add_argument("--ddp", action="store_true", help="training with Distributed Data Parallel")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DDP")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--cleanup_every", type=int, default=100, help="Memory cleanup every N batches")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden channels in autoencoder")
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension. If None, uses C.")
    parser.add_argument("--recon_loss", type=str, choices=["mse", "l1"], default="mse", help="Reconstruction loss")
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
        "memory_cleanup_interval": args.cleanup_every,
        "grad_accum_steps": args.grad_accum_steps,
        "amp": True,
        "save_dir": ROOT_DIR / "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_channels": args.hidden_channels,
        "latent_dim": args.latent_dim,
        "recon_loss": args.recon_loss,
    }

    main(config=config, args=args)