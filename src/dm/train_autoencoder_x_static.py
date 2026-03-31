import os
import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import xarray as xr

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

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
    normalize_to_neg_one_one_with_minmax_ignore_nan,
    ensure_bchw_CxHxW,
)


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
            nn.AdaptiveAvgPool2d((102, 102)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Autoencoder_x_static(nn.Module):
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

        posterior_mean = posterior.mean
        posterior_logvar = posterior.logvar
        return recon, posterior_mean, posterior_logvar



# ------------------------------------------------------------------
# LOSS
# ------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    L = L_recon + lambda1 * L_grad + lambda2 * L_norm + lambda3 * L_ssim
    versione per 1 canale
    """

    def __init__(
        self,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        learnable_lambdas=False,
        use_sobel=True,
        ssim_window_size=11,
    ):
        super().__init__()

        self.learnable_lambdas = learnable_lambdas
        self.use_sobel = use_sobel
        self.ssim_window_size = ssim_window_size

        if learnable_lambdas:
            self.raw_lambda1 = nn.Parameter(torch.tensor(float(lambda1)))
            self.raw_lambda2 = nn.Parameter(torch.tensor(float(lambda2)))
            self.raw_lambda3 = nn.Parameter(torch.tensor(float(lambda3)))
        else:
            self.register_buffer("fixed_lambda1", torch.tensor(float(lambda1)))
            self.register_buffer("fixed_lambda2", torch.tensor(float(lambda2)))
            self.register_buffer("fixed_lambda3", torch.tensor(float(lambda3)))

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def lambdas(self):
        if self.learnable_lambdas:
            lambda1 = F.softplus(self.raw_lambda1)
            lambda2 = F.softplus(self.raw_lambda2)
            lambda3 = F.softplus(self.raw_lambda3)
        else:
            lambda1 = self.fixed_lambda1
            lambda2 = self.fixed_lambda2
            lambda3 = self.fixed_lambda3
        return lambda1, lambda2, lambda3

    def _sobel_gradients(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return gx, gy

    def _simple_gradients(self, x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    def _compute_normals(self, z):
        """
        z: [B, 1, H, W]
        costruisce pseudo-normali: n = [-dz/dx, -dz/dy, 1]
        ritorna [B, 1, 3, H, W]
        """
        b, c, h, w = z.shape
        if c != 1:
            raise ValueError(f"Expected 1 channel input, got shape {z.shape}")

        gx = torch.zeros_like(z)
        gy = torch.zeros_like(z)

        gx[:, :, :, 1:-1] = 0.5 * (z[:, :, :, 2:] - z[:, :, :, :-2])
        gy[:, :, 1:-1, :] = 0.5 * (z[:, :, 2:, :] - z[:, :, :-2, :])

        nx = -gx
        ny = -gy
        nz = torch.ones_like(z)

        normals = torch.stack([nx, ny, nz], dim=2)   # [B,1,3,H,W]
        normals = F.normalize(normals, p=2, dim=2, eps=1e-6)
        return normals

    def _ssim(self, x, y, window_size=11, C1=0.01**2, C2=0.03**2):
        pad = window_size // 2

        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x2
        sigma_y2 = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y2
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_xy

        ssim_num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        ssim_den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim = ssim_num / (ssim_den + 1e-8)
        return ssim.mean()

    def forward(self, pred, target, valid_mask=None):
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got pred={pred.shape}, target={target.shape}")
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
        if pred.shape[1] != 1:
            raise ValueError(f"Expected tensors with 1 channel, got {pred.shape}")

        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        if valid_mask.shape != target.shape:
            raise ValueError(f"Mask shape mismatch: mask={valid_mask.shape}, target={target.shape}")

        diff_abs = torch.abs(target - pred)
        diff_abs_valid = diff_abs[valid_mask]
        if diff_abs_valid.numel() == 0:
            total = pred.sum() * 0.0
            zero = total.detach()
            return total, {
                "total": total,
                "recon": zero,
                "grad": zero,
                "norm": zero,
                "ssim": zero,
                "lambda1": self.lambdas()[0].detach(),
                "lambda2": self.lambdas()[1].detach(),
                "lambda3": self.lambdas()[2].detach(),
            }

        # recon
        l_recon = diff_abs_valid.mean()

        # grad
        abs_err_full = diff_abs.clone()
        abs_err_full = abs_err_full.masked_fill(~valid_mask, 0.0)

        if self.use_sobel:
            gx, gy = self._sobel_gradients(abs_err_full)
        else:
            gx, gy = self._simple_gradients(abs_err_full)

        l_grad = torch.mean(torch.abs(gx)) + torch.mean(torch.abs(gy))

        # normals
        pred_masked = pred.masked_fill(~valid_mask, 0.0)
        target_masked = target.masked_fill(~valid_mask, 0.0)

        pred_normals = self._compute_normals(pred_masked)
        target_normals = self._compute_normals(target_masked)
        cos_sim = torch.sum(pred_normals * target_normals, dim=2)  # [B,1,H,W]
        l_norm = (1.0 - cos_sim)[valid_mask].mean()

        # ssim
        l_ssim = 1.0 - self._ssim(pred_masked, target_masked, window_size=self.ssim_window_size)

        lambda1, lambda2, lambda3 = self.lambdas()
        total = l_recon + lambda1 * l_grad + lambda2 * l_norm + lambda3 * l_ssim

        stats = {
            "total": total,
            "recon": l_recon.detach(),
            "grad": l_grad.detach(),
            "norm": l_norm.detach(),
            "ssim": l_ssim.detach(),
            "lambda1": lambda1.detach(),
            "lambda2": lambda2.detach(),
            "lambda3": lambda3.detach(),
        }
        return total, stats


# ------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------

def train_one_epoch(
    model,
    loss_module,
    train_loader,
    optimizer,
    device,
    data_min,
    data_max,
    beta_kl=0.01,
    use_kl=False,
    scaler=None,
    cfg=None,
    logger=None,
    is_ddp=False,
):
    model.train()
    loss_module.train()

    running_loss = 0.0
    running_recon = 0.0
    running_grad = 0.0
    running_norm = 0.0
    running_ssim = 0.0
    running_kl = 0.0
    n_samples = 0
    num_steps = 0

    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1) if cfg else 1))
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        x_high, x_med, x_static, x_cond, y_batch = [
            b.to(device, non_blocking=True) for b in batch
        ]

        x_static = ensure_bchw_CxHxW(x_static.float(), C=1, H=102, W=102)

        valid_mask = (~torch.isnan(x_static)).bool()
        if not valid_mask.any():
            continue

        x_in = normalize_to_neg_one_one_with_minmax_ignore_nan(
            x_static,
            data_min,
            data_max,
            fill_value=-2.0,
        )

        with torch.amp.autocast(
            enabled=cfg.get("amp", True) if cfg else True,
            device_type=cfg.get("device", "cuda") if cfg else "cuda",
        ):
            recon, posterior_mean, posterior_logvar = model(x_in)

            if recon.shape != x_in.shape:
                raise ValueError(f"Shape mismatch: recon {recon.shape} vs x_in {x_in.shape}")

            base_loss, loss_stats = loss_module(recon, x_in, valid_mask=valid_mask)
            kl_loss = -0.5 * torch.mean(
                1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp()
            )
            total_loss = base_loss + beta_kl * kl_loss if use_kl else base_loss

            if torch.isnan(total_loss):
                if logger:
                    logger.warning("[train] Loss is NaN, skipping this batch")
                continue

            scaled_loss = total_loss / grad_accum_steps

        if scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        num_steps += 1
        should_step = (num_steps % grad_accum_steps) == 0

        if should_step:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = x_in.size(0)
        n_samples += bs
        running_loss += total_loss.item() * bs
        running_recon += loss_stats["recon"].item() * bs
        running_grad += loss_stats["grad"].item() * bs
        running_norm += loss_stats["norm"].item() * bs
        running_ssim += loss_stats["ssim"].item() * bs
        running_kl += kl_loss.item() * bs

        del x_high, x_med, x_static, x_cond, y_batch, x_in, recon, posterior_mean, posterior_logvar, total_loss, valid_mask

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
        return {
            "loss": float("nan"),
            "recon": float("nan"),
            "grad": float("nan"),
            "norm": float("nan"),
            "ssim": float("nan"),
            "kl": float("nan"),
            "lambda1": float("nan"),
            "lambda2": float("nan"),
            "lambda3": float("nan"),
        }

    if is_ddp:
        metrics_tensor = torch.tensor(
            [running_loss, running_recon, running_grad, running_norm, running_ssim, running_kl, n_samples],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
        running_loss = metrics_tensor[0].item()
        running_recon = metrics_tensor[1].item()
        running_grad = metrics_tensor[2].item()
        running_norm = metrics_tensor[3].item()
        running_ssim = metrics_tensor[4].item()
        running_kl = metrics_tensor[5].item()
        n_samples = int(metrics_tensor[6].item())

    lm = loss_module.module if hasattr(loss_module, "module") else loss_module
    lambda1, lambda2, lambda3 = lm.lambdas()

    return {
        "loss": running_loss / n_samples,
        "recon": running_recon / n_samples,
        "grad": running_grad / n_samples,
        "norm": running_norm / n_samples,
        "ssim": running_ssim / n_samples,
        "kl": running_kl / n_samples,
        "lambda1": lambda1.item(),
        "lambda2": lambda2.item(),
        "lambda3": lambda3.item(),
    }


@torch.inference_mode()
def evaluate_one_epoch(
    model,
    loss_module,
    loader,
    device,
    data_min,
    data_max,
    beta_kl=0.01,
    use_kl=False,
    desc="Eval",
    cfg=None,
    logger=None,
    is_ddp=False,
):
    model.eval()
    loss_module.eval()

    running_loss = 0.0
    running_recon = 0.0
    running_grad = 0.0
    running_norm = 0.0
    running_ssim = 0.0
    running_kl = 0.0
    n_samples = 0

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue

        x_high, x_med, x_static, x_cond, y_batch = [
            b.to(device, non_blocking=True) for b in batch
        ]

        x_static = ensure_bchw_CxHxW(x_static.float(), C=1, H=102, W=102)

        valid_mask = (~torch.isnan(x_static)).bool()
        if not valid_mask.any():
            continue

        x_in = normalize_to_neg_one_one_with_minmax_ignore_nan(
            x_static,
            data_min,
            data_max,
            fill_value=-2.0,
        )

        with torch.amp.autocast(
            enabled=cfg.get("amp", True) if cfg else True,
            device_type=cfg.get("device", "cuda") if cfg else "cuda",
        ):
            recon, posterior_mean, posterior_logvar = model(x_in)

            if recon.shape != x_in.shape:
                raise ValueError(f"Shape mismatch: recon {recon.shape} vs x_in {x_in.shape}")

            base_loss, loss_stats = loss_module(recon, x_in, valid_mask=valid_mask)
            kl_loss = -0.5 * torch.mean(
                1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp()
            )
            total_loss = base_loss + beta_kl * kl_loss if use_kl else base_loss

            if torch.isnan(total_loss):
                if logger:
                    logger.warning(f"[{desc}] Loss is NaN, skipping this batch")
                continue

        bs = x_in.size(0)
        n_samples += bs
        running_loss += total_loss.item() * bs
        running_recon += loss_stats["recon"].item() * bs
        running_grad += loss_stats["grad"].item() * bs
        running_norm += loss_stats["norm"].item() * bs
        running_ssim += loss_stats["ssim"].item() * bs
        running_kl += kl_loss.item() * bs

        del x_high, x_med, x_static, x_cond, y_batch, x_in, recon, posterior_mean, posterior_logvar, total_loss, valid_mask

        cleanup_every = int(cfg.get("memory_cleanup_interval", 0) if cfg else 0)
        if cleanup_every > 0 and (batch_idx + 1) % cleanup_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if n_samples == 0:
        return {
            "loss": float("nan"),
            "recon": float("nan"),
            "grad": float("nan"),
            "norm": float("nan"),
            "ssim": float("nan"),
            "kl": float("nan"),
            "lambda1": float("nan"),
            "lambda2": float("nan"),
            "lambda3": float("nan"),
        }

    if is_ddp:
        metrics_tensor = torch.tensor(
            [running_loss, running_recon, running_grad, running_norm, running_ssim, running_kl, n_samples],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
        running_loss = metrics_tensor[0].item()
        running_recon = metrics_tensor[1].item()
        running_grad = metrics_tensor[2].item()
        running_norm = metrics_tensor[3].item()
        running_ssim = metrics_tensor[4].item()
        running_kl = metrics_tensor[5].item()
        n_samples = int(metrics_tensor[6].item())

    lm = loss_module.module if hasattr(loss_module, "module") else loss_module
    lambda1, lambda2, lambda3 = lm.lambdas()

    return {
        "loss": running_loss / n_samples,
        "recon": running_recon / n_samples,
        "grad": running_grad / n_samples,
        "norm": running_norm / n_samples,
        "ssim": running_ssim / n_samples,
        "kl": running_kl / n_samples,
        "lambda1": lambda1.item(),
        "lambda2": lambda2.item(),
        "lambda3": lambda3.item(),
    }


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main(config: dict | None = None, args=None):
    run_dir, checkpoint_dir, log_dir, plot_dir, logger = setup_run_dirs_and_logger(
        config=config,
        default_save_dir=ROOT_DIR / "checkpoints",
        run_prefix="run",
    )

    logger.info("Starting VAE training run on x_static")
    logger.info(f"Run directory: {run_dir}")

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
        "lambda1": 1.0,
        "lambda2": 1.0,
        "lambda3": 1.0,
        "learnable_lambdas": False,
        "use_sobel": True,
        "ssim_window_size": 11,
    }
    if config:
        default_config.update(config)
    cfg = default_config

    is_ddp, device, world_size = setup_device_and_distributed(args, cfg, logger)

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
        viirs,
        era5,
        static,
        x_spatial,
        y,
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

    # min/max per x_static
    logger.info("Computing global min/max on x_static...")
    data_min = float(static.data.min())
    data_max = float(static.data.max())
    logger.info(f"x_static min: {data_min:.6f}, max: {data_max:.6f}")

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

    model = Autoencoder_x_static(vae).to(device)
    loss_module = CompositeLoss(
        lambda1=cfg["lambda1"],
        lambda2=cfg["lambda2"],
        lambda3=cfg["lambda3"],
        learnable_lambdas=cfg["learnable_lambdas"],
        use_sobel=cfg["use_sobel"],
        ssim_window_size=cfg["ssim_window_size"],
    ).to(device)

    model = wrap_model_for_parallel(model, args, device)
    loss_module = wrap_model_for_parallel(loss_module, args, device)

    if cfg["learnable_lambdas"]:
        optim_params = list(model.parameters()) + list(loss_module.parameters())
    else:
        optim_params = list(model.parameters())

    optimizer = AdamW(optim_params, lr=cfg["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler(enabled=cfg["amp"])

    es_config = SimpleNamespace(patience=cfg["patience"], min_patience=0)
    early_stopper = EarlyStopping(es_config, verbose=True)

    train_losses = []
    val_losses = []

    logger.info(f"Starting training for {cfg['num_epochs']} epochs...")

    for epoch in range(1, cfg["num_epochs"] + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model=model,
            loss_module=loss_module,
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
            loss_module=loss_module,
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

        if not is_ddp or args.local_rank == 0:
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_metrics['loss']:.6f} | "
                f"Recon: {train_metrics['recon']:.6f} | "
                f"Grad: {train_metrics['grad']:.6f} | "
                f"Norm: {train_metrics['norm']:.6f} | "
                f"SSIM: {train_metrics['ssim']:.6f} | "
                f"KL: {train_metrics['kl']:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"ValRecon: {val_metrics['recon']:.6f} | "
                f"ValGrad: {val_metrics['grad']:.6f} | "
                f"ValNorm: {val_metrics['norm']:.6f} | "
                f"ValSSIM: {val_metrics['ssim']:.6f} | "
                f"ValKL: {val_metrics['kl']:.6f} | "
                f"lambda1: {train_metrics['lambda1']:.6f} | "
                f"lambda2: {train_metrics['lambda2']:.6f} | "
                f"lambda3: {train_metrics['lambda3']:.6f}"
            )

            train_losses.append(train_metrics["loss"])
            val_losses.append(val_metrics["loss"])
            plot_learning_curves(train_losses, val_losses, plot_dir, logger)

            model_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss_module_state": loss_module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "data_min": data_min,
                "data_max": data_max,
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
    parser = argparse.ArgumentParser(description="Train VAE on x_static.")
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
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for L_grad")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Weight for L_norm")
    parser.add_argument("--lambda3", type=float, default=1.0, help="Weight for L_ssim")
    parser.add_argument("--learnable_lambdas", action="store_true", help="Learn lambda1/lambda2/lambda3")
    parser.add_argument("--no_sobel", action="store_true", help="Disable Sobel gradients")
    parser.add_argument("--ssim_window_size", type=int, default=11, help="SSIM window size")
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
        "norm_num_groups": args.norm_num_groups,
        "use_kl": args.use_kl,
        "memory_cleanup_interval": args.cleanup_every,
        "grad_accum_steps": args.grad_accum_steps,
        "amp": True,
        "save_dir": ROOT_DIR / "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "loss_fn": args.loss_fn,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "lambda3": args.lambda3,
        "learnable_lambdas": args.learnable_lambdas,
        "use_sobel": not args.no_sobel,
        "ssim_window_size": args.ssim_window_size,
    }

    main(config=config, args=args)