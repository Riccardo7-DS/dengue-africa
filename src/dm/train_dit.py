import os
import gc
import argparse
import logging
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from DiT import DiT_models
from diffusion import create_diffusion

from models import (
    VIIRSData,
    ERA5Daily,
    StaticLayer,
    XarraySpatioTemporalDataset,
    DengueDataset,
    collate_skip_none,
)
from definitions import DATA_PATH, ROOT_DIR
from utils import latin_box

from train_autoencoder_y_batch import Autoencoder_y_batch
from train_autoencoder_x_high import Autoencoder_x_high
from train_autoencoder_x_med import Autoencoder_x_med
from train_autoencoder_x_static import Autoencoder_x_static


# ============================================================
# Utils
# ============================================================

def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
        raise ValueError(f"Expected single channel, got shape {tuple(x.shape)}")

    if x.shape[2] != 86 or x.shape[3] != 86:
        raise ValueError(f"Expected H=W=86, got shape {tuple(x.shape)}")

    return x


def normalize_to_neg_one_one_ignore_nan(x, data_min, data_max, fill_value=-2.0):
    data_min = torch.as_tensor(data_min, device=x.device, dtype=x.dtype)
    data_max = torch.as_tensor(data_max, device=x.device, dtype=x.dtype)
    denom = torch.clamp(data_max - data_min, min=1e-6)

    valid_mask = ~torch.isnan(x)
    x_norm = x.clone()
    x_norm[valid_mask] = 2.0 * (x[valid_mask] - data_min) / denom - 1.0
    x_norm[~valid_mask] = fill_value
    return x_norm


def save_checkpoint(path, model, ema, optimizer, epoch, train_steps, args_dict, extra=None):
    ckpt = {
        "epoch": epoch,
        "train_steps": train_steps,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args_dict,
    }
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, model, ema=None, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if ema is not None and "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    train_steps = ckpt.get("train_steps", 0)
    return start_epoch, train_steps, ckpt


def load_model_flex(module, checkpoint_path, map_location="cpu", strict=True):
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    module.load_state_dict(state_dict, strict=strict)
    return module


# ============================================================
# Logging
# ============================================================

def create_logger(log_dir, is_main_process=True):
    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger("train_dit_latent")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
    else:
        logger = logging.getLogger("train_dit_latent")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        return logger


# ============================================================
# Dataset
# ============================================================

def build_dataloaders(cfg, logger, is_ddp=False, rank=0, world_size=1):
    viirs_data_path = DATA_PATH / "modis" / "VIIRS_nightlight"
    era5_path = DATA_PATH / "ERA5" / "Latin_america"
    risk_raster_path = DATA_PATH / "riskmaps_public main data" / "DEN_riskmap_wmean_masked.tif"
    admin_path = DATA_PATH / "dengue_cases"

    start_date = cfg["start_date"]
    end_date = cfg["end_date"]

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

    def _worker_init_fn(_):
        torch.set_num_threads(1)

    train_loader_kwargs = dict(
        dataset=full_dataset,
        batch_size=cfg["batch_size"],
        shuffle=(not is_ddp),
        collate_fn=collate_skip_none,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        worker_init_fn=_worker_init_fn if cfg["num_workers"] > 0 else None,
        drop_last=True,
    )

    if cfg["num_workers"] > 0:
        train_loader_kwargs["persistent_workers"] = False

    train_sampler = None
    if is_ddp:
        train_sampler = DistributedSampler(
            full_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg["global_seed"],
        )
        train_loader_kwargs["sampler"] = train_sampler
        del train_loader_kwargs["shuffle"]

    train_loader = DataLoader(**train_loader_kwargs)

    data_min = float(ds_cases["dengue_total"].min().values)
    data_max = float(ds_cases["dengue_total"].max().values)
    logger.info(f"Global target min/max from dengue_total: {data_min:.6f} / {data_max:.6f}")

    return train_loader, train_sampler, data_min, data_max


# ============================================================
# Frozen autoencoders
# ============================================================

def build_base_kl_1ch(cfg):
    return AutoencoderKL(
        in_channels=1,
        out_channels=1,
        latent_channels=cfg["latent_channels"],
        down_block_types=("DownEncoderBlock2D",) * len(cfg["block_out_channels"]),
        up_block_types=("UpDecoderBlock2D",) * len(cfg["block_out_channels"]),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg["layers_per_block"],
        norm_num_groups=cfg["norm_num_groups"],
    )


def build_frozen_autoencoders(cfg, device, logger, train_loader):
    base_vae_y = build_base_kl_1ch(cfg)
    ae_y = Autoencoder_y_batch(base_vae_y).to(device)
    ae_y = load_model_flex(ae_y, cfg["ae_y_checkpoint"], map_location="cpu", strict=True)
    ae_y.eval()
    requires_grad(ae_y, False)
    logger.info(f"Loaded frozen AE [ae_y] from: {cfg['ae_y_checkpoint']}")

    ae_x_high = None
    if cfg["use_x_high_latent"]:
        ae_x_high = Autoencoder_x_high(
            latent_channels=cfg["latent_channels"],
            block_out_channels=tuple(cfg["x_high_block_out_channels"]),
            layers_per_block=cfg["x_high_layers_per_block"],
            norm_num_groups=cfg["x_high_norm_num_groups"],
        ).to(device)
        ae_x_high = load_model_flex(ae_x_high, cfg["ae_x_high_checkpoint"], map_location="cpu", strict=True)
        ae_x_high.eval()
        requires_grad(ae_x_high, False)
        logger.info(f"Loaded frozen AE [ae_x_high] from: {cfg['ae_x_high_checkpoint']}")

    ae_x_med = None
    if cfg["use_x_med_latent"]:
        first_batch = next(iter(train_loader))
        x_med_sample = first_batch[1].to(device, non_blocking=True)

        if x_med_sample.ndim != 5:
            raise ValueError(f"x_med must be [B,T,C,H,W], got {tuple(x_med_sample.shape)}")

        in_channels_x_med = x_med_sample.shape[2]
        latent_dim_x_med = cfg["x_med_latent_dim"] if cfg["x_med_latent_dim"] is not None else in_channels_x_med

        ae_x_med = Autoencoder_x_med(
            in_channels=in_channels_x_med,
            latent_dim=latent_dim_x_med,
            hidden_channels=cfg["x_med_hidden_channels"],
        ).to(device)
        ae_x_med = load_model_flex(ae_x_med, cfg["ae_x_med_checkpoint"], map_location="cpu", strict=True)
        ae_x_med.eval()
        requires_grad(ae_x_med, False)
        logger.info(
            f"Loaded frozen AE [ae_x_med] from: {cfg['ae_x_med_checkpoint']} | "
            f"in_channels={in_channels_x_med}, latent_dim={latent_dim_x_med}"
        )

    ae_x_static = None
    if cfg["use_x_static_latent"]:
        base_vae_static = build_base_kl_1ch(cfg)
        ae_x_static = Autoencoder_x_static(base_vae_static).to(device)
        ae_x_static = load_model_flex(ae_x_static, cfg["ae_x_static_checkpoint"], map_location="cpu", strict=True)
        ae_x_static.eval()
        requires_grad(ae_x_static, False)
        logger.info(f"Loaded frozen AE [ae_x_static] from: {cfg['ae_x_static_checkpoint']}")

    return ae_y, ae_x_high, ae_x_med, ae_x_static


def encode_latents(ae, x, sample_posterior=False, scale_factor=1.0):
    if ae is None:
        return None

    with torch.no_grad():
        if hasattr(ae, "encode") and callable(ae.encode):
            try:
                z = ae.encode(x, sample_posterior=sample_posterior)
            except TypeError:
                z = ae.encode(x)
            return z * scale_factor

        if hasattr(ae, "pre") and hasattr(ae, "vae"):
            x_in = ae.pre(x)
            enc = ae.vae.encode(x_in)
            posterior = enc.latent_dist
            z = posterior.rsample() if sample_posterior and hasattr(posterior, "rsample") else posterior.mean
            return z * scale_factor

        if hasattr(ae, "vae"):
            enc = ae.vae.encode(x)
            posterior = enc.latent_dist
            z = posterior.rsample() if sample_posterior and hasattr(posterior, "rsample") else posterior.mean
            return z * scale_factor

        raise AttributeError(f"{ae.__class__.__name__} does not expose a supported encoding interface")


# ============================================================
# Build DiT
# ============================================================

def build_dit(cfg, latent_size, device, logger):
    model = DiT_models[cfg["model"]](
        input_size=latent_size,
        in_channels=cfg["latent_channels"],
        climate_channel=cfg["climate_channel"],
        image_time_length=cfg["image_time_length"],
        learn_sigma=cfg["learn_sigma"],
    )
    return model.to(device)


# ============================================================
# Batch prep
# ============================================================

def prepare_batch_and_conditioning(
    batch,
    ae_y,
    ae_x_high,
    ae_x_med,
    ae_x_static,
    device,
    data_min,
    data_max,
    cfg,
):
    x_high, x_med, x_static, x_cond, y_batch = [b.to(device, non_blocking=True) for b in batch]

    y_batch = ensure_bchw_1x86x86(y_batch)
    valid_mask = (~torch.isnan(y_batch)).bool()
    if not valid_mask.any():
        return None

    y_batch_norm = normalize_to_neg_one_one_ignore_nan(
        y_batch,
        data_min,
        data_max,
        fill_value=-2.0,
    )

    with torch.no_grad():
        y_latents = encode_latents(
            ae_y,
            y_batch_norm,
            sample_posterior=cfg["sample_posterior_in_train"],
            scale_factor=cfg["latent_scaling_factor"],
        )

        x_high_latents = None
        if ae_x_high is not None:
            x_high_latents = encode_latents(
                ae_x_high,
                x_high,
                sample_posterior=False,
                scale_factor=cfg["x_high_latent_scaling_factor"],
            )

        x_med_latents = None
        if ae_x_med is not None:
            x_med_latents = encode_latents(
                ae_x_med,
                x_med,
                sample_posterior=False,
                scale_factor=cfg["x_med_latent_scaling_factor"],
            )

        x_static_latents = None
        if ae_x_static is not None:
            x_static_latents = encode_latents(
                ae_x_static,
                x_static,
                sample_posterior=False,
                scale_factor=cfg["x_static_latent_scaling_factor"],
            )

    return {
        "latents": y_latents,
        "target": y_batch_norm,
        "valid_mask": valid_mask,
        "x_high_latents": x_high_latents,
        "x_med_latents": x_med_latents,
        "x_static_latents": x_static_latents,
        "x_cond": x_cond,
    }


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    ema,
    ae_y,
    ae_x_high,
    ae_x_med,
    ae_x_static,
    diffusion,
    train_loader,
    optimizer,
    device,
    data_min,
    data_max,
    cfg,
    logger,
    train_steps,
    is_ddp=False,
):
    model.train()
    running_loss = 0.0
    n_seen = 0

    scaler = torch.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")
    grad_accum_steps = max(1, int(cfg["grad_accum_steps"]))
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, disable=not cfg["is_main_process"], desc="Train", leave=False)

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        prepared = prepare_batch_and_conditioning(
            batch=batch,
            ae_y=ae_y,
            ae_x_high=ae_x_high,
            ae_x_med=ae_x_med,
            ae_x_static=ae_x_static,
            device=device,
            data_min=data_min,
            data_max=data_max,
            cfg=cfg,
        )
        if prepared is None:
            continue

        latents = prepared["latents"]

        t = torch.randint(
            0, diffusion.num_timesteps, (latents.shape[0],), device=device
        )

        model_kwargs = {
            "y": None,
            "x_high_latent": prepared["x_high_latents"] if cfg["use_x_high_latent"] else None,
            "x_med_latent": prepared["x_med_latents"] if cfg["use_x_med_latent"] else None,
            "x_static_latent": prepared["x_static_latents"] if cfg["use_x_static_latent"] else None,
            "climate_var": prepared["x_cond"] if cfg["use_climate_var"] else None,
        }

        with torch.amp.autocast(
            enabled=cfg["amp"] and device.type == "cuda",
            device_type="cuda" if device.type == "cuda" else "cpu",
        ):
            loss_dict = diffusion.training_losses(model, latents, t, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()

        if torch.isnan(loss):
            logger.warning("NaN loss encountered, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if cfg["grad_clip_norm"] is not None and cfg["grad_clip_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model.module if is_ddp else model, decay=cfg["ema_decay"])

        bs = latents.size(0)
        running_loss += loss.item() * bs
        n_seen += bs
        train_steps += 1

        if n_seen > 0:
            pbar.set_postfix(loss=f"{running_loss / n_seen:.6f}")

        if cfg["memory_cleanup_interval"] > 0 and (batch_idx + 1) % cfg["memory_cleanup_interval"] == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del prepared, latents, t, model_kwargs, loss_dict, loss

    if is_ddp:
        loss_tensor = torch.tensor(running_loss, device=device)
        count_tensor = torch.tensor(n_seen, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor.item()
        total_count = int(count_tensor.item())
    else:
        total_loss = running_loss
        total_count = n_seen

    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, train_steps


# ============================================================
# Main
# ============================================================

def main(cfg):
    is_ddp = cfg["ddp"]

    try:
        if is_ddp:
            assert torch.cuda.is_available(), "DDP richiede GPU."
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", cfg["local_rank"]))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            seed = cfg["global_seed"] * world_size + rank
            is_main_process = rank == 0
        else:
            rank = 0
            world_size = 1
            local_rank = 0
            device = torch.device(cfg["device"])
            seed = cfg["global_seed"]
            is_main_process = True

        torch.manual_seed(seed)
        np.random.seed(seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg["save_dir"]) / f"dit_run_{timestamp}"
        ckpt_dir = run_dir / "checkpoints"
        log_dir = run_dir / "logs"

        if is_main_process:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

        logger = create_logger(str(log_dir), is_main_process=is_main_process)
        cfg["is_main_process"] = is_main_process

        logger.info(f"Starting training | ddp={is_ddp} | rank={rank} | device={device}")
        logger.info(f"Run dir: {run_dir}")

        train_loader, train_sampler, data_min, data_max = build_dataloaders(
            cfg=cfg,
            logger=logger,
            is_ddp=is_ddp,
            rank=rank,
            world_size=world_size,
        )

        ae_y, ae_x_high, ae_x_med, ae_x_static = build_frozen_autoencoders(
            cfg=cfg,
            device=device,
            logger=logger,
            train_loader=train_loader,
        )

        first_batch = next(iter(train_loader))
        first_prepared = prepare_batch_and_conditioning(
            batch=first_batch,
            ae_y=ae_y,
            ae_x_high=ae_x_high,
            ae_x_med=ae_x_med,
            ae_x_static=ae_x_static,
            device=device,
            data_min=data_min,
            data_max=data_max,
            cfg=cfg,
        )
        if first_prepared is None:
            raise RuntimeError("Impossibile inferire i latenti dal primo batch.")

        first_latents = first_prepared["latents"]
        latent_size = first_latents.shape[-1]
        logger.info(f"Target latent shape inferred: {tuple(first_latents.shape)}")

        if first_prepared["x_high_latents"] is not None:
            logger.info(f"x_high latent shape: {tuple(first_prepared['x_high_latents'].shape)}")
        if first_prepared["x_med_latents"] is not None:
            logger.info(f"x_med latent shape: {tuple(first_prepared['x_med_latents'].shape)}")
        if first_prepared["x_static_latents"] is not None:
            logger.info(f"x_static latent shape: {tuple(first_prepared['x_static_latents'].shape)}")

        model = build_dit(cfg, latent_size, device, logger)
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)

        if is_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        diffusion = create_diffusion(timestep_respacing="")

        optimizer = AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.999),
        )

        update_ema(ema, model.module if is_ddp else model, decay=0)

        start_epoch = 1
        train_steps = 0

        if cfg["resume_from"] is not None:
            start_epoch, train_steps, _ = load_checkpoint(
                cfg["resume_from"],
                model.module if is_ddp else model,
                ema=ema,
                optimizer=optimizer,
                map_location=device,
            )
            logger.info(f"Resumed from {cfg['resume_from']} | start_epoch={start_epoch} | train_steps={train_steps}")

        logger.info(f"Training for {cfg['num_epochs']} epochs...")

        for epoch in range(start_epoch, cfg["num_epochs"] + 1):
            if is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss, train_steps = train_one_epoch(
                model=model,
                ema=ema,
                ae_y=ae_y,
                ae_x_high=ae_x_high,
                ae_x_med=ae_x_med,
                ae_x_static=ae_x_static,
                diffusion=diffusion,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                data_min=data_min,
                data_max=data_max,
                cfg=cfg,
                logger=logger,
                train_steps=train_steps,
                is_ddp=is_ddp,
            )

            if is_main_process:
                logger.info(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

                last_ckpt = ckpt_dir / "last.pt"
                save_checkpoint(
                    path=str(last_ckpt),
                    model=model.module if is_ddp else model,
                    ema=ema,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_steps=train_steps,
                    args_dict=cfg,
                    extra={
                        "data_min": data_min,
                        "data_max": data_max,
                    },
                )

                if epoch % cfg["save_every_epochs"] == 0:
                    epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
                    save_checkpoint(
                        path=str(epoch_ckpt),
                        model=model.module if is_ddp else model,
                        ema=ema,
                        optimizer=optimizer,
                        epoch=epoch,
                        train_steps=train_steps,
                        args_dict=cfg,
                        extra={
                            "data_min": data_min,
                            "data_max": data_max,
                        },
                    )
                    logger.info(f"Saved checkpoint: {epoch_ckpt}")

            if is_ddp:
                dist.barrier()

        logger.info("Training finished.")

    finally:
        cleanup_ddp()


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT on multiple frozen AE latents.")

    parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--save_dir", type=str, default=str(ROOT_DIR / "checkpoints_dit"))
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--global_seed", type=int, default=0)

    parser.add_argument("--start_date", type=str, default="2012-01-01")
    parser.add_argument("--end_date", type=str, default="2023-12-31")
    parser.add_argument("--loss_fn", type=str, choices=["mse", "poisson"], default="mse")

    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--block_out_channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--layers_per_block", type=int, default=2)
    parser.add_argument("--norm_num_groups", type=int, default=32)

    parser.add_argument("--latent_scaling_factor", type=float, default=1.0)
    parser.add_argument("--x_high_latent_scaling_factor", type=float, default=1.0)
    parser.add_argument("--x_med_latent_scaling_factor", type=float, default=1.0)
    parser.add_argument("--x_static_latent_scaling_factor", type=float, default=1.0)

    parser.add_argument("--sample_posterior_in_train", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--memory_cleanup_interval", type=int, default=100)
    parser.add_argument("--save_every_epochs", type=int, default=5)

    parser.add_argument("--use_x_high_latent", action="store_true")
    parser.add_argument("--use_x_med_latent", action="store_true")
    parser.add_argument("--use_x_static_latent", action="store_true")
    parser.add_argument("--use_climate_var", action="store_true")

    parser.add_argument("--climate_channel", type=int, default=None)
    parser.add_argument("--image_time_length", type=int, default=None)
    parser.add_argument("--learn_sigma", action="store_true")

    parser.add_argument("--ae_y_checkpoint", type=str, required=True)
    parser.add_argument("--ae_x_high_checkpoint", type=str, default=None)
    parser.add_argument("--ae_x_med_checkpoint", type=str, default=None)
    parser.add_argument("--ae_x_static_checkpoint", type=str, default=None)

    parser.add_argument("--x_high_block_out_channels", type=int, nargs="+", default=[64, 128, 256, 512, 512, 512])
    parser.add_argument("--x_high_layers_per_block", type=int, default=2)
    parser.add_argument("--x_high_norm_num_groups", type=int, default=32)

    parser.add_argument("--x_med_hidden_channels", type=int, default=32)
    parser.add_argument("--x_med_latent_dim", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = {
        "ddp": args.ddp,
        "local_rank": args.local_rank,
        "device": args.device,
        "save_dir": args.save_dir,
        "resume_from": args.resume_from,
        "model": args.model,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "min_lr": args.min_lr,
        "global_seed": args.global_seed,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "loss_fn": args.loss_fn,
        "latent_channels": args.latent_channels,
        "block_out_channels": tuple(args.block_out_channels),
        "layers_per_block": args.layers_per_block,
        "norm_num_groups": args.norm_num_groups,
        "latent_scaling_factor": args.latent_scaling_factor,
        "x_high_latent_scaling_factor": args.x_high_latent_scaling_factor,
        "x_med_latent_scaling_factor": args.x_med_latent_scaling_factor,
        "x_static_latent_scaling_factor": args.x_static_latent_scaling_factor,
        "sample_posterior_in_train": args.sample_posterior_in_train,
        "amp": args.amp,
        "grad_accum_steps": args.grad_accum_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "ema_decay": args.ema_decay,
        "memory_cleanup_interval": args.memory_cleanup_interval,
        "save_every_epochs": args.save_every_epochs,
        "use_x_high_latent": args.use_x_high_latent,
        "use_x_med_latent": args.use_x_med_latent,
        "use_x_static_latent": args.use_x_static_latent,
        "use_climate_var": args.use_climate_var,
        "climate_channel": args.climate_channel,
        "image_time_length": args.image_time_length,
        "learn_sigma": args.learn_sigma,

        "ae_y_checkpoint": args.ae_y_checkpoint,
        "ae_x_high_checkpoint": args.ae_x_high_checkpoint,
        "ae_x_med_checkpoint": args.ae_x_med_checkpoint,
        "ae_x_static_checkpoint": args.ae_x_static_checkpoint,

        "x_high_block_out_channels": tuple(args.x_high_block_out_channels),
        "x_high_layers_per_block": args.x_high_layers_per_block,
        "x_high_norm_num_groups": args.x_high_norm_num_groups,

        "x_med_hidden_channels": args.x_med_hidden_channels,
        "x_med_latent_dim": args.x_med_latent_dim,
    }

    main(cfg)