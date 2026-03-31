import os
import gc
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
from definitions import DATA_PATH
from utils import latin_box

from train_autoencoder_y_batch import Autoencoder_y_batch
from train_autoencoder_x_high import Autoencoder_x_high
from train_autoencoder_x_med import Autoencoder_x_med
from train_autoencoder_x_static import Autoencoder_x_static

from utils_training_autoencoder import (
    ensure_bchw_CxHxW,
    normalize_to_neg_one_one_ignore_nan,
    standardize_tensor,
    denormalize_from_neg_one_one,
    create_logger,
)


# ============================================================
# DDP
# ============================================================

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# Checkpoint helpers
# ============================================================

def load_state_dict_flex(module, ckpt_path, map_location="cpu", strict=True):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    module.load_state_dict(state_dict, strict=strict)
    return ckpt


# ============================================================
# Dataset
# ============================================================

def build_dataloader(cfg, logger, is_ddp=False, rank=0, world_size=1):
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

    shared_cache_dir = Path(cfg["sample_dir"]) / "dataset_cache"
    full_dataset = DengueDataset(
        viirs, era5, static, x_spatial, y,
        bbox=latin_box(),
        skip_era5_bounds=True,
        cache_dir=shared_cache_dir,
        num_zones=num_zones,
        loss_fn=cfg.get("loss_fn", "mse"),
    )

    logger.info(f"Dataset size: {len(full_dataset)}")

    def _worker_init_fn(_):
        torch.set_num_threads(1)

    loader_kwargs = dict(
        dataset=full_dataset,
        batch_size=cfg["per_proc_batch_size"],
        shuffle=False,
        collate_fn=collate_skip_none,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        worker_init_fn=_worker_init_fn if cfg["num_workers"] > 0 else None,
        drop_last=False,
    )

    sampler = None
    if is_ddp:
        sampler = DistributedSampler(
            full_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        loader_kwargs["sampler"] = sampler

    loader = DataLoader(**loader_kwargs)

    y_min = float(ds_cases["dengue_total"].min().values)
    y_max = float(ds_cases["dengue_total"].max().values)

    viirs_arr = np.asarray(viirs.data)
    x_high_min = float(np.nanmin(viirs_arr))
    x_high_max = float(np.nanmax(viirs_arr))

    return loader, sampler, {
        "y_min": y_min,
        "y_max": y_max,
        "x_high_min": x_high_min,
        "x_high_max": x_high_max,
    }


# ============================================================
# Frozen autoencoders
# ============================================================

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


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


def build_frozen_autoencoders(cfg, device, logger, loader):
    base_vae_y = build_base_kl_1ch(cfg)
    ae_y = Autoencoder_y_batch(base_vae_y).to(device)
    ckpt_y = load_state_dict_flex(ae_y, cfg["ae_y_checkpoint"], map_location="cpu", strict=True)
    ae_y.eval()
    requires_grad(ae_y, False)
    logger.info(f"Loaded AE y from {cfg['ae_y_checkpoint']}")

    ae_x_high = None
    ckpt_x_high = None
    if cfg["use_x_high_latent"]:
        ae_x_high = Autoencoder_x_high(
            latent_channels=cfg["latent_channels"],
            block_out_channels=tuple(cfg["x_high_block_out_channels"]),
            layers_per_block=cfg["x_high_layers_per_block"],
            norm_num_groups=cfg["x_high_norm_num_groups"],
        ).to(device)
        ckpt_x_high = load_state_dict_flex(ae_x_high, cfg["ae_x_high_checkpoint"], map_location="cpu", strict=True)
        ae_x_high.eval()
        requires_grad(ae_x_high, False)
        logger.info(f"Loaded AE x_high from {cfg['ae_x_high_checkpoint']}")

    ae_x_med = None
    ckpt_x_med = None
    if cfg["use_x_med_latent"]:
        first_batch = next(iter(loader))
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
        ckpt_x_med = load_state_dict_flex(ae_x_med, cfg["ae_x_med_checkpoint"], map_location="cpu", strict=True)
        ae_x_med.eval()
        requires_grad(ae_x_med, False)
        logger.info(
            f"Loaded AE x_med from {cfg['ae_x_med_checkpoint']} | "
            f"in_channels={in_channels_x_med}, latent_dim={latent_dim_x_med}"
        )

    ae_x_static = None
    ckpt_x_static = None
    if cfg["use_x_static_latent"]:
        base_vae_static = build_base_kl_1ch(cfg)
        ae_x_static = Autoencoder_x_static(base_vae_static).to(device)
        ckpt_x_static = load_state_dict_flex(ae_x_static, cfg["ae_x_static_checkpoint"], map_location="cpu", strict=True)
        ae_x_static.eval()
        requires_grad(ae_x_static, False)
        logger.info(f"Loaded AE x_static from {cfg['ae_x_static_checkpoint']}")

    return ae_y, ae_x_high, ae_x_med, ae_x_static, ckpt_y, ckpt_x_high, ckpt_x_med, ckpt_x_static


# ============================================================
# Encode / decode helpers
# ============================================================

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


def decode_y_latents(ae_y, latents, scale_factor=1.0):
    with torch.no_grad():
        z = latents / scale_factor

        if hasattr(ae_y, "vae") and hasattr(ae_y, "post"):
            recon_up = ae_y.vae.decode(z).sample
            recon = ae_y.post(recon_up)
            return recon

        if hasattr(ae_y, "decode") and callable(ae_y.decode):
            return ae_y.decode(z)

        raise AttributeError("Autoencoder_y_batch does not expose a supported decoding interface")


# ============================================================
# DiT
# ============================================================

def build_dit(cfg, latent_size, device):
    model = DiT_models[cfg["model"]](
        input_size=latent_size,
        in_channels=cfg["latent_channels"],
        climate_channel=cfg["climate_channel"],
        image_time_length=cfg["image_time_length"],
        learn_sigma=cfg["learn_sigma"],
    ).to(device)
    return model


# ============================================================
# Conditioning prep
# ============================================================

def prepare_batch_and_conditioning(
    batch,
    ae_y,
    ae_x_high,
    ae_x_med,
    ae_x_static,
    device,
    stats,
    norm_stats,
    cfg,
):
    x_high, x_med, x_static, x_cond, y_batch = [b.to(device, non_blocking=True) for b in batch]

    y_batch = ensure_bchw_CxHxW(y_batch, C=1, H=86, W=86)
    valid_mask = (~torch.isnan(y_batch)).bool()
    if not valid_mask.any():
        return None

    y_batch_norm = normalize_to_neg_one_one_ignore_nan(
        y_batch.float(),
        stats["y_min"],
        stats["y_max"],
        fill_value=-2.0,
    )

    x_high_norm = None
    if ae_x_high is not None:
        x_high = ensure_bchw_CxHxW(x_high, C=3, H=1024, W=1024)
        x_high_norm = normalize_to_neg_one_one_ignore_nan(
            x_high.float(),
            stats["x_high_min"],
            stats["x_high_max"],
            fill_value=-2.0,
        )

    x_med_norm = None
    if ae_x_med is not None:
        xmed_mean = norm_stats["xmed_mean"]
        xmed_std = norm_stats["xmed_std"]
        x_med_norm = standardize_tensor(x_med.float(), xmed_mean, xmed_std)

    x_static_norm = None
    if ae_x_static is not None:
        x_static_norm = normalize_to_neg_one_one_ignore_nan(
            x_static.float(),
            norm_stats["xstatic_min"],
            norm_stats["xstatic_max"],
            fill_value=-2.0,
        )

    with torch.no_grad():
        y_latents = encode_latents(
            ae_y,
            y_batch_norm,
            sample_posterior=False,
            scale_factor=cfg["latent_scaling_factor"],
        )

        x_high_latents = None
        if ae_x_high is not None:
            x_high_latents = encode_latents(
                ae_x_high,
                x_high_norm,
                sample_posterior=False,
                scale_factor=cfg["x_high_latent_scaling_factor"],
            )

        x_med_latents = None
        if ae_x_med is not None:
            x_med_latents = encode_latents(
                ae_x_med,
                x_med_norm,
                sample_posterior=False,
                scale_factor=cfg["x_med_latent_scaling_factor"],
            )

        x_static_latents = None
        if ae_x_static is not None:
            x_static_latents = encode_latents(
                ae_x_static,
                x_static_norm,
                sample_posterior=False,
                scale_factor=cfg["x_static_latent_scaling_factor"],
            )

    return {
        "target_latents": y_latents,
        "target_y_norm": y_batch_norm,
        "valid_mask": valid_mask,
        "x_high_latents": x_high_latents,
        "x_med_latents": x_med_latents,
        "x_static_latents": x_static_latents,
        "x_cond": x_cond,
    }


# ============================================================
# Saving
# ============================================================

def save_prediction_artifact(sample_folder_dir, index, pred_y_denorm):
    pred_np = pred_y_denorm.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(sample_folder_dir, f"{index:06d}.npy"), pred_np)


def create_npz_from_sample_folder(sample_dir):
    files = sorted(
        [f for f in os.listdir(sample_dir) if f.endswith(".npy")]
    )
    samples = []
    for fname in tqdm(files, desc="Building npz from samples"):
        arr = np.load(os.path.join(sample_dir, fname))
        samples.append(arr.astype(np.float32))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}]")
    return npz_path


# ============================================================
# Main
# ============================================================

def main(args):
    torch.set_grad_enabled(False)

    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)

        seed = args.global_seed * world_size + rank
        torch.manual_seed(seed)
        np.random.seed(seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"sample_dit_{timestamp}"
        run_dir = Path(args.sample_dir) / run_name
        log_dir = run_dir / "logs"
        out_dir = run_dir / "predictions"

        is_main = rank == 0
        if is_main:
            out_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

        logger = create_logger(str(log_dir), is_main_process=is_main)
        logger.info(f"Starting sampling rank={rank}/{world_size} on cuda:{device}")

        cfg = {
            "sample_dir": str(run_dir),
            "batch_size": args.per_proc_batch_size,
            "per_proc_batch_size": args.per_proc_batch_size,
            "num_workers": args.num_workers,
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
            "use_x_high_latent": args.use_x_high_latent,
            "use_x_med_latent": args.use_x_med_latent,
            "use_x_static_latent": args.use_x_static_latent,
            "use_climate_var": args.use_climate_var,
            "climate_channel": args.climate_channel,
            "image_time_length": args.image_time_length,
            "learn_sigma": args.learn_sigma,
            "model": args.model,
            "ae_y_checkpoint": args.ae_y_checkpoint,
            "ae_x_high_checkpoint": args.ae_x_high_checkpoint,
            "ae_x_med_checkpoint": args.ae_x_med_checkpoint,
            "ae_x_static_checkpoint": args.ae_x_static_checkpoint,
            "x_high_block_out_channels": tuple(args.x_high_block_out_channels),
            "x_high_layers_per_block": args.x_high_layers_per_block,
            "x_high_norm_num_groups": args.x_high_norm_num_groups,
            "x_med_hidden_channels": args.x_med_hidden_channels,
            "x_med_latent_dim": args.x_med_latent_dim,
            "memory_cleanup_interval": args.memory_cleanup_interval,
        }

        loader, sampler, stats = build_dataloader(
            cfg=cfg,
            logger=logger,
            is_ddp=True,
            rank=rank,
            world_size=world_size,
        )

        (
            ae_y,
            ae_x_high,
            ae_x_med,
            ae_x_static,
            ckpt_y,
            ckpt_x_high,
            ckpt_x_med,
            ckpt_x_static,
        ) = build_frozen_autoencoders(cfg, device, logger, loader)

        norm_stats = {}

        if ae_x_med is not None:
            if "xmed_mean" not in ckpt_x_med or "xmed_std" not in ckpt_x_med:
                raise KeyError("x_med checkpoint must contain 'xmed_mean' and 'xmed_std'")
            norm_stats["xmed_mean"] = ckpt_x_med["xmed_mean"].to(device)
            norm_stats["xmed_std"] = ckpt_x_med["xmed_std"].to(device)

        if ae_x_static is not None:
            if "data_min" not in ckpt_x_static or "data_max" not in ckpt_x_static:
                raise KeyError("x_static checkpoint must contain 'data_min' and 'data_max'")
            norm_stats["xstatic_min"] = ckpt_x_static["data_min"]
            norm_stats["xstatic_max"] = ckpt_x_static["data_max"]

        first_batch = next(iter(loader))
        first_prepared = prepare_batch_and_conditioning(
            batch=first_batch,
            ae_y=ae_y,
            ae_x_high=ae_x_high,
            ae_x_med=ae_x_med,
            ae_x_static=ae_x_static,
            device=device,
            stats=stats,
            norm_stats=norm_stats,
            cfg=cfg,
        )
        if first_prepared is None:
            raise RuntimeError("Could not infer latent shape from first batch")

        latent_shape = first_prepared["target_latents"].shape[1:]
        latent_size = first_prepared["target_latents"].shape[-1]

        logger.info(f"Target latent shape: {tuple(first_prepared['target_latents'].shape)}")

        model = build_dit(cfg, latent_size, device)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        model.eval()

        diffusion = create_diffusion(str(args.num_sampling_steps))

        total_written = 0
        pbar = tqdm(loader, disable=not is_main, desc="Sampling", leave=False)

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
                stats=stats,
                norm_stats=norm_stats,
                cfg=cfg,
            )
            if prepared is None:
                continue

            bs = prepared["target_latents"].shape[0]
            z = torch.randn((bs, *latent_shape), device=device)

            model_kwargs = {
                "y": None,
                "x_high_latent": prepared["x_high_latents"] if cfg["use_x_high_latent"] else None,
                "x_med_latent": prepared["x_med_latents"] if cfg["use_x_med_latent"] else None,
                "x_static_latent": prepared["x_static_latents"] if cfg["use_x_static_latent"] else None,
                "climate_var": prepared["x_cond"] if cfg["use_climate_var"] else None,
            }

            samples_latent = diffusion.p_sample_loop(
                model.forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )

            pred_y_norm = decode_y_latents(
                ae_y,
                samples_latent,
                scale_factor=cfg["latent_scaling_factor"],
            )

            pred_y = denormalize_from_neg_one_one(
                pred_y_norm,
                stats["y_min"],
                stats["y_max"],
            )

            for i in range(bs):
                global_index = batch_idx * args.per_proc_batch_size * world_size + i * world_size + rank
                if args.max_samples is not None and global_index >= args.max_samples:
                    continue
                save_prediction_artifact(str(out_dir), global_index, pred_y[i:i+1])
                total_written += 1

            if cfg["memory_cleanup_interval"] > 0 and (batch_idx + 1) % cfg["memory_cleanup_interval"] == 0:
                gc.collect()
                torch.cuda.empty_cache()

            del prepared, z, model_kwargs, samples_latent, pred_y_norm, pred_y

        dist.barrier()

        if is_main:
            create_npz_from_sample_folder(str(out_dir))
            logger.info("Sampling finished.")

        dist.barrier()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--sample-dir", type=str, default="samples_dit")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)

    parser.add_argument("--start-date", type=str, default="2012-01-01")
    parser.add_argument("--end-date", type=str, default="2023-12-31")
    parser.add_argument("--loss-fn", type=str, choices=["mse", "poisson"], default="mse")

    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--block-out-channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--layers-per-block", type=int, default=2)
    parser.add_argument("--norm-num-groups", type=int, default=32)

    parser.add_argument("--latent-scaling-factor", type=float, default=1.0)
    parser.add_argument("--x-high-latent-scaling-factor", type=float, default=1.0)
    parser.add_argument("--x-med-latent-scaling-factor", type=float, default=1.0)
    parser.add_argument("--x-static-latent-scaling-factor", type=float, default=1.0)

    parser.add_argument("--use-x-high-latent", action="store_true")
    parser.add_argument("--use-x-med-latent", action="store_true")
    parser.add_argument("--use-x-static-latent", action="store_true")
    parser.add_argument("--use-climate-var", action="store_true")

    parser.add_argument("--climate-channel", type=int, default=None)
    parser.add_argument("--image-time-length", type=int, default=None)
    parser.add_argument("--learn-sigma", action="store_true")

    parser.add_argument("--ae-y-checkpoint", type=str, required=True)
    parser.add_argument("--ae-x-high-checkpoint", type=str, default=None)
    parser.add_argument("--ae-x-med-checkpoint", type=str, default=None)
    parser.add_argument("--ae-x-static-checkpoint", type=str, default=None)

    parser.add_argument("--x-high-block-out-channels", type=int, nargs="+", default=[64, 128, 256, 512, 512, 512])
    parser.add_argument("--x-high-layers-per-block", type=int, default=2)
    parser.add_argument("--x-high-norm-num-groups", type=int, default=32)

    parser.add_argument("--x-med-hidden-channels", type=int, default=32)
    parser.add_argument("--x-med-latent-dim", type=int, default=None)
    parser.add_argument("--memory-cleanup-interval", type=int, default=100)

    args = parser.parse_args()
    main(args)