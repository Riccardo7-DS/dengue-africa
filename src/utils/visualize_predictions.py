"""
Visualize transformer predictions vs actual dengue case counts.

Loads a checkpoint, runs inference on a handful of samples, and produces
5-panel figures per sample:
  1. VIIRS nightlight input (downsampled for display)
  2. Pixel-level predicted log-rate  [B, 1, 86, 86] → heatmap
  3. Zone-level predicted cases (choropleth)
  4. Zone-level actual cases    (choropleth)
  5. Difference: actual − predicted (diverging)

Usage:
    python src/utils/visualize_predictions.py \
        --checkpoint checkpoints/run_YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
        --n_samples 6 \
        --output_dir outputs/viz_poisson \
        --loss_fn poisson \
        --med_in_ch 18
"""

import argparse
import os
import sys
import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import (
    VIIRSData,
    ERA5Daily,
    StaticLayer,
    XarraySpatioTemporalDataset,
    DengueDataset,
    collate_skip_none,
)
from definitions import DATA_PATH, ROOT_DIR, SM_DATA_PATH
from models.transformer import DenguePredictor
from models.model_utils import SoilMoistureData


def latin_box():
    return [-35.317366, -86.308594, 13.111580, -34.277344]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zone_map_to_image(values: torch.Tensor, zone_map: torch.Tensor, num_zones: int) -> np.ndarray:
    H, W = zone_map.shape
    img = np.full((H, W), np.nan, dtype=np.float32)
    zone_map_np = zone_map.numpy()
    values_np = values.numpy()
    valid = (zone_map_np >= 0) & (zone_map_np < num_zones)
    img[valid] = values_np[zone_map_np[valid]]
    return img


def load_model(checkpoint_path: str, cfg: dict, num_zones: int, device: torch.device):
    model = DenguePredictor(
        med_in_ch=cfg["med_in_ch"],
        num_zones=num_zones,
        swin_model=cfg["swin_model"],
        use_titok=cfg["use_titok"],
        titok_backbone=cfg["titok_backbone"],
        titok_num_latent_tokens=cfg["titok_num_latent_tokens"],
        add_sm=cfg.get("add_sm", False),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def plot_sample(
    sample_idx: int,
    x_high: torch.Tensor,
    x_cond: torch.Tensor,
    pred_log_rate: torch.Tensor,
    zone_pred_rates: torch.Tensor,
    zone_actual: torch.Tensor,
    num_zones: int,
    output_dir: Path,
    timestamp: str,
    burden_map: torch.Tensor = None,
):
    zone_map = x_cond.squeeze().long()
    H, W = zone_map.shape

    viirs_np = x_high[0].numpy()
    viirs_small = torch.nn.functional.interpolate(
        torch.from_numpy(viirs_np)[None, None],
        size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    viirs_small = np.nan_to_num(viirs_small, nan=0.0)

    pred_np = pred_log_rate.squeeze().numpy()

    zone_pred_rounded = zone_pred_rates.round()
    zone_actual_rounded = torch.where(
        torch.isfinite(zone_actual),
        zone_actual.round(),
        torch.full_like(zone_actual, float("nan"))
    )
    zone_diff = torch.where(
        torch.isfinite(zone_actual_rounded),
        zone_actual_rounded - zone_pred_rounded,
        torch.full_like(zone_actual_rounded, float("nan"))
    )

    zone_pred_img   = zone_map_to_image(zone_pred_rounded,   zone_map, num_zones)
    zone_actual_img = zone_map_to_image(zone_actual_rounded, zone_map, num_zones)
    zone_diff_img   = zone_map_to_image(zone_diff,           zone_map, num_zones)

    n_panels = 6 if burden_map is not None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5, 5))
    fig.suptitle(f"Sample {sample_idx}  |  {timestamp}", fontsize=12)

    ax = axes[0]
    im = ax.imshow(viirs_small, cmap="inferno", origin="upper")
    ax.set_title("VIIRS nightlight\n(downsampled)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1]
    im = ax.imshow(pred_np, cmap="YlOrRd", origin="upper")
    ax.set_title("Predicted log-rate\n(pixel level, 86×86)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[2]
    vmax_pred = np.nanpercentile(zone_pred_img, 99) if np.any(np.isfinite(zone_pred_img)) else 1.0
    im = ax.imshow(zone_pred_img, cmap="YlOrRd", origin="upper", vmin=0, vmax=vmax_pred)
    ax.set_title("Predicted cases\n(zone level)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[3]
    vmax_act = np.nanpercentile(zone_actual_img, 99) if np.any(np.isfinite(zone_actual_img)) else 1.0
    im = ax.imshow(zone_actual_img, cmap="Blues", origin="upper", vmin=0, vmax=vmax_act)
    ax.set_title("Actual cases\n(zone level, ground truth)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[4]
    finite_diff = zone_diff_img[np.isfinite(zone_diff_img)]
    absmax = max(np.nanpercentile(np.abs(finite_diff), 99), 1.0) if finite_diff.size > 0 else 1.0
    im = ax.imshow(zone_diff_img, cmap="RdBu", origin="upper", vmin=-absmax, vmax=absmax)
    ax.set_title("Difference\n(actual − predicted)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    if burden_map is not None:
        burden_np = burden_map.numpy()
        ax = axes[5]
        finite_b = burden_np[np.isfinite(burden_np)]
        vmax_b = np.nanpercentile(finite_b, 99) if finite_b.size > 0 else 1.0
        im = ax.imshow(burden_np, cmap="YlOrRd", origin="upper", vmin=0, vmax=vmax_b)
        ax.set_title("Pixel burden\n(cases / admin-grid cell)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    plt.tight_layout()
    out_path = output_dir / f"pred_vs_actual_{sample_idx:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize dengue transformer predictions")
    parser.add_argument("--checkpoint",   required=True,  help="Path to best_model.pth checkpoint")
    parser.add_argument("--n_samples",    type=int, default=6)
    parser.add_argument("--output_dir",   default="outputs/viz")
    parser.add_argument("--split",        default="val", choices=["train", "val"])
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--loss_fn",      default="poisson", choices=["mse", "poisson"])
    parser.add_argument("--med_in_ch",    type=int, default=18)
    parser.add_argument("--swin_model",   default="swin_tiny_patch4_window7_224.ms_in1k")
    parser.add_argument("--use_titok",    action="store_true")
    parser.add_argument("--titok_backbone",          default="vit_base_patch16_224.mae")
    parser.add_argument("--titok_num_latent_tokens", type=int, default=32)
    parser.add_argument("--train_split",  type=float, default=0.8)
    parser.add_argument("--add_sm",       action="store_true",
                        help="Enable soil moisture branch (restricts data to 2016-2022)")
    parser.add_argument("--sm_data_path", type=str, default=SM_DATA_PATH,
                        help="Root directory of SM_[A|D]_YYYYMMDD.tif files (required with --add_sm)")
    parser.add_argument("--gpw_path", type=str, default=None,
                        help="Path to GPWv4 data directory. If given, adds a per-capita risk panel.")
    args = parser.parse_args()

    if args.add_sm and not args.sm_data_path:
        parser.error("--sm_data_path is required when --add_sm is set")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viirs_data_path  = DATA_PATH / "modis" / "VIIRS_nightlight"
    era5_path        = DATA_PATH / "ERA5" / "Latin_america"
    risk_raster_path = DATA_PATH / "riskmaps" / "DEN_riskmap_wmean_masked.tif"
    admin_path       = DATA_PATH / "dengue_cases"
    shared_cache_dir = ROOT_DIR / "checkpoints" / "dataset_cache"
    start_date, end_date = "2012-01-01", "2023-12-31"
    if args.add_sm:
        start_date = max(start_date, "2016-01-01")
        end_date   = min(end_date,   "2022-12-31")

    print("Loading datasets...")
    ds_cases = (
        xr.open_mfdataset(os.path.join(admin_path, "*.nc"))
        .sel(time=slice(start_date, end_date))
        .chunk(chunks={"time": 1})
    )
    num_zones = len(np.unique(ds_cases["FAO_GAUL_code"].values))
    ds_cases.load()
    import pandas as pd
    _dc_vars   = {v: (ds_cases[v].dims,  np.asarray(ds_cases[v].values))  for v in ds_cases.data_vars}
    _dc_coords = {c: np.asarray(ds_cases.coords[c].values) for c in ds_cases.coords}
    ds_cases = xr.Dataset(_dc_vars, _dc_coords)

    y         = XarraySpatioTemporalDataset(ds_cases, variables=["dengue_total"], T_max=1)
    x_spatial = XarraySpatioTemporalDataset(ds_cases, variables=["FAO_GAUL_code"], T_max=1)

    era5   = ERA5Daily(era5_path, T_max=63, min_date=start_date, max_date=end_date,
                       weekly_cache_dir=shared_cache_dir)
    viirs  = VIIRSData(viirs_data_path, min_date=start_date, max_date=end_date)
    static = StaticLayer(risk_raster_path, nodata=-3.3999999521443642e+38)

    soil_moisture = None
    if args.add_sm:
        soil_moisture = SoilMoistureData(args.sm_data_path,
                                         min_date=start_date, max_date=end_date,
                                         cache_dir=shared_cache_dir)

    full_dataset = DengueDataset(
        viirs, era5, static, x_spatial, y,
        bbox=latin_box(),
        skip_era5_bounds=True,
        cache_dir=shared_cache_dir,
        num_zones=num_zones,
        loss_fn=args.loss_fn,
        soil_moisture=soil_moisture,
    )

    train_size = int(args.train_split * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    dataset = val_ds if args.split == "val" else train_ds
    print(f"Using {args.split} split: {len(dataset)} samples")

    gpw_population  = None   # GPWv4Population object, or None
    gpw_pixel_arr   = None   # [H_model, W_model] population per model pixel (lazy init)
    if args.gpw_path:
        from models.model_utils import GPWv4Population, _resample_gpw_to_pixel_grid
        gpw_population = GPWv4Population(args.gpw_path)

    cfg = vars(args)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, cfg, num_zones, device)

    criterion = torch.nn.PoissonNLLLoss(log_input=True, reduction="mean")

    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))
    done = 0
    i = 0
    sample_records = []

    while done < args.n_samples and i < len(indices):
        item = dataset[indices[i].item()]
        i += 1
        if item is None:
            continue

        if args.add_sm:
            x_high, x_med, x_static, x_suit, x_cond, x_sm, y_val = item
        else:
            x_high, x_med, x_static, x_suit, x_cond, y_val = item
            x_sm = None

        x_high_b   = x_high.unsqueeze(0).to(device)
        x_med_b    = x_med.unsqueeze(0).to(device)
        x_static_b = x_static.unsqueeze(0).to(device)
        x_suit_b   = x_suit.unsqueeze(0).to(device)
        x_cond_b   = x_cond.unsqueeze(0).to(device)
        x_sm_b     = x_sm.unsqueeze(0).to(device) if x_sm is not None else None

        x_high_b   = torch.nan_to_num(x_high_b,   nan=0.0)
        x_med_b    = torch.nan_to_num(x_med_b,    nan=0.0)
        x_static_b = torch.nan_to_num(x_static_b, nan=0.0)
        x_suit_b   = torch.nan_to_num(x_suit_b,   nan=0.0)
        if x_sm_b is not None:
            x_sm_b = torch.nan_to_num(x_sm_b, nan=0.0)

        with torch.no_grad():
            pred = model(x_high_b, x_med_b, x_static_b, x_suit_b, x_cond_b, x_sm_b)

        pred_cpu = pred[0].cpu()
        zone_map = x_cond.squeeze().long()

        log_rates  = pred_cpu.squeeze()
        valid_mask = (zone_map >= 0) & (zone_map < num_zones)
        zm_flat    = zone_map[valid_mask].view(-1)
        pv_flat    = log_rates[valid_mask].view(-1)
        zone_max   = torch.full((num_zones,), float('-inf'))
        zone_max.scatter_reduce_(0, zm_flat, pv_flat, reduce='amax', include_self=True)
        exp_sum    = torch.zeros(num_zones)
        exp_sum.scatter_add_(0, zm_flat, torch.exp(pv_flat - zone_max[zm_flat]))
        zone_log_rates  = zone_max + torch.log(exp_sum + 1e-8)
        zone_pred_rates = torch.exp(zone_log_rates)

        if args.loss_fn == "poisson":
            zone_actual = y_val.float()
        else:
            y_2d = y_val.squeeze()
            zone_actual = torch.zeros(num_zones)
            zone_cnt    = torch.zeros(num_zones)
            valid_y     = valid_mask & torch.isfinite(y_2d)
            if valid_y.any():
                zm2 = zone_map[valid_y].view(-1)
                yv2 = y_2d[valid_y].float().view(-1)
                zone_actual.scatter_add_(0, zm2, yv2)
                zone_cnt.scatter_add_(0, zm2, torch.ones_like(yv2))
            zone_actual = torch.where(zone_cnt > 0, zone_actual / zone_cnt,
                                      torch.full_like(zone_actual, float('nan')))

        # Lazily build the GPW pixel grid once we know the model output resolution
        if gpw_population is not None and gpw_pixel_arr is None:
            H_m, W_m = log_rates.shape
            gpw_pixel_arr = _resample_gpw_to_pixel_grid(x_spatial, gpw_population, H_m, W_m)
            print(f"GPW pixel grid: shape={gpw_pixel_arr.shape}, "
                  f"total_pop={gpw_pixel_arr.sum():.3e}, max_cell={gpw_pixel_arr.max():.0f}")

        burden_map = None
        if gpw_pixel_arr is not None:
            P    = torch.from_numpy(gpw_pixel_arr)        # [H, W] population per pixel
            suit = torch.exp(log_rates)                   # [H, W] exp(ŷ_i), pixel suitability
            Ps   = P * suit                               # [H, W] population-weighted suitability

            # D_z = Σ_{j∈z} P_j · exp(ŷ_j)  (independent of Ŷ_z)
            D_z = torch.zeros(num_zones)
            D_z.scatter_add_(0, zm_flat, Ps[valid_mask].view(-1))

            # Broadcast Ŷ_z and D_z from zone → pixel
            zone_clamped = zone_map.clamp(min=0, max=num_zones - 1)
            Yhat_z_px = zone_pred_rates[zone_clamped]     # [H, W]
            D_z_px    = D_z[zone_clamped]                 # [H, W]

            # b_i = Ŷ_z · P_i · exp(ŷ_i) / D_z  (sums to Ŷ_z within each zone)
            b = Yhat_z_px * Ps / D_z_px.clamp(min=1e-8)
            burden_map = b.masked_fill(~valid_mask, float("nan"))

        zone_counts = torch.zeros(num_zones)
        zone_counts.scatter_add_(0, zm_flat, torch.ones_like(pv_flat))
        present = (zone_counts > 0) & torch.isfinite(zone_actual)
        if present.any():
            lam_v       = zone_pred_rates[present]
            actual      = zone_actual[present]
            pred_int    = lam_v.round()
            sample_mae  = (pred_int - actual).abs().mean().item()
            sample_bias = (pred_int - actual).mean().item()
            sample_loss = criterion(zone_log_rates[present], actual).item()
            sample_rmse = ((lam_v - actual) ** 2).mean().sqrt().item()
            sample_rmsle = ((torch.log1p(lam_v) - torch.log1p(actual)) ** 2).mean().sqrt().item()
            sample_log_ratio = (torch.log1p(lam_v) - torch.log1p(actual)).abs().mean().item()
        else:
            sample_mae = sample_bias = sample_loss = float("nan")
            sample_rmse = sample_rmsle = sample_log_ratio = float("nan")

        sample_records.append({
            "sample":      done,
            "poisson_nll": sample_loss,
            "mae":         sample_mae,
            "bias":        sample_bias,
            "rmse":        sample_rmse,
            "rmsle":       sample_rmsle,
            "log_ratio":   sample_log_ratio,
            "n_zones":     int(present.sum().item()),
        })
        print(f"  sample {done:03d} | loss={sample_loss:.4f} | mae={sample_mae:.2f} | "
              f"bias={sample_bias:+.2f} | rmse={sample_rmse:.2f} | "
              f"rmsle={sample_rmsle:.4f} | log_ratio={sample_log_ratio:.4f} | zones={present.sum().item()}")

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_sample(
            sample_idx=done,
            x_high=x_high.cpu(),
            x_cond=x_cond.cpu(),
            pred_log_rate=pred_cpu,
            zone_pred_rates=zone_pred_rates,
            zone_actual=zone_actual,
            num_zones=num_zones,
            output_dir=output_dir,
            timestamp=ts,
            burden_map=burden_map,
        )
        done += 1

    if sample_records:
        import csv, math
        valid_recs = [r for r in sample_records if not math.isnan(r["poisson_nll"])]
        n_valid = max(len(valid_recs), 1)
        avg = {
            "sample":      "AVERAGE",
            "poisson_nll": sum(r["poisson_nll"] for r in valid_recs) / n_valid,
            "mae":         sum(r["mae"]         for r in valid_recs) / n_valid,
            "bias":        sum(r["bias"]        for r in valid_recs) / n_valid,
            "rmse":        sum(r["rmse"]        for r in valid_recs) / n_valid,
            "rmsle":       sum(r["rmsle"]       for r in valid_recs) / n_valid,
            "log_ratio":   sum(r["log_ratio"]   for r in valid_recs) / n_valid,
            "n_zones":     sum(r["n_zones"]     for r in valid_recs) // n_valid,
        }
        rows = sample_records + [avg]
        fieldnames = ["sample", "poisson_nll", "mae", "bias", "rmse", "rmsle", "log_ratio", "n_zones"]
        csv_path = output_dir / "metrics_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
        print(f"\nMetrics summary saved to {csv_path}")
        print(f"\n{'sample':>8} {'poisson_nll':>12} {'mae':>8} {'bias':>8} {'rmse':>8} {'rmsle':>8} {'log_ratio':>10} {'n_zones':>8}")
        print("-" * 80)
        for r in rows:
            nll       = f"{r['poisson_nll']:.4f}" if isinstance(r['poisson_nll'], float) else r['poisson_nll']
            mae       = f"{r['mae']:.2f}"          if isinstance(r['mae'],         float) else r['mae']
            bias      = f"{r['bias']:+.2f}"        if isinstance(r['bias'],        float) else r['bias']
            rmse      = f"{r['rmse']:.2f}"         if isinstance(r['rmse'],        float) else r['rmse']
            rmsle     = f"{r['rmsle']:.4f}"        if isinstance(r['rmsle'],       float) else r['rmsle']
            log_ratio = f"{r['log_ratio']:.4f}"    if isinstance(r['log_ratio'],   float) else r['log_ratio']
            print(f"{str(r['sample']):>8} {nll:>12} {mae:>8} {bias:>8} {rmse:>8} {rmsle:>8} {log_ratio:>10} {str(r['n_zones']):>8}")

    print(f"\nDone — {done} figures saved to {output_dir}")


if __name__ == "__main__":
    main()
