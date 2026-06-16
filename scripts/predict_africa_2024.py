"""
predict_africa_2024.py

Run inference over all of Africa for every month of 2024 using up to three
trained checkpoints (baseline, GPW, SM). Produces a CSV with predicted dengue
cases per country per month.

Zone conditioning note
----------------------
The trained models' zone-embedding tables are sized for Latin America Admin2
zones. For zero-shot Africa transfer we pass zone_id=0 everywhere (disabling
zone-specific conditioning while keeping all other branches intact). The GAUL
Africa zone map is used only for *output aggregation* (log-rates → zone and
country totals).

SM model note
-------------
Soil-moisture data only extends to 2022. For 2024 Africa predictions the SM
branch receives zeros (ascending and descending SM channels both set to 0),
which effectively switches it off at inference.

Usage example
-------------
PYTHONPATH=/path/to/dengue-africa/src \
uv run python scripts/predict_africa_2024.py \
    --gpw_ckpt   checkpoints/run_20260612_105059/best_model.pth \
    --sm_ckpt    checkpoints/run_sm_only/checkpoints/checkpoint_epoch_77/model_state_epoch_77.pth \
    --output_dir outputs/africa_2024
"""

import argparse
import os
import sys
import math
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize

warnings.filterwarnings("ignore", category=UserWarning)

# ── Project path ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from definitions import DATA_PATH
from models.transformer import DenguePredictor
from utils.seasonal_encoding import append_seasonal_encoding

# ── Constants ────────────────────────────────────────────────────────────────
# Africa bounding box (from ERA5/VIIRS data extent)
AFRICA_BBOX = dict(lon_min=-25.5, lon_max=63.5, lat_min=-20.5, lat_max=27.5)

# ERA5 resolution and patch sizes (must match training)
ERA5_RES  = 0.1   # degrees per pixel
ERA5_PATCH = 43   # ERA5 pixels per tile

# VIIRS tile size fed to the model (must match Swin encoder input)
VIIRS_PATCH = 1024

# Static (risk raster) patch size
STATIC_PATCH = 102

# Output patch size (spatial conditioning and model output)
OUT_PATCH = 86

# Number of weekly time steps (T_max=63 days / 7)
N_WEEKS = 9

# ERA5 channels after weekly aggregation (3 sums + 5*3 stats = 18) + 2 seasonal (sin/cos)
MED_IN_CH = 20

# Soil moisture channels (ascending + descending)
SM_IN_CH = 2

# Embedding table size the checkpoint was trained with (Latin America zones)
LATAM_NUM_ZONES = 4057

ERA5_SUM_VARS  = ["tp", "pev", "e"]
ERA5_TEMP_VARS = ["skt", "stl1", "stl2", "stl3", "stl4"]


# ── ERA5 helpers ─────────────────────────────────────────────────────────────

def load_era5_weekly(era5_path: Path) -> xr.Dataset:
    """Load Africa ERA5, normalise coords, aggregate to weekly."""
    ds = xr.open_dataset(era5_path)

    rename = {}
    for k in ("valid_time", "date"):
        if k in ds.coords:
            rename[k] = "time"
    for k in ("latitude",):
        if k in ds.coords:
            rename[k] = "y"
    for k in ("longitude",):
        if k in ds.coords:
            rename[k] = "x"
    if rename:
        ds = ds.rename(rename)

    present_sum  = [v for v in ERA5_SUM_VARS  if v in ds]
    present_temp = [v for v in ERA5_TEMP_VARS if v in ds]

    wsum  = ds[present_sum].resample(time="1W-MON", label="left").sum()
    wmean = ds[present_temp].resample(time="1W-MON", label="left").mean()
    wmin  = ds[present_temp].resample(time="1W-MON", label="left").min()
    wmax  = ds[present_temp].resample(time="1W-MON", label="left").max()

    wmean = wmean.rename({v: f"{v}_mean" for v in present_temp})
    wmin  = wmin.rename({v:  f"{v}_min"  for v in present_temp})
    wmax  = wmax.rename({v:  f"{v}_max"  for v in present_temp})

    ds_weekly = xr.merge([wsum, wmean, wmin, wmax])
    ds_weekly.load()
    return ds_weekly


def era5_channels_for_month(ds_weekly: xr.Dataset, month: int) -> np.ndarray:
    """
    Extract the 9 weekly ERA5 composites preceding the start of `month`.
    Returns float32 array of shape [N_WEEKS, MED_IN_CH, H_era5, W_era5].
    MED_IN_CH = 18 ERA5 channels + 2 seasonal encoding channels (sin/cos doy).
    """
    month_start = pd.Timestamp(f"2024-{month:02d}-01")
    times = pd.DatetimeIndex(ds_weekly.time.values)

    # Select the N_WEEKS weeks that end at or before month_start
    before = times[times < month_start]
    if len(before) < N_WEEKS:
        sel_times = times[:N_WEEKS]
    else:
        sel_times = before[-N_WEEKS:]

    vars_order = (
        [v for v in ERA5_SUM_VARS  if v in ds_weekly]
        + [f"{v}_mean" for v in ERA5_TEMP_VARS if f"{v}_mean" in ds_weekly]
        + [f"{v}_min"  for v in ERA5_TEMP_VARS if f"{v}_min"  in ds_weekly]
        + [f"{v}_max"  for v in ERA5_TEMP_VARS if f"{v}_max"  in ds_weekly]
    )

    slices = []
    for t in sel_times:
        arr = np.stack(
            [ds_weekly[v].sel(time=t).values for v in vars_order], axis=0
        ).astype(np.float32)  # [C, H, W]
        slices.append(arr)

    era5 = np.stack(slices, axis=0)  # [T, 18, H, W]

    # Pad temporally if fewer than N_WEEKS available
    if era5.shape[0] < N_WEEKS:
        pad = np.zeros((N_WEEKS - era5.shape[0], *era5.shape[1:]), dtype=np.float32)
        era5 = np.concatenate([pad, era5], axis=0)
        # Pad timestamps with the earliest available time repeated
        sel_times = list(sel_times)
        sel_times = [sel_times[0]] * (N_WEEKS - len(sel_times)) + sel_times

    era5 = append_seasonal_encoding(era5, sel_times)   # [N_WEEKS, 20, H, W]

    return era5  # [N_WEEKS, MED_IN_CH, H_era5, W_era5]


# ── Zone map ─────────────────────────────────────────────────────────────────

def build_africa_zone_map(gaul_path: Path, out_res: float = 0.05):
    """
    Rasterise GAUL Africa Admin2 zones to `out_res` degree grid.

    Returns
    -------
    zone_arr  : int32 ndarray [H, W],  gaul2_code (or -1 for ocean/nodata)
    gaul2_to_country : dict  gaul2_code -> country_name
    transform : rasterio.Affine
    bbox      : (lon_min, lat_min, lon_max, lat_max)
    """
    print("Loading GAUL shapefile …", flush=True)
    gdf = gpd.read_file(gaul_path)
    gdf_africa = gdf[gdf["continent"] == "Africa"].copy()

    lon_min = AFRICA_BBOX["lon_min"]
    lon_max = AFRICA_BBOX["lon_max"]
    lat_min = AFRICA_BBOX["lat_min"]
    lat_max = AFRICA_BBOX["lat_max"]

    W = int(round((lon_max - lon_min) / out_res))
    H = int(round((lat_max - lat_min) / out_res))
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)

    shapes = [
        (geom, int(code))
        for geom, code in zip(gdf_africa.geometry, gdf_africa["gaul2_code"])
        if geom is not None and np.isfinite(code)
    ]

    zone_arr = rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=-1,
        dtype="int32",
    )
    # Flip to lat-ascending (rasterio gives top-down)
    zone_arr = zone_arr[::-1].copy()

    gaul2_to_country = dict(
        zip(gdf_africa["gaul2_code"].astype(int), gdf_africa["gaul0_name"])
    )

    print(f"  Zone map: {H}×{W}, {len(gaul2_to_country)} Admin2 zones", flush=True)
    return zone_arr, gaul2_to_country, transform, (lon_min, lat_min, lon_max, lat_max)


# ── Static risk raster ───────────────────────────────────────────────────────

def load_static_raster(raster_path: Path):
    """Load the dengue suitability TIF into a numpy array with its transform."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(np.float32)  # [H, W]
        transform = src.transform
        nodata = src.nodata or -3.4e38
    arr[arr < -1e30] = np.nan
    return arr, transform


# ── VIIRS helpers ─────────────────────────────────────────────────────────────

def load_viirs_monthly(viirs_dir: Path, month: int) -> np.ndarray:
    """
    Load VNP46A3_africa_YYYYMM.tif  →  float32 [H, W] (lat-ascending).
    Returns None if the file is missing.
    """
    fname = viirs_dir / f"VNP46A3_africa_2024{month:02d}.tif"
    if not fname.exists():
        return None
    with rasterio.open(fname) as src:
        arr = src.read(1).astype(np.float32)  # [H, W]
    # rasterio returns (north-down); flip to lat-ascending
    arr = arr[::-1].copy()
    arr[arr < 0] = np.nan
    return arr  # [H, W]


# ── Tiling helpers ───────────────────────────────────────────────────────────

def _bbox_to_slice(bbox_vals, coord_arr):
    """Return (start, end) integer indices that cover bbox_vals in coord_arr."""
    lo, hi = min(bbox_vals), max(bbox_vals)
    start = max(0, int(np.searchsorted(coord_arr, lo)) - 1)
    end   = min(len(coord_arr), int(np.searchsorted(coord_arr, hi, side='right')) + 1)
    return start, end


def spatial_crop(arr_2d, row0, row1, col0, col1, target_h=None, target_w=None):
    """
    Crop a [H, W] numpy array and optionally resize to (target_h, target_w).
    Handles out-of-bounds by zero-padding.
    """
    H, W = arr_2d.shape
    r0, r1 = max(0, row0), min(H, row1)
    c0, c1 = max(0, col0), min(W, col1)
    patch = arr_2d[r0:r1, c0:c1]

    # Pad if crop went out of bounds
    pad_top    = max(0, -row0)
    pad_bottom = max(0, row1 - H)
    pad_left   = max(0, -col0)
    pad_right  = max(0, col1 - W)
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        patch = np.pad(patch,
                       ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode="constant", constant_values=np.nan)

    if target_h is not None and target_w is not None and patch.shape != (target_h, target_w):
        t = torch.from_numpy(patch)[None, None].float()
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        patch = t[0, 0].numpy()

    return patch.astype(np.float32)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, add_sm: bool, device: torch.device,
               num_zones: int = LATAM_NUM_ZONES) -> DenguePredictor:
    model = DenguePredictor(
        med_in_ch=MED_IN_CH,
        num_zones=num_zones,
        swin_model="swin_tiny_patch4_window7_224.ms_in1k",
        add_sm=add_sm,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    # Strip DDP 'module.' prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}", flush=True)
    return model


# ── Core inference ───────────────────────────────────────────────────────────

@torch.inference_mode()
def run_tile(model, x_high_np, x_era5_np, x_static_np,
             device, add_sm=False) -> np.ndarray:
    """
    Run one forward pass for a single tile.

    Parameters
    ----------
    x_high_np   : [H_v, W_v]       VIIRS (single channel)
    x_era5_np   : [T, C, H_e, W_e] ERA5 weekly
    x_static_np : [H_s, W_s]       risk raster

    Returns
    -------
    log_rate : [OUT_PATCH, OUT_PATCH] float32
    """
    def _nan_to_num(arr):
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- x_high [1, 3, VIIRS_PATCH, VIIRS_PATCH] ----
    v = _nan_to_num(x_high_np)
    v_t = torch.from_numpy(v)[None, None]  # [1, 1, H, W]
    v_t = F.interpolate(v_t, size=(VIIRS_PATCH, VIIRS_PATCH),
                        mode="bilinear", align_corners=False)
    v_t = v_t.repeat(1, 3, 1, 1).to(device)  # [1, 3, 1024, 1024]

    # ---- x_med [1, T, C, H, W] ----
    e = _nan_to_num(x_era5_np)
    e_t = torch.from_numpy(e)[None].to(device)  # [1, T, C, H, W]

    # ---- x_static [1, 1, STATIC_PATCH, STATIC_PATCH] ----
    s = _nan_to_num(x_static_np)
    s_t = torch.from_numpy(s)[None, None]  # [1, 1, H, W]
    s_t = F.interpolate(s_t, size=(STATIC_PATCH, STATIC_PATCH),
                        mode="bilinear", align_corners=False).to(device)

    # ---- x_cond (zone ids) — all zeros for Africa transfer ----
    x_cond = torch.zeros(1, 1, OUT_PATCH, OUT_PATCH, device=device).long()

    # ---- x_sm (zeros if SM model) ----
    x_sm = None
    if add_sm:
        x_sm = torch.zeros(1, N_WEEKS, SM_IN_CH, ERA5_PATCH, ERA5_PATCH,
                           device=device)

    pred = model(v_t, e_t, s_t, x_cond, x_sm)  # [1, 1, OUT_PATCH, OUT_PATCH]
    return pred[0, 0].cpu().numpy().astype(np.float32)


# ── Main inference loop ──────────────────────────────────────────────────────

def predict_africa(
    model,
    add_sm: bool,
    era5_weekly: xr.Dataset,
    viirs_dir: Path,
    static_arr: np.ndarray,
    static_transform,
    zone_arr: np.ndarray,
    gaul2_to_country: dict,
    device: torch.device,
    out_res: float = 0.05,
):
    """
    Loop over all months and tiles; return a DataFrame with columns
    [month, country, predicted_cases].
    """
    lon_min = AFRICA_BBOX["lon_min"]
    lon_max = AFRICA_BBOX["lon_max"]
    lat_min = AFRICA_BBOX["lat_min"]
    lat_max = AFRICA_BBOX["lat_max"]

    # ERA5 coordinate arrays (lat-ascending)
    era5_lons = era5_weekly.x.values
    era5_lats = era5_weekly.y.values
    if era5_lats[0] > era5_lats[-1]:
        era5_lats = era5_lats[::-1]

    # VIIRS pixel size
    viirs_lons = np.linspace(lon_min, lon_max,
                             int(round((lon_max - lon_min) / 0.004167)))
    viirs_lats = np.linspace(lat_min, lat_max,
                             int(round((lat_max - lat_min) / 0.004167)))

    # Static raster coordinate arrays (lat-ascending)
    s_transform = static_transform
    static_arr_h, static_arr_w = static_arr.shape
    s_lons = np.array([s_transform.c + (j + 0.5) * s_transform.a
                       for j in range(static_arr_w)])
    s_lats = np.array([s_transform.f + (i + 0.5) * s_transform.e
                       for i in range(static_arr_h)])
    if s_lats[0] > s_lats[-1]:
        s_lats = s_lats[::-1]
        static_arr = static_arr[::-1].copy()

    # Zone map coordinate arrays
    zone_h, zone_w = zone_arr.shape
    z_lons = np.linspace(lon_min, lon_max, zone_w)
    z_lats = np.linspace(lat_min, lat_max, zone_h)

    # Tile grid (ERA5 patches)
    tile_deg = ERA5_PATCH * ERA5_RES  # degrees per tile
    tile_lon_starts = np.arange(lon_min, lon_max, tile_deg)
    tile_lat_starts = np.arange(lat_min, lat_max, tile_deg)

    # Output accumulator: zone_code -> monthly log-sum-exp (for Poisson aggregation)
    records = []

    for month in range(1, 13):
        print(f"\n── Month 2024-{month:02d} ──────────────────────────", flush=True)

        era5_tiles = era5_channels_for_month(era5_weekly, month)  # [T, C, H, W]
        viirs_month = load_viirs_monthly(viirs_dir, month)

        # Per-tile accumulator: zone_code -> list of pixel log-rates
        zone_log_rates: dict[int, list] = {}

        n_tiles = len(tile_lon_starts) * len(tile_lat_starts)
        done = 0

        for lat_start in tile_lat_starts:
            lat_end = lat_start + tile_deg

            for lon_start in tile_lon_starts:
                lon_end = lon_start + tile_deg

                # ---- ERA5 tile ----
                e_r0, e_r1 = _bbox_to_slice([lat_start, lat_end], era5_lats)
                e_c0, e_c1 = _bbox_to_slice([lon_start, lon_end], era5_lons)
                # clip to ERA5_PATCH exactly
                e_r1 = e_r0 + ERA5_PATCH
                e_c1 = e_c0 + ERA5_PATCH
                era5_tile = era5_tiles[:, :, e_r0:e_r1, e_c0:e_c1]  # [T,C,43,43]
                if era5_tile.shape[-2:] != (ERA5_PATCH, ERA5_PATCH):
                    # Pad if near boundary
                    ph = ERA5_PATCH - era5_tile.shape[-2]
                    pw = ERA5_PATCH - era5_tile.shape[-1]
                    era5_tile = np.pad(era5_tile,
                                      ((0,0),(0,0),(0,max(ph,0)),(0,max(pw,0))),
                                      constant_values=0.0)

                # ---- VIIRS tile ----
                if viirs_month is not None:
                    v_r0 = int((lat_start - lat_min) / (lat_max - lat_min) * len(viirs_lats))
                    v_r1 = int((lat_end   - lat_min) / (lat_max - lat_min) * len(viirs_lats))
                    v_c0 = int((lon_start - lon_min) / (lon_max - lon_min) * len(viirs_lons))
                    v_c1 = int((lon_end   - lon_min) / (lon_max - lon_min) * len(viirs_lons))
                    viirs_tile = viirs_month[v_r0:v_r1, v_c0:v_c1]
                else:
                    viirs_tile = np.zeros((VIIRS_PATCH, VIIRS_PATCH), dtype=np.float32)

                # ---- Static tile ----
                s_r0 = max(0, int((lat_start - s_lats[0]) / (s_lats[-1] - s_lats[0]) * len(s_lats)))
                s_r1 = s_r0 + int(round(tile_deg / abs(s_transform.e)))
                s_c0 = max(0, int((lon_start - s_lons[0]) / (s_lons[-1] - s_lons[0]) * len(s_lons)))
                s_c1 = s_c0 + int(round(tile_deg / abs(s_transform.a)))
                static_tile = spatial_crop(static_arr, s_r0, s_r1, s_c0, s_c1,
                                           STATIC_PATCH, STATIC_PATCH)

                # ---- Zone tile (for aggregation only) ----
                z_r0 = int((lat_start - z_lats[0]) / (z_lats[-1] - z_lats[0]) * zone_h)
                z_r1 = z_r0 + OUT_PATCH
                z_c0 = int((lon_start - z_lons[0]) / (z_lons[-1] - z_lons[0]) * zone_w)
                z_c1 = z_c0 + OUT_PATCH
                zone_tile = zone_arr[
                    max(0, z_r0): min(zone_h, z_r1),
                    max(0, z_c0): min(zone_w, z_c1)
                ]
                # Pad if near boundary
                if zone_tile.shape != (OUT_PATCH, OUT_PATCH):
                    ph = OUT_PATCH - zone_tile.shape[0]
                    pw = OUT_PATCH - zone_tile.shape[1]
                    zone_tile = np.pad(zone_tile,
                                      ((0, max(ph,0)), (0, max(pw,0))),
                                      constant_values=-1)

                # ---- Forward pass ----
                try:
                    log_rate = run_tile(model, viirs_tile, era5_tile, static_tile,
                                        device, add_sm=add_sm)  # [86, 86]
                except Exception as exc:
                    print(f"  Tile ({lat_start:.1f},{lon_start:.1f}) failed: {exc}",
                          flush=True)
                    done += 1
                    continue

                # ---- Accumulate per-zone log-rates ----
                valid = zone_tile >= 0
                if not valid.any():
                    done += 1
                    continue

                for zone_code, lr in zip(zone_tile[valid].ravel(),
                                         log_rate[valid].ravel()):
                    zone_code = int(zone_code)
                    if zone_code not in zone_log_rates:
                        zone_log_rates[zone_code] = []
                    zone_log_rates[zone_code].append(float(lr))

                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{n_tiles} tiles done", flush=True)

        # ── Aggregate zones → countries (log-sum-exp for Poisson) ──────────
        country_cases: dict[str, float] = {}
        for zone_code, lrs in zone_log_rates.items():
            country = gaul2_to_country.get(zone_code, "Unknown")
            lrs_arr = np.array(lrs, dtype=np.float64)
            # log-sum-exp for numerical stability
            max_lr = lrs_arr.max()
            zone_rate = float(np.exp(max_lr) * np.sum(np.exp(lrs_arr - max_lr)))
            country_cases[country] = country_cases.get(country, 0.0) + zone_rate

        for country, cases in sorted(country_cases.items()):
            records.append({
                "month": f"2024-{month:02d}",
                "country": country,
                "predicted_cases": round(cases, 2),
            })

        print(f"  → {len(country_cases)} countries with predictions", flush=True)

    return pd.DataFrame(records)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_ckpt", default=None,
                   help="Checkpoint for baseline model (no GPW, no SM)")
    p.add_argument("--gpw_ckpt", default=None,
                   help="Checkpoint for GPW model")
    p.add_argument("--sm_ckpt", default=None,
                   help="Checkpoint for SM model (trained with --add_sm)")
    p.add_argument("--output_dir", default="outputs/africa_2024")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data paths ────────────────────────────────────────────────────────────
    era5_path   = DATA_PATH / "ERA5" / "era5land_africa_2024.nc"
    viirs_dir   = DATA_PATH / "nightlights" / "VNP46A3_africa"
    static_path = DATA_PATH / "riskmaps" / "DEN_riskmap_wmean_masked.tif"
    gaul_path   = DATA_PATH / "shapefiles" / "GAUL_2024.zip"

    # ── Load shared data (once) ───────────────────────────────────────────────
    print("Loading ERA5 and computing weekly aggregations …", flush=True)
    era5_weekly = load_era5_weekly(era5_path)

    print("Loading static risk raster …", flush=True)
    static_arr, static_transform = load_static_raster(static_path)

    print("Building Africa zone map …", flush=True)
    zone_arr, gaul2_to_country, _, _ = build_africa_zone_map(gaul_path)

    # ── Run each model ────────────────────────────────────────────────────────
    models_to_run = []
    if args.baseline_ckpt:
        models_to_run.append(("baseline", args.baseline_ckpt, False))
    if args.gpw_ckpt:
        models_to_run.append(("gpw", args.gpw_ckpt, False))
    if args.sm_ckpt:
        models_to_run.append(("sm", args.sm_ckpt, True))

    if not models_to_run:
        print("No checkpoints provided. Use --baseline_ckpt / --gpw_ckpt / --sm_ckpt.")
        return

    all_dfs = []
    for model_name, ckpt_path, add_sm in models_to_run:
        print(f"\n{'='*60}\nRunning model: {model_name}\n{'='*60}", flush=True)
        model = load_model(ckpt_path, add_sm=add_sm, device=device)

        df = predict_africa(
            model=model,
            add_sm=add_sm,
            era5_weekly=era5_weekly,
            viirs_dir=viirs_dir,
            static_arr=static_arr,
            static_transform=static_transform,
            zone_arr=zone_arr,
            gaul2_to_country=gaul2_to_country,
            device=device,
        )
        df["model"] = model_name
        all_dfs.append(df)

        # Save per-model CSV
        out_path = output_dir / f"predictions_{model_name}_2024.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}", flush=True)

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Combined CSV ──────────────────────────────────────────────────────────
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        pivot = combined.pivot_table(
            index=["month", "country"],
            columns="model",
            values="predicted_cases",
        ).reset_index()
        pivot.to_csv(output_dir / "predictions_all_models_2024.csv", index=False)
        print(f"\nCombined CSV saved to {output_dir}/predictions_all_models_2024.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
