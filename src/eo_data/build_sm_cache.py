#!/usr/bin/env python3
"""
Precompute SM 6-day composites to a zarr cache at native ~1 km resolution.

Runs once as a CPU PBS job; training then reads the cache via lazy zarr
slicing instead of per-sample rasterio reprojection.

Output zarr shape: [n_windows, 2, H_global, W_global]
  axis 0: 6-day windows aligned to START_DATE
  axis 1: channel 0 = SM value in m³/m³ (0 where gap)
           channel 1 = validity mask (1 where data, 0 where gap)
  axis 2/3: lat/lon on a 0.01° grid (~1 km) covering the training bbox

Usage:
  python src/eo_data/build_sm_cache.py \
      --sm_root /path/to/soil_moisture \
      --cache_dir /path/to/dataset_cache \
      [--workers N]
"""
import argparse
import datetime
import re
import traceback
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

import zarr
from numcodecs import Blosc

# ────────────────────────── constants ──────────────────────────
START_DATE  = datetime.date(2016, 1, 1)
END_DATE    = datetime.date(2022, 12, 31)
WINDOW_DAYS = 6

# Latin-America training bbox
LON_MIN, LON_MAX = -87.0, -33.0   # 54° → 5400 px at 0.01°
LAT_MIN, LAT_MAX = -36.0, +14.0   # 50° → 5000 px at 0.01°
RES      = 0.01
W_GLOBAL = round((LON_MAX - LON_MIN) / RES)   # 5400
H_GLOBAL = round((LAT_MAX - LAT_MIN) / RES)   # 5000

# SM nodata and physical scale (from WorldSAR product spec)
SM_NODATA = 10000.0   # values >= 10000 are fill/flag (32766, 32767, etc.)
SM_SCALE  = 0.001     # raw int16 → m³/m³  (valid range ~0.0–0.7 m³/m³)

# Module-level so workers inherit via fork (no pickle needed)
_DST_TRANSFORM = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W_GLOBAL, H_GLOBAL)
_DST_CRS       = 'EPSG:4326'


# ────────────────────────── index builder ──────────────────────
def build_index(sm_root: Path) -> dict:
    """Return {date: {'A': path, 'D': path}} for all SM files."""
    index: dict = {}
    year_re = re.compile(r'(\d{8})')
    for fpath in sorted(sm_root.rglob('SM_[AD]_*.tif')):
        m = year_re.search(fpath.stem)
        if not m:
            continue
        parts = fpath.stem.split('_')
        if len(parts) < 3:
            continue
        sensor = parts[1]
        dt = datetime.datetime.strptime(m.group(1), '%Y%m%d').date()
        if dt < START_DATE or dt > END_DATE:
            continue
        index.setdefault(dt, {})[sensor] = fpath
    return index


# ────────────────────────── per-window worker ──────────────────
def _compute_window(args):
    """Compute one 6-day composite. Returns (win_idx, array[2, H, W]) in m³/m³."""
    win_idx, window_dates, index = args
    stacked = []
    for dt in window_dates:
        entry = index.get(dt, {})
        for sensor in ('A', 'D'):
            fpath = entry.get(sensor)
            if fpath is None:
                continue
            try:
                with rasterio.open(fpath) as src:
                    dst_arr = np.full((1, H_GLOBAL, W_GLOBAL), np.nan, dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dst_arr,
                        dst_transform=_DST_TRANSFORM,
                        dst_crs=_DST_CRS,
                        resampling=Resampling.nearest,
                        dst_nodata=np.nan,
                    )
                    # Mask primary nodata (32767) and secondary fill flags (>= 10000)
                    nd = src.nodata if src.nodata is not None else 32767.0
                    dst_arr[dst_arr >= nd]        = np.nan
                    dst_arr[dst_arr >= SM_NODATA] = np.nan
                    dst_arr *= SM_SCALE            # scale to m³/m³ in-place
                    stacked.append(dst_arr[0])     # [H, W]
            except Exception:
                traceback.print_exc()

    if not stacked:
        return win_idx, np.zeros((2, H_GLOBAL, W_GLOBAL), dtype=np.float32)

    arr    = np.stack(stacked, axis=0)                            # [N, H, W]
    valid  = np.any(np.isfinite(arr), axis=0)                     # [H, W]
    sm_max = np.where(valid, np.nanmax(arr, axis=0), 0.0).astype(np.float32)
    mask   = valid.astype(np.float32)
    return win_idx, np.stack([sm_max, mask], axis=0)              # [2, H, W]


# ────────────────────────── main ──────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sm_root',   required=True,
                        help='Root directory of SM_[A|D]_YYYYMMDD.tif files')
    parser.add_argument('--cache_dir', required=True,
                        help='Output directory; zarr written as sm_6day_1km.zarr inside it')
    parser.add_argument('--workers',   type=int, default=24)
    args = parser.parse_args()

    sm_root    = Path(args.sm_root)
    cache_path = Path(args.cache_dir) / 'sm_6day_1km.zarr'

    print(f"SM root  : {sm_root}", flush=True)
    print(f"Cache    : {cache_path}", flush=True)
    print(f"Grid     : {W_GLOBAL}×{H_GLOBAL} @ {RES}° (~1 km) | "
          f"lon [{LON_MIN},{LON_MAX}] lat [{LAT_MIN},{LAT_MAX}]", flush=True)
    print(f"Workers  : {args.workers}", flush=True)

    print("Building SM file index…", flush=True)
    index = build_index(sm_root)
    print(f"Indexed {len(index)} days ({min(index)} – {max(index)})", flush=True)

    total_days = (END_DATE - START_DATE).days + 1
    n_windows  = (total_days + WINDOW_DAYS - 1) // WINDOW_DAYS
    print(f"Windows  : {n_windows}", flush=True)

    # ── Create zarr store ───────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(
        str(cache_path), mode='w',
        shape=(n_windows, 2, H_GLOBAL, W_GLOBAL),
        chunks=(1, 2, 50, 50),   # one window × both channels × 50×50 spatial tile
        dtype='float32',
        fill_value=0.0,
        zarr_format=2,
        compressor=Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE),
    )
    store.attrs['start_date']  = START_DATE.isoformat()
    store.attrs['window_days'] = WINDOW_DAYS
    store.attrs['lon_min']     = LON_MIN
    store.attrs['lat_min']     = LAT_MIN
    store.attrs['resolution']  = RES
    store.attrs['n_windows']   = n_windows
    store.attrs['h_global']    = H_GLOBAL
    store.attrs['w_global']    = W_GLOBAL
    store.attrs['sm_scale']    = SM_SCALE
    store.attrs['units']       = 'm3/m3'

    # ── Build tasks ─────────────────────────────────────────────
    tasks = []
    for win_idx in range(n_windows):
        win_start = START_DATE + datetime.timedelta(days=win_idx * WINDOW_DAYS)
        dates = [win_start + datetime.timedelta(days=d) for d in range(WINDOW_DAYS)]
        tasks.append((win_idx, dates, index))

    # ── Process in parallel ─────────────────────────────────────
    t0 = datetime.datetime.now()
    n_done = 0
    with Pool(processes=args.workers) as pool:
        for win_idx, composite in pool.imap_unordered(
            _compute_window, tasks, chunksize=2
        ):
            store[win_idx] = composite
            n_done += 1
            if n_done % 50 == 0 or n_done == n_windows:
                elapsed = (datetime.datetime.now() - t0).total_seconds()
                rate = n_done / elapsed
                eta = (n_windows - n_done) / max(rate, 1e-6)
                print(
                    f"  {n_done}/{n_windows} windows | "
                    f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s",
                    flush=True,
                )

    elapsed = (datetime.datetime.now() - t0).total_seconds()
    size_mb = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / 1e6
    print(f"\nDone in {elapsed:.0f}s. Cache size: {size_mb:.0f} MB at {cache_path}",
          flush=True)


if __name__ == '__main__':
    main()
