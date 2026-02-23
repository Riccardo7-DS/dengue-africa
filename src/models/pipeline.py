import logging
import sys
import argparse

from models.model_utils import export_batches

def run_pipeline():
    from models import DengueConvLSTM
    import torch
    from utils import admin2_aggregate, aggregate_to_admin, load_admin_data
    from torch.nn import MSELoss
    from definitions import DATA_PATH

    tabular_data = load_admin_data(DATA_PATH / "Spatial_extract_V1_3.csv",
        temporal_resolution="Week", 
        spatial_resolution="Admin2", 
        filter_year=2000
    )

    data_path = DATA_PATH / "Spatial_extract_V1_3.csv"
    weekly_admin2_cases = load_admin_data(data_path, 
        temporal_resolution="Week", 
        spatial_resolution="Admin2", 
        filter_year=2000
    )

    criterion = MSELoss()
    model = DengueConvLSTM(raster_channels=3, tab_features=20, hidden_dim=64, weeks_out=4)
    raster_seq = torch.randn(8, 10, 3, 32, 32)  # (B, T, C, H, W)

    
    # Training loop
    weekly_pred = model(raster_seq, tabular_data)  # (B, weeks, H, W)
    agg_pred = aggregate_to_admin(weekly_pred, admin2_mask)
    loss = criterion(agg_pred, weekly_admin2_cases)  # MSE / Poisson / etc.



def tabular_dataset_pipeline(sample=False):
    from definitions import DATA_PATH
    from pathlib import Path
    import xarray as xr
    import pandas as pd
    import numpy as np
    from eo_data import read_predict_arbodata
    from tqdm import tqdm
    from utils import prepare
    from dask.diagnostics import ProgressBar
    from eo_data import load_nightlights_data
    from utils import init_logging

    logger = init_logging(log_file="dengue_pipeline.log", verbose=False)

    n_sample_points = 50
    n_sample_times = 10

    # ------------------------------
    # 1. Load dengue risk data
    # ------------------------------
    risk_data_path = Path(DATA_PATH) / "riskmaps_public main data/intermediate_datasets/Arbo_model_fit_data.rds"
    df_arbo = read_predict_arbodata(risk_data_path, model="GAM")

    lon_min, lon_max = -83.1, -55.1
    lat_min, lat_max = -28.8, 13.4

    df_filtered = df_arbo[
        (df_arbo["Longitude"].between(lon_min, lon_max)) &
        (df_arbo["Latitude"].between(lat_min, lat_max))
    ].reset_index(drop=True)

    if sample:
        df_filtered = df_filtered.sample(n=min(n_sample_points, len(df_filtered)), random_state=42).reset_index(drop=True)

    df_risk = df_filtered[["pred_dengue_risk", "Latitude", "Longitude"]].copy()
    df_risk["location_id"] = range(len(df_risk))

    tif_dir = DATA_PATH / "modis/-83.1_-55.1_-28.8_13.4"
    da = load_nightlights_data(tif_dir, n_sample_times= n_sample_times if sample else 0)

    lat_da = xr.DataArray(df_risk["Latitude"].values, dims="points")
    lon_da = xr.DataArray(df_risk["Longitude"].values, dims="points")

    with ProgressBar():
        da = da.rename({"x": "lon", "y": "lat"})
        ds_values = da.interp(lat=lat_da, lon=lon_da, method="nearest").compute()

    times = ds_values.time.values
    # ------------------------------
    # 4. Convert nightlight to long DataFrame
    # ------------------------------
    df_long = pd.DataFrame({
        "location_id": np.repeat(df_risk["location_id"].values, len(ds_values.time)),
        "Latitude": np.repeat(df_risk["Latitude"].values, len(ds_values.time)),
        "Longitude": np.repeat(df_risk["Longitude"].values, len(ds_values.time)),
        "date": np.tile(pd.to_datetime(times), len(df_risk)),
        "nightlight": ds_values.values.T.flatten()
    })

    df_long = df_long.merge(df_risk[["location_id", "pred_dengue_risk"]], on="location_id")
    df_long["year"] = df_long["date"].dt.year
    df_long["month"] = df_long["date"].dt.month
    df_long["week"] = df_long["date"].dt.isocalendar().week

    # ------------------------------
    # 5. Load ERA5 data
    # ------------------------------
    era5_dir = DATA_PATH / "ERA5"
    years_needed = range(df_long["year"].min(), df_long["year"].max() + 1)
    era5_dfs = []

    for y in tqdm(years_needed, desc="Processing ERA5 years"):
        era5_file = era5_dir / f"era5land_latin_america_{y}.nc"
        if not era5_file.exists():
            logger.warning(f"ERA5 file not found for year {y}")
            continue

        ds_era = xr.open_dataset(era5_file, chunks={"lat":100,"lon":100,"time":30})
        if "valid_time" in ds_era.dims:
            ds_era = ds_era.rename({"valid_time": "time"})
        ds_era = prepare(ds_era)
        ds_era = ds_era.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

        if sample:
            ds_era = ds_era.isel(time=slice(0, n_sample_times))

        with ProgressBar():
            ds_points = ds_era.interp(lat=lat_da, lon=lon_da, method="nearest").compute()

        if ds_points.time.size == 0:
            continue

        df_vars = pd.DataFrame({
            "location_id": np.repeat(df_risk["location_id"].values, len(ds_points.time)),
            "date": np.tile(pd.to_datetime(ds_points.time.values), len(df_risk)),
        })

        for var in ds_points.data_vars:
            df_vars[var] = ds_points[var].values.T.flatten()

        era5_dfs.append(df_vars)

    if era5_dfs:
        df_era5 = pd.concat(era5_dfs, axis=0, ignore_index=True)
        df_era5 = df_era5.groupby(["location_id", "date"]).first().reset_index()
        df_long = df_long.merge(df_era5, on=["location_id", "date"], how="left")

    # ------------------------------
    # 6. Export
    # ------------------------------
    output_file = DATA_PATH / ("ml_ready_dataframe_sample.parquet" if sample else "ml_ready_dataframe.parquet")
    df_long.to_parquet(output_file, index=False)
    logger.info(f"ML-ready dataframe saved to: {output_file}")
    logger.info(df_long.head())





def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device):
    """Load checkpoint from specified directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., checkpoints/checkpoint_epoch_5)
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Starting epoch number
    """
    import torch
    import os
    from pathlib import Path
    import logging
    logger = logging.getLogger(__name__)
    
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find metadata file
    metadata_files = list(checkpoint_dir.glob("metadata_epoch_*.pth"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata file found in {checkpoint_dir}")
    
    metadata_path = metadata_files[0]
    metadata = torch.load(metadata_path, map_location=device)
    epoch = metadata['epoch']
    
    logger.info(f"Loading checkpoint from epoch {epoch}...")
    
    # Load each component
    for key, path in metadata['components'].items():
        if not os.path.exists(path):
            logger.warning(f"Component file not found: {path}")
            continue
            
        state_dict = torch.load(path, map_location=device)
        
        if 'model_state' in key:
            model.load_state_dict(state_dict)
            logger.info("Loaded model state")
        elif 'optimizer_state' in key:
            optimizer.load_state_dict(state_dict)
            logger.info("Loaded optimizer state")
        elif 'scheduler_state' in key:
            scheduler.load_state_dict(state_dict)
            logger.info("Loaded scheduler state")
    
    return epoch


def main(config: dict | None = None, checkpoint_epoch: int | None = None, sample: bool = False, fillna: bool = False):
    import logging
    import re
    from pathlib import Path
    from types import SimpleNamespace
    from datetime import datetime

    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, random_split
    import xarray as xr

    from models import (
        VIIRSData,
        ERA5Daily,
        StaticLayer,
        XarraySpatioTemporalDataset,
        DengueDataset,
        DenguePredictor,
        collate_skip_none,
    )
    from models.model_utils import (
        masked_custom_loss, 
        EarlyStopping, 
        nan_checks_replace, 
        debug_nan,
        standardize_tensor,
        export_batches
    )
    from definitions import DATA_PATH
    from utils import latin_box, init_logging
    
    # Create a simple namespace to mimic args for backward compatibility
    from types import SimpleNamespace
    args = SimpleNamespace(sample=sample, fillna=fillna)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    default_config = {
        "batch_size": 32,
        "num_epochs": 20,
        "learning_rate": 1e-4,
        "train_split": 0.8,
        "patience": 5,
        "num_workers": 4,
        "amp": True,
        "grad_clip": 1.0,
        "save_dir": "./checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if config is not None:
        default_config.update(config)
    cfg = default_config

    device = torch.device(cfg["device"])
    
    # Handle checkpoint loading or create new run directory
    base_dir = Path(cfg["save_dir"])
    start_epoch = 0
    
    if checkpoint_epoch is not None:
        # Find the most recent run directory
        run_dirs = sorted([d for d in base_dir.glob("run_*") if d.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"No previous runs found in {base_dir}")
        
        run_dir = run_dirs[-1]  # Use most recent run
        
        logger_temp = logging.getLogger(__name__)
        logger_temp.info(f"Resuming from run: {run_dir}")
        timestamp = re.search(r"\d{8}_\d{6}", str(run_dir / "logs" / "training.log")).group()
    else:
        # Create unique run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        

    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    plot_dir = run_dir / "plots"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    logger = init_logging(log_file=log_dir / "training.log", verbose=False)
    logger.info(f"Starting training run at {timestamp}")
    logger.info(f"Run directory: {run_dir}")

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------
    viirs_data_path = DATA_PATH / "modis" / "VIIRS_nightlight"
    era5_path = DATA_PATH / "ERA5" / "Latin_america"
    risk_raster_path = DATA_PATH / "riskmaps_public main data" / "DEN_riskmap_wmean_masked.tif"
    admin_path = DATA_PATH / "weekly_admin2_cases.nc"

    ds_cases = xr.open_dataset(admin_path)
    if args.sample:
        sample_dates = 20
        ds_cases = ds_cases.isel(time=slice(0, sample_dates))
        first_date = ds_cases.isel(time=0)["time"].values
        last_date = ds_cases.isel(time=-1)["time"].values
    else:
        first_date, last_date = None, None

    y = XarraySpatioTemporalDataset(ds_cases, T_max=1)
    era5 = ERA5Daily(era5_path, T_max=63, min_date=first_date, max_date=last_date)
    viirs = VIIRSData(viirs_data_path, min_date=first_date, max_date=last_date)
    static = StaticLayer(risk_raster_path, nodata=-3.3999999521443642e+38)

    full_dataset = DengueDataset(viirs, era5, static, y, bbox=latin_box(), 
                                 skip_era5_bounds=True, cache_dir=str(run_dir / "cache"))


    logger.info(f"Dataset bbox clipped to ERA5 valid extent: {full_dataset.bbox}")
    logger.info(f"Full dataset size: {len(full_dataset)}")


    # Train/val split
    train_size = int(cfg["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_skip_none,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_skip_none,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------------
    model = DenguePredictor(
        pretrained_autoencoder="swin_tiny_patch4_window7_224.ms_in1k",
        high_in_ch=3,
        med_in_ch=18,
        static_in_ch=1
    ).to(device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"])
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )

    # Load checkpoint if specified
    if checkpoint_epoch is not None:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{checkpoint_epoch}"
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        logger.info(f"Resuming training from epoch {start_epoch + 1}")

    es_config = SimpleNamespace(patience=cfg["patience"], min_patience=0)
    early_stopper = EarlyStopping(es_config, verbose=True)

    scaler = torch.amp.GradScaler(enabled=cfg["amp"])
    
    # Initialize loss tracking for plotting
    train_losses = []
    val_losses = []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def align_preds_targets(preds, targets):
        if preds.shape == targets.shape:
            return preds, targets

        if targets.dim() == preds.dim() + 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        if preds.dim() == targets.dim() + 1 and preds.size(1) == 1:
            preds = preds.squeeze(1)

        return preds.squeeze(), targets.squeeze()
    
    def process_datasets(datasets, replace_nan=0.0, fill_nans=False, skip_normalize=False):
        
        if fill_nans:
            cleaned = nan_checks_replace(datasets, replace_nan=replace_nan)
        else:
            cleaned = datasets

        if skip_normalize:
            return cleaned
        else:
            return standardize_tensor(cleaned, replace_nan=replace_nan)

    def train_one_epoch(args):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            export_batches(
                batch_idx, epoch,
                {'x_high': x_high, 'x_med': x_med, 'x_static': x_static, 'y_batch': y_batch},
                run_dir, logger
            )

            x_high, x_med, x_static, y_batch = [
                b.to(device, non_blocking=True) for b in batch
            ]
            x_high, x_med, x_static, y_batch = process_datasets(
                [x_high, x_med, x_static, y_batch], 
                replace_nan=-99, 
                fill_nans=args.fillna
            )

            optimizer.zero_grad()

            with torch.amp.autocast(enabled=cfg["amp"], device_type=cfg["device"]):
                y_pred = model(x_high, x_med, x_static)
                y_pred, y_batch = align_preds_targets(y_pred, y_batch)

                mask = torch.isfinite(y_batch).float()

                debug_nan([x_high, x_med, x_static, y_pred, y_batch, mask],
                    ['x_high','x_med','x_static','y_pred','y_batch','mask'])

                loss = masked_custom_loss(criterion, y_pred, y_batch, mask)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    logger.warning("[train] Loss is NaN, skipping this batch")
                    continue
                else:
                    scaler.scale(loss).backward()

            if cfg["grad_clip"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / max(1, n_batches)

    def validate_one_epoch(args):
        model.eval()
        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                x_high, x_med, x_static, y_batch = [b.to(device) for b in batch]
                x_high, x_med, x_static, y_batch = process_datasets([x_high, x_med, x_static, y_batch], replace_nan=-99, fill_nans=args.fillna)

                with torch.amp.autocast(enabled=cfg["amp"], device_type=cfg["device"]):
                    y_pred = model(x_high, x_med, x_static)
                    y_pred, y_batch = align_preds_targets(y_pred, y_batch)

                    mask = torch.isfinite(y_batch).float()
                    loss = masked_custom_loss(criterion, y_pred, y_batch, mask)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    logger.warning("[validate] Loss is NaN, skipping this batch")
                    continue

                running_loss += loss.item()
                n_batches += 1

        return running_loss / max(1, n_batches)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch + 1, cfg["num_epochs"] + 1):
        train_loss = train_one_epoch(args)
        val_loss = validate_one_epoch(args)
        
        # Track losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        plot_learning_curves(train_losses, val_losses, plot_dir, logger)

        model_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        early_stopper(val_loss, model_dict, epoch, str(checkpoint_dir))

        if early_stopper.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info("Training finished.")
    

def plot_learning_curves(train_losses, val_losses, plot_dir, logger):
    # ------------------------------------------------------------------
    # Plot learning curves
    # ------------------------------------------------------------------
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



if __name__ == "__main__":
    import argparse
    from definitions import ROOT_DIR
    import os
    from dask.diagnostics import ProgressBar
    import torch
    parser = argparse.ArgumentParser(description="Run the dataset preparation pipeline.")
    parser.add_argument("--sample", action="store_true", help="Whether to run in sample mode with fewer points and times for quick testing.")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 32)), help="Batch size for training")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--fillna", action="store_true", help="Whether to fill NaN values in the dataset with a specified value (e.g., -99)")
    parser.add_argument("--checkpoint", type=int, default=None, help="Epoch number to resume from (loads from most recent run directory)")
        
    args = parser.parse_args()
    # dataset_pipeline(sample=args.sample)
    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "train_split": 0.8,
        "patience": args.patience,
        "num_workers": 8,
        "amp": True,
        "grad_clip": 1.0,
        "save_dir": ROOT_DIR / "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # tb = ProgressBar().register()

    main(config=config, checkpoint_epoch=args.checkpoint, sample=args.sample, fillna=args.fillna)