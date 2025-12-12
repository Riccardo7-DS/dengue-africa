import logging
logger = logging.getLogger(__name__)

def run_pipeline():
    from models import DengueConvLSTM
    import torch
    from utils import admin2_aggregate, aggregate_to_admin, load_data
    from torch.nn import MSELoss
    from definitions import DATA_PATH

    data_path = DATA_PATH / "Spatial_extract_V1_3.csv"
    weekly_tabular_data = load_data(data_path, 
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



def dataset_pipeline(sample=False):
    import rioxarray
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

    n_sample_points = 50
    n_sample_times = 10

    # ------------------------------
    # 1. Load dengue risk data
    # ------------------------------
    risk_data_path = "riskmaps_public main data/intermediate_datasets/Arbo_model_fit_data.rds"
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

    tif_dir = DATA_PATH / "modis/VNP46A3_061/tiffs/-83.1_-55.1_-28.8_13.4"
    da = load_nightlights_data(tif_dir, n_sample_times= n_sample_times if sample else 0)

    lat_da = xr.DataArray(df_risk["Latitude"].values, dims="points")
    lon_da = xr.DataArray(df_risk["Longitude"].values, dims="points")

    with ProgressBar():
        ds_values = da.interp(lat=lat_da, lon=lon_da, method="nearest").compute()

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
        ds_era = prepare(ds_era).rename({"valid_time": "time"})
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



if __name__ == "__main__":
    dataset_pipeline(False)