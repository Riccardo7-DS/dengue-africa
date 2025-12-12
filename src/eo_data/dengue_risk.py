from definitions import DATA_PATH
from pathlib import Path
import rioxarray
import pyreadr
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from typing import Union
import logging
import xarray as xr
from utils import prepare
from tqdm import tqdm
logger = logging.getLogger(__name__)

# path = "riskmaps_public main data/intermediate_datasets/DEN_occ_map_extract.rds"

# df_occ_dengue = pyreadr.read_r(Path(DATA_PATH) / path )[None]
# df_survey = pyreadr.read_r(Path(DATA_PATH) / "riskmaps_public main data/intermediate_datasets/Surv_model_fit_data.rds")[None]


def load_nightlights_data(tif_dir, n_sample_times=0, reproject:bool=False) -> xr.DataArray:
    # ------------------------------
    # 2. Load MODIS nightlight rasters
    # ------------------------------
    files = sorted(Path(tif_dir).glob("*.tif"))

    arrays, times, bad_files = [], [], []
    for f in tqdm(files):
        try:
            arr = rioxarray.open_rasterio(f, chunks=True).isel(band=0)
            if reproject:
                arr = arr.rio.reproject("EPSG:4326")
            if len(arrays) >= n_sample_times and n_sample_times !=0:
                break
            arrays.append(arr)
            date_str = f.stem.split("_")[-1]
            times.append(pd.to_datetime(date_str, format="%Y%m%d"))
        except Exception as e:
            logger.warning(f"Could not open {f}: {e}")
            bad_files.append(f)

    if not arrays:
        raise RuntimeError("No valid raster files could be opened.")

    da = xr.concat(arrays, dim="time").assign_coords(time=("time", times))
    da = da * 0.0001
    da = da.chunk({"y": 1000, "x": 1000})
    return da


def read_predict_arbodata(risk_data_path:str, model="Logit"):

    df_arbo = pyreadr.read_r(Path(DATA_PATH) / risk_data_path)[None]
    df_arbo = df_arbo[df_arbo["disease"] == "dengue"].reset_index(drop=True)
    df_arbo = df_arbo[df_arbo["Admin"].isin([-999, 2.0, 1.0])]

    if model not in ["Logit", "GAM"]:
        raise ValueError("Model must be either 'Logit' or 'GAM'")

    # da_den_prev =  rioxarray.open_rasterio(Path(DATA_PATH) / "riskmaps_public main data" / "intermediate_datasets" / "DEN_previous_binrast.tif" )


    # 1️⃣ Prepare y
    y = df_arbo["PA"].astype(int)

    eps = 1e-6  # to avoid log(0)
    df_arbo['p_det_clamped'] = np.clip(df_arbo['Surv'], eps, 1 - eps)
    df_arbo['logit_psurv'] = np.log(df_arbo['p_det_clamped'] / (1 - df_arbo['p_det_clamped']))

    # 2️⃣ Prepare X
    X = df_arbo.drop(columns=["PA", "disease", "Admin", "GAUL", 'Surv', 'p_det_clamped'])

    # Convert categorical columns to dummies
    X = pd.get_dummies(X, drop_first=True)

    # Ensure all boolean columns are converted to int
    X = X.apply(lambda col: col.astype(int) if col.dtype == "bool" else col)

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors='raise')


    if model == "Logit":
        # 3️⃣ Add constant for intercept
        X = sm.add_constant(X)

        # 4️⃣ Fit Logit
        mdl = Logit(endog=y, exog=X)
        result = mdl.fit()
        logger.info(result.summary())
        dengue_risk = result.predict(X)

    elif model == "GAM":
        from pygam import LogisticGAM, s, te, l
        # --- define the GAM using the working syntax ---
        result = LogisticGAM(
        
                # 2D spatial smooth (lon, lat)
                te(0, 1)

                # splines for continuous predictors: columns 2–10
                + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10)

                + s(11) + l(12) + l(13) + l(14) + l(15) + l(16) + l(17) + l(18) + l(19) + l(20) + l(21) + l(22) + l(23)# linear terms for binary predictors

            ).fit(X, y)
        
        logger.info(result.summary())
        dengue_risk = result.predict_mu(X)


    
    df_arbo["pred_dengue_risk"] = dengue_risk
    return df_arbo
