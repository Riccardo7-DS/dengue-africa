from cuda.bindings import utils
from definitions import DATA_PATH
from pathlib import Path
import rioxarray
import pyreadr
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import logging
import xarray as xr
from tqdm import tqdm
import torch
import pandas as pd 
from torchgeo.samplers import GridGeoSampler
import numpy as np


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

    df_arbo = pyreadr.read_r(risk_data_path)[None]
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



if __name__=="__main__":
    from models import VIIRSData, ERA5Daily, StaticLayer, XarraySpatioTemporalDataset  
    from definitions import DATA_PATH
    from pathlib import Path   
    import logging
    from models import collate_skip_none, DengueDataset
    from utils import latin_box
    logger = logging.getLogger(__name__)
    viirs_data_path = DATA_PATH / "modis" / "VIIRS_nightlight"
    era5_path = DATA_PATH / "ERA5"
    risk_raster_path = DATA_PATH / "riskmaps_public main data"/ "DEN_riskmap_wmean_masked.tif"
    admin_path = DATA_PATH / "weekly_admin2_cases.nc"
    # -----------------------------------------------------------------------------
    # Instantiate everything
    # -----------------------------------------------------------------------------
    era5 = ERA5Daily(era5_path, T_max=32)
    viirs = VIIRSData(viirs_data_path)
    static = StaticLayer(risk_raster_path)
    ds_cases = xr.open_dataset(admin_path)
    y = XarraySpatioTemporalDataset(ds_cases,T_max=1)
    dataset = DengueDataset(viirs, era5, static, y, bbox=latin_box())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_skip_none,
    )
    print(f"Dataset bbox clipped to ERA5 valid extent: {dataset.bbox}")

    from models import DenguePredictor
    from models.model_utils import masked_custom_loss, EarlyStopping
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from pathlib import Path
    from types import SimpleNamespace

    # Training config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    learning_rate = 1e-4
    save_dir = Path("./checkpoints")
    save_dir.mkdir(exist_ok=True)

    # Model, loss, optimizer
    model = DenguePredictor(high_in_ch=3, 
                            med_in_ch=8, 
                            static_in_ch=1).to(device)
    # Use unreduced criterion so masked_custom_loss can apply the mask
    criterion = nn.MSELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Early stopping (uses config-like object)
    es_config = SimpleNamespace(patience=5, min_patience=0)
    early_stopper = EarlyStopping(es_config, verbose=True)

    # Helper: align preds and targets to compatible shapes
    def align_preds_targets(preds, targets):
        # If identical, quick return
        if preds.shape == targets.shape:
            return preds, targets

        # Common case: targets have an extra time dim of length 1: [B, T=1, C, H, W]
        if targets.dim() == preds.dim() + 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)
            if preds.shape == targets.shape:
                return preds, targets

        # Common case: preds have an extra singleton channel dim
        if preds.dim() == targets.dim() + 1 and preds.size(1) == 1:
            preds = preds.squeeze(1)
            if preds.shape == targets.shape:
                return preds, targets

        # Try squeezing any singleton dims from targets
        t_squeezed = targets.squeeze()
        if preds.shape == t_squeezed.shape:
            return preds, t_squeezed

        # Try squeezing preds
        p_squeezed = preds.squeeze()
        if p_squeezed.shape == targets.shape:
            return p_squeezed, targets

        # As a last resort, try expanding preds to targets shape if compatible
        try:
            preds_exp = preds.expand_as(targets)
            return preds_exp, targets
        except Exception:
            return preds, targets

    # Simple training + validation loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(dataloader):
            if batch is None:
                continue

            x_high, x_med, x_static, y_batch = batch
            x_high = x_high.to(device)
            x_med = x_med.to(device)
            x_static = x_static.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_high, x_med, x_static)

            # Align shapes
            y_pred, y_batch = align_preds_targets(y_pred, y_batch)

            # Optional debug for first few batches
            if i < 3:
                print(f"Train batch {i}: y_pred {y_pred.shape}, y_batch {y_batch.shape}")

            # Compute mask where target is finite (after alignment)
            mask_batch = torch.isfinite(y_batch)
            mask_t = mask_batch.float().to(device)

            # Compute masked loss (masked_custom_loss returns scalar)
            loss = masked_custom_loss(criterion, y_pred, y_batch, mask_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch}/{num_epochs} — train loss: {avg_train_loss:.6f} — batches: {n_batches}")

        # Validation: quick pass over up to 5 batches
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for j, batch in enumerate(dataloader):
                if batch is None:
                    continue
                x_high, x_med, x_static, y_batch = batch
                x_high = x_high.to(device)
                x_med = x_med.to(device)
                x_static = x_static.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(x_high, x_med, x_static)

                # Align shapes
                y_pred, y_batch = align_preds_targets(y_pred, y_batch)

                if j < 3:
                    print(f"Val batch {j}: y_pred {y_pred.shape}, y_batch {y_batch.shape}")

                mask_batch = torch.isfinite(y_batch)
                mask_t = mask_batch.float().to(device)
                vloss = masked_custom_loss(criterion, y_pred, y_batch, mask_t)
                val_loss += vloss.item()
                val_batches += 1
                if val_batches >= 5:
                    break

        avg_val_loss = val_loss / max(1, val_batches)
        print(f"Epoch {epoch} — val loss: {avg_val_loss:.6f} — val_batches: {val_batches}")

        # Early stopping check (saves checkpoint on improvement)
        model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        early_stopper(avg_val_loss, model_dict, epoch, str(save_dir))

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Training finished")
