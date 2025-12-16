# from utils import process_gdf, countries_to_bbox
# from definitions import DATA_PATH
# from eo_data.era5 import Era5CDS, era5_variables
# from pathlib import Path
# from dotenv import load_dotenv
# from utils import init_logging

# init_logging("era5_logger.log")
# load_dotenv()

# year_min = 2000
# year_max = 2024

# # country = ["Administrative divisions of Taiwan"]
# # era5_downloader = Era5CDS(era5_variables, output_name="Taiwan")
# # era5_downloader.run(country, year_min, year_max)

# # country = ["Pakistan"]
# # era5_downloader = Era5CDS(era5_variables, output_name="Pakistan")
# # era5_downloader.run(country, year_min, year_max)

# countries = ["Peru", "Brazil", "Colombia", "Panama", "Argentina"]
# era5_downloader = Era5CDS(era5_variables, output_name="latin_america")
# era5_downloader.run(countries, year_min, year_max)


from definitions import DATA_PATH, ROOT_DIR
from torchgeo.datasets import RasterDataset, XarrayDataset
from torch.utils.data import DataLoader, Dataset
import sys
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets.utils import stack_samples, lazy_import
import torch.nn.functional as F
from pathlib import Path
from utils import latin_box,  load_admin_data, process_gdf, df_to_xarray, rasterize_timeseries, init_logging, prepare
import torch 
import pandas as pd 
import numpy as np
from datetime import timedelta
import os 
import xarray as xr
from torch.nn import MSELoss
from models.model_utils import EarlyStopping, collate_skip_none, MetricsRecorder, rolling_split
from models.config import config_transf as model_config
import argparse
from tqdm import tqdm
import logging
import math

logger = init_logging(Path(ROOT_DIR) / "training_transformer.log")
# simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser(description='test', conflict_handler="resolve")
parser.add_argument('--model', default=os.environ.get('model', "TRANSF"))
parser.add_argument('--mode', default=os.environ.get('mode', "train"))

args = parser.parse_args()

save_dataset = True
time_sample = 10 

viirs_data_path = DATA_PATH / "modis/VNP46A3_061/tiffs/-83.1_-55.1_-28.8_13.4/"
risk_raster_path = DATA_PATH / "riskmaps_public main data" 
era5_path = DATA_PATH / "ERA5"
y_path = DATA_PATH / "Spatial_extract_V1_3.csv"


if os.path.exists(DATA_PATH / "weekly_admin2_cases.nc"):
    raster_da = xr.load_dataarray(DATA_PATH / "weekly_admin2_cases.nc")
else:
    data_process = load_admin_data(y_path, temporal_resolution="Week", spatial_resolution="Admin2")
    data_process = data_process[data_process["year"]>=2012].reset_index(drop=True)

    countries = [ "Colombia", "Brazil", "Peru", "Argentina", "Panama"]
    # Path to the zip file containing the shapefile
    zip_path = DATA_PATH / "shapefiles" / "GAUL_2024.zip"

    gdf = process_gdf(zip_path, countries=countries)
    # dissolve all admin2 into admin0 boundaries
    gdf_admin0 = gdf.dissolve(by="adm_0_name", as_index=False)

    da = df_to_xarray(data_process, countries=countries, fill_value=np.nan)
    raster_da = rasterize_timeseries(da, gdf, region_col="region_id", res=0.05)

    if save_dataset:
        raster_da.to_netcdf(DATA_PATH / "weekly_admin2_cases.nc")

class VIIRSData(RasterDataset):
    filename_glob = "VNP*.tif"
    filename_regex = r'^VNP46A3_(?P<date>\d{8})\.tif$'
    date_format = "%Y%m%d"
    separate_files = False
    is_image = True
    all_bands = None
    rgb_bands = None


class XarrayDataset(RasterDataset):

    def __init__(self, root, variables=None, transform=None, T_MAX=1):
        xr = lazy_import("xarray")
        self.root = Path(root)
        self.variables = variables
        self.transform = transform

        self._files = sorted(self.root.glob(self.filename_glob))
        if len(self._files) == 0:
            raise ValueError(f"No files found in {root} matching {self.filename_glob}")

        self._aggregated_dataset = xr.open_mfdataset(
            [str(f) for f in self._files], combine="by_coords"
        )
        self.res = prepare(self._aggregated_dataset).rio.resolution()
        self._aggregated_dataset = self._normalize_lat_lon_coords(self._aggregated_dataset)
        self.T_max = T_MAX

    @property
    def files(self):
        return self._files

    @property
    def index(self):
        """Return a list of time indices for TorchGeo compatibility"""
        return pd.to_datetime(self._aggregated_dataset["time"].values)
        

    def _normalize_lat_lon_coords(self, src):
        rename_map = {}
        for name in ("x", "lon", "longitude"):
            if name in src.coords and name != "x":
                rename_map[name] = "x"
                break
        for name in ("y", "lat", "latitude"):
            if name in src.coords and name != "y":
                rename_map[name] = "y"
                break
        for name in ("time", "datetime", "valid_time", "date"):
            if name in src.coords and name != "time":
                rename_map[name] = "time"
                break
        if rename_map:
            src = src.rename(rename_map)
        if "y" in src.coords and src["y"].values[0] > src["y"].values[-1]:
            src = src.sortby("y")
        return src

    def __getitem__(self, query):
        """
        query: tuple of slices (x_slice, y_slice[, t_slice])
        Returns: {"image": torch.Tensor} of shape:
            - [C, H, W] if single time
            - [C, T, H, W] if multiple times
        """
        if len(query) == 2:
            x_slice, y_slice = query
            t_slice = slice(None)
        elif len(query) == 3:
            x_slice, y_slice, t_slice = query
        else:
            raise ValueError(f"Expected query of length 2 or 3, got {len(query)}")

        ds = self._aggregated_dataset

        # select time using sel
        if t_slice != slice(None):
            start = t_slice.start if isinstance(t_slice.start, pd.Timestamp) else None
            stop = t_slice.stop if isinstance(t_slice.stop, pd.Timestamp) else None
            ds = ds.sel(time=slice(start, stop))

        # select spatial indices
        ds_patch = ds.sel(
            x=slice(x_slice.start, x_slice.stop),
            y=slice(y_slice.start, y_slice.stop)
        )

        # select variables
        data_vars = self.variables or list(ds_patch.data_vars)

        # convert to torch tensor
        patch_list = []
        for var in data_vars:
            arr = ds_patch[var].values  # shape: [T,H,W] or [H,W]
            arr = torch.tensor(arr)
            if arr.ndim == 2:
                arr = arr.unsqueeze(0)  # [1,H,W]
            elif arr.ndim == 3:
                # [T,H,W] → [1,T,H,W] per variable
                arr = arr.unsqueeze(0)
            patch_list.append(arr)

        # concatenate variables along channel dimension
        patch = torch.cat(patch_list, dim=0)  # [C,T,H,W] if multiple times, [C,H,W] if single time
        
        # If there is a time dimension, swap C and T
        if patch.ndim == 4:
            patch = patch.permute(1, 0, 2, 3)  # [T, C, H, W]
        
        
        if patch.shape[0] < self.T_max:
            pad_len = self.T_max - patch.shape[0]
            pad = torch.zeros((pad_len, *patch.shape[1:]), dtype=patch.dtype)
            patch = torch.cat([pad, patch], dim=0)

        if self.transform:
            patch = self.transform(patch)

        return {"image": patch}
    
    def __len__(self):
        return len(self._aggregated_dataset["time"])

class ERA5Daily(XarrayDataset):
    filename_glob = "era5land_latin_america*.nc"
    filename_regex = r"era5land_latin_america_(?P<date>\d{4})\.nc$"
    date_format = "%Y"
    is_image = True
    separate_files = False

class StaticLayer(RasterDataset):
    filename_glob = "DEN_riskmap_wmean_masked.tif"
    is_image = True
    separate_files = False

class DIRData(XarrayDataset):
    filename_glob = "weekly_admin2_cases.nc"
    is_image = False
    separate_files = False


class DengueDataset(Dataset):
    """
    Multi-resolution dataset:
    - VIIRS (high resolution)
    - ERA5 (medium resolution)
    - Static maps (coarse)
    """

    def __init__(
        self,
        y,
        viirs,
        era5,
        static,
        bbox,
        patch_size=(1024, 1024),
        transform=None,
        offset_days=7,
    ):
        self.y = y
        self.viirs = viirs
        self.era5 = era5
        self.static = static
        self.bbox = bbox
        self.patch_size = patch_size
        self.transform = transform
        self.offset_days = offset_days

        self.samples = []
        self._generate_samples()


    def _derive_width_height(self, coordinate_span, pixel_resolution):
        return int(round(coordinate_span / pixel_resolution))
    

    def _target_patch_size(self, high_patch_size, high_res, low_res):
        return int(math.ceil(high_patch_size * abs(high_res) / abs(low_res)))

    def _padding_coarse(self, X, target_h, target_w):
        """
        Pads a coarse-resolution tensor to (target_h, target_w)
        Padding is applied on bottom and right only.
        """
        if len(X.shape) == 4:
            c, t, h, w = X.shape
        elif len(X.shape) == 3:
            t, h, w = X.shape
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        if pad_h > 0 or pad_w > 0:
            # F.pad format: (left, right, top, bottom)
            X = F.pad(X, (0, pad_w, 0, pad_h))

        return X

    # -------------------------------------------------
    # Build (time, space) index
    # -------------------------------------------------
    def _generate_samples(self):
        
        for t in tqdm(range(len(self.y.index))):
            t_current = self.y.index[t]

            t_next = self.y.index[t + 1] if t + 1 < len(self.y.index) else t_current

            t_query_start = t_current - timedelta(days=self.offset_days)
            t_query_end = t_current

            sampler = GridGeoSampler(
                self.viirs,
                size=self.patch_size,
                roi=self.bbox,
                toi=pd.Interval(t_query_start, t_query_end)
            )
            
            queries = list(sampler)
            dropped = 0

            if len(queries) == 0:
                continue  # safely skip

            for (x_slice, y_slice, _) in queries:
                h = self._derive_width_height((x_slice.stop - x_slice.start), self.viirs.res[0])
                w = self._derive_width_height((y_slice.stop - y_slice.start), self.viirs.res[1])

                # Skip patch if it’s smaller than desired patch_size
                if h != self.patch_size[0] or w != self.patch_size[1]:
                    dropped += 1
                    continue


                # Store slice info only; no array is loaded
                self.samples.append(
                    (t_query_start, t_query_end, t_next, x_slice, y_slice)
                )
                if dropped > 0:
                    logger.info(f"In total {dropped} samples have been dropped")

    def get_target_times(self):
        return np.array([s[2] for s in self.samples]) 
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_start, t_end, t_next, x_slice, y_slice = self.samples[idx]

        query_x = (x_slice, y_slice, slice(t_start, t_end))
        query_y = (x_slice, y_slice, slice(t_next, t_next))

        # Load data lazily
        x_high = self.viirs[query_x]["image"].float()
        x_med = self.era5[query_x]["image"].float()
        x_static = self.static[query_x]["image"].float()
        y = self.y[query_y]["image"].float()

        target_h_med = self._target_patch_size(
            self.patch_size[0], self.viirs.res[0], self.era5.res[0])
        target_w_med = self._target_patch_size(
            self.patch_size[1], self.viirs.res[1], self.era5.res[1])
        x_med = self._padding_coarse(x_med, target_h_med, target_w_med)

        # Static
        target_h_static = self._target_patch_size(
            self.patch_size[0], self.viirs.res[0], self.static.res[0])
        target_w_static = self._target_patch_size(
            self.patch_size[1], self.viirs.res[1], self.static.res[1])
        x_static = self._padding_coarse(x_static, target_h_static, target_w_static)

        ## target
        target_h_y = self._target_patch_size(
            self.patch_size[0], self.viirs.res[0], self.y.res[0])
        target_w_y = self._target_patch_size(
            self.patch_size[1], self.viirs.res[1], self.y.res[1])
        y = self._padding_coarse(y, target_h_y, target_w_y)

        # Normalize per-sample (optional but consistent with your intent)
        x_high = (x_high - x_high.mean()) / (x_high.std() + 1e-6)
        x_med = (x_med - x_med.mean()) / (x_med.std() + 1e-6)
        x_static = (x_static - x_static.mean()) / (x_static.std() + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)

        if self.transform:
            x_high, x_med, x_static = self.transform(
                x_high, x_med, x_static
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"Index {idx} High-res: {x_high.shape}")
            logger.info(f"Index {idx} Medium-res: {x_med.shape}")
            logger.info(f"Index {idx} Static: {x_static.shape}")
            logger.info(f"Index {idx} y: {y.shape}")

        return x_high, x_med, x_static, y

bbox = latin_box()

#bbox = [xmin, ymin, xmax, ymax] in CRS coordinates
patch_size = model_config.patch_size

# Base datasets
y_dir = DIRData(DATA_PATH, T_MAX=1)
era5 = ERA5Daily(era5_path, T_MAX=32)
risk = StaticLayer(risk_raster_path)
viirs = VIIRSData(viirs_data_path)

dataset = DengueDataset(
    y=y_dir, 
    viirs=viirs, 
    era5=era5, 
    static=risk, 
    bbox=bbox, 
    patch_size=patch_size
)

train_ds, val_ds = rolling_split(
    dataset,
    train_end=pd.Timestamp("2018-12-31"),
    horizon_days=365,
)

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,      # ✔️ OK for training
    num_workers=4,
    collate_fn=collate_skip_none,
    persistent_workers=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,     # ❌ NEVER shuffle validation
    num_workers=4,
    collate_fn=collate_skip_none,
)

from models.transformer import DenguePredictor

model = DenguePredictor(
    high_in_ch=3,
    med_in_ch=8,
    static_in_ch=1,
    high_out=128,
    med_out=128,
    static_out=128,
    hidden_dim=256,
    output_size=(22, 22)
)

if model_config.masked_loss is False:
    criterion = MSELoss().to(model_config.device)
else:
    criterion = MSELoss(reduction='none').to(model_config.device)

learning_rate = model_config.learning_rate
early_stopping = EarlyStopping(model_config, verbose=True)

optimizer = torch.optim.Adam(model.parameters(), 
    lr=learning_rate,
    weight_decay=model_config.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=model_config.scheduler_factor, 
    patience=model_config.scheduler_patience, 
)

metrics_recorder = MetricsRecorder()

class Trainer():
    def __init__(self, 
        args, 
        config, 
        model,  
        optimizer, 
        scheduler, 
        loss,
        checkpoint_dir,
        ema = None
    ): 
        self.config = config
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = config.device
        self.loss = loss
        self.model = model.to(self.device)
        self.ema = ema
        self.checkpoint_dir = checkpoint_dir
        self._prev_lr = None


def validation_loop(epoch, validation_loader, trainer):
    """Run validation over `validation_loader` and return aggregated metrics.

    Returns a dict with keys: 'loss', 'mape', 'rmse', 'lr' (per-batch lists).
    """
    import sys
    from models import masked_mape, masked_rmse, masked_custom_loss, mask_mape, mask_rmse

    trainer.model.eval()
    val_records = {'loss': [], 'mape': [], 'rmse': [], 'lr': []}

    with torch.no_grad():
        for batch in validation_loader:
            if batch is None:
                continue

            x_high, x_med, x_static, y = batch
            # Move to device
            x_high = x_high.to(trainer.device)
            x_med = x_med.to(trainer.device)
            x_static = x_static.to(trainer.device)
            y = y.to(trainer.device)
            
            outputs = trainer.model(x_high, x_med, x_static)
            y = y.squeeze(1)

            mask_batch = torch.isfinite(y)

            if not model_config.masked_loss:
                loss = trainer.loss(outputs, y)
                mape = masked_mape(outputs, y).item()
                rmse = masked_rmse(outputs, y).item()
            else:
                mask_t = mask_batch.float().to(trainer.device)
                loss = masked_custom_loss(trainer.loss, outputs, y, mask_t)
                mape = mask_mape(outputs, y, mask_t).item()
                rmse = mask_rmse(outputs, y, mask_t).item()

            val_records['loss'].append(loss.item())
            val_records['mape'].append(mape)
            val_records['rmse'].append(rmse)

    # aggregate
    mean_loss = np.mean(val_records['loss'])
    mean_mape = np.mean(val_records['mape'])
    mean_rmse = np.mean(val_records['rmse'])

    trainer.scheduler.step(mean_loss)

    prev_lr = getattr(trainer, "_prev_lr", None)
    current_lr = trainer.optimizer.param_groups[0]['lr']
    if prev_lr is not None and current_lr < prev_lr:
        logger.info(
            f"LR reduced from {prev_lr:.2e} → {current_lr:.2e} at epoch {epoch}"
        )

    trainer._prev_lr = current_lr
    val_records['lr'].append(current_lr)
    

    logger.info(
        f"Epoch: {epoch:03d}, "
        f"Val Loss: {mean_loss:.4f}, "
        f"Val MAPE: {mean_mape:.4f}, "
        f"Val RMSE: {mean_rmse:.4f}, "
        f"LR: {current_lr:.2e}"
    )

    model_dict = {
        'epoch': epoch,
        'state_dict': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'lr_sched': trainer.scheduler.state_dict()
    }

    # Early stopping driven by validation loss
    early_stopping(mean_loss, model_dict, epoch, trainer.checkpoint_dir)
    stop = early_stopping.early_stop
    
    if stop:
        logger.info("Early stopping")

    return val_records, stop


def training_loop(epoch, dataloader, trainer):
    from models import masked_mape, masked_rmse, masked_custom_loss, mask_mape, mask_rmse

    trainer.model.train()
    epoch_records = {'loss': [], "mape": [], "rmse": []}

    for batch in dataloader:
        if batch is None:
            continue
        x_high, x_med, x_static, y = batch
        # Move to device
        x_high = x_high.to(trainer.device)
        x_med = x_med.to(trainer.device)
        x_static = x_static.to(trainer.device)
        y = y.to(trainer.device)
        
        outputs = trainer.model(x_high, x_med, x_static)
        y = y.squeeze(1)
        mask_batch = torch.isfinite(y)

        if not model_config.masked_loss:
            loss = trainer.loss(outputs, y)
            mape = masked_mape(outputs, y).item()
            rmse = masked_rmse(outputs, y).item()
        else:
            mask_t = mask_batch.float().to(trainer.device)
            loss = masked_custom_loss(trainer.loss, outputs, y, mask_t)
            mape = mask_mape(outputs, y, mask_t).item()
            rmse = mask_rmse(outputs, y, mask_t).item()

        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        epoch_records['loss'].append(loss.item())
        epoch_records['mape'].append(mape)
        epoch_records['rmse'].append(rmse)
        
    # ---- epoch-level logging ----
    mean_loss = np.mean(epoch_records['loss'])
    mean_mape = np.mean(epoch_records['mape'])
    mean_rmse = np.mean(epoch_records['rmse'])

    logger.info(
        f"Epoch: {epoch:03d}, "
        f"Train Loss: {mean_loss:.4f}, "
        f"Train MAPE: {mean_mape:.4f}, "
        f"Train RMSE: {mean_rmse:.4f}"
    )
    return epoch_records

def pipeline(
    trainer,
    start_epoch=0):
    
    for epoch in tqdm(range(start_epoch, trainer.config.epochs)):
        
        train_records = training_loop(epoch, train_loader, trainer)
        val_records, stop = validation_loop(epoch, val_loader, trainer)
        if stop:
            break

        metrics_recorder.add_train_metrics(train_records, epoch)
        metrics_recorder.add_val_metrics(val_records, epoch)


checkpoint_dir = ROOT_DIR / "outputs" / f"{args.model}" / "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = None

try:
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        checkp_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {checkp_epoch}")
    
    start_epoch = 0 if checkpoint_path is None else checkp_epoch

except FileNotFoundError:
    from models.model_utils import load_checkpoint
    model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler) 

trainer = Trainer(
    args, 
    config=model_config, 
    model=model, 
    optimizer=optimizer, 
    scheduler=scheduler, 
    loss=criterion,
    checkpoint_dir=checkpoint_dir,
)

pipeline(
    trainer,
    start_epoch=start_epoch
)
