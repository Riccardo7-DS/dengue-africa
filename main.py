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
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets.utils import stack_samples, lazy_import
from pathlib import Path
from utils import latin_box,  load_admin_data, process_gdf, df_to_xarray, rasterize_timeseries, init_logging
import torch 
import pandas as pd 
import numpy as np
from datetime import timedelta
import os 
import xarray as xr
from torch.nn import MSELoss
from models.model_utils import EarlyStopping, collate_pad
from models.config import config_transf as model_config
import argparse
from tqdm import tqdm
import logging

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

    # -------------------------------------------------
    # Build (time, space) index
    # -------------------------------------------------
    def _generate_samples(self):
        for t in range(len(self.y.index)):
            t_current = self.y.index[t]

            if t + 1 < len(self.y.index):
                t_next = self.y.index[t + 1]
            else:
                t_next = self.y.index[t]

            t_query_start = t_current - timedelta(days=self.offset_days)
            t_query_end = t_current

            sampler = GridGeoSampler(
                self.viirs,
                size=self.patch_size,
                roi=self.bbox,
                toi=pd.Interval(t_query_start, t_query_end)
            )

            queries = list(sampler)

            if len(queries) == 0:
                continue  # safely skip

            for (x_slice, y_slice, _) in tqdm(queries, desc="Preparing patches..."):
                self.samples.append(
                    (
                        t_query_start,
                        t_query_end,
                        t_next,
                        x_slice,
                        y_slice,
                    )
                )

    # -------------------------------------------------
    # PyTorch API
    # -------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (
            t_query_start,
            t_query_end,
            t_next,
            x_slice,
            y_slice,
        ) = self.samples[idx]

        query_x = (x_slice, y_slice, slice(t_query_start, t_query_end))
        query_y = (x_slice, y_slice, slice(t_next, t_next))

        # Load data lazily
        x_high = self.viirs[query_x]["image"].float()
        x_med = self.era5[query_x]["image"].float()
        x_static = self.static[query_x]["image"].float()
        y = self.y[query_y]["image"].float()

        # Normalize per-sample (optional but consistent with your intent)
        x_high = (x_high - x_high.mean()) / (x_high.std() + 1e-6)
        x_med = (x_med - x_med.mean()) / (x_med.std() + 1e-6)
        x_static = (x_static - x_static.mean()) / (x_static.std() + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)

        if self.transform:
            x_high, x_med, x_static = self.transform(
                x_high, x_med, x_static
            )

        return x_high, x_med, x_static, y


bbox = latin_box()

#bbox = [xmin, ymin, xmax, ymax] in CRS coordinates
patch_size = (1024, 1024)

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

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_pad,
    num_workers=0
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
    output_dim=1
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

class Trainer():
    def __init__(self, 
        args, 
        config, 
        model,  
        optimizer, 
        scheduler, 
        loss,
        ema = None
    ): 
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = config.device
        self.loss = loss
        self.model = model
        self.ema = ema

def training_loop(
                  dataloader,
                  trainer,
                  checkpoint_dir,
                  mask = None,
                  start_epoch=0):
    
    from models import masked_mape, masked_rmse, masked_custom_loss, mask_mape, mask_rmse

    train_loss_records = []
    epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
    
    for epoch in tqdm(range(start_epoch, trainer.config.epochs)):
        model.train()

        for batch in dataloader:
            if batch is None:
                continue
            x_high, x_med, x_static, y = batch
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.info(f"High-res: {x_high.shape}")
                logger.info(f"Medium-res: {x_med.shape}")
                logger.info(f"Static: {x_static.shape}")

            outputs = trainer.model(x_high, x_med, x_static)
            y = y.squeeze(1)
            mask = np.isfinite(y)

            if model_config.masked_loss is False:
                losses = trainer.loss(outputs, y)
                mape = masked_mape(outputs, y).item()
                rmse = masked_rmse(outputs, y).item()
            else:
                if mask is None:
                    raise ValueError("Please provide a mask for loss computation")
                else:
                    mask = mask.float().to(trainer.config.device)
                    losses =  masked_custom_loss(trainer.loss, outputs, y, mask)
                    mape = mask_mape(outputs, y, mask).item()
                    rmse = mask_rmse(outputs, y, mask).item()

            trainer.optimizer.zero_grad()
            losses.backward()
            trainer.optimizer.step()

            epoch_records['loss'].append(losses.item())
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)

            train_loss_records.append(np.mean(epoch_records['loss']))

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            logger.info(log.format(epoch, np.mean(epoch_records['loss']), 
                                   np.mean(epoch_records['mape']), 
                                   np.mean(epoch_records['rmse'])))
            
            model_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "lr_sched": scheduler.state_dict()}
        
            early_stopping(np.mean(losses.item()), 
                        model_dict, epoch, checkpoint_dir)
            
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

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

mask = None
trainer = Trainer(args, config=model_config, model=model, optimizer=optimizer, scheduler=scheduler, loss=criterion)
training_loop(loader, trainer, checkpoint_dir, start_epoch=start_epoch, mask=mask)
