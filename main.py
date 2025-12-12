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


from definitions import DATA_PATH
from torchgeo.datasets import RasterDataset, XarrayDataset
from torch.utils.data import DataLoader, Dataset
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets.utils import stack_samples, lazy_import
from pathlib import Path
from utils import latin_box
import torch 
import pandas as pd 
import numpy as np
from datetime import timedelta
# simplefilter("ignore", UserWarning)

viirs_data_path = DATA_PATH / "modis/VNP46A3_061/tiffs/-83.1_-55.1_-28.8_13.4/"
risk_raster_path = DATA_PATH / "riskmaps_public main data" 
era5_path = DATA_PATH / "ERA5"

class VIIRSData(RasterDataset):
    filename_glob = "VNP*.tif"
    filename_regex = r'^VNP46A3_(?P<date>\d{8})\.tif$'
    date_format = "%Y%m%d"
    separate_files = False
    is_image = True
    all_bands = None
    rgb_bands = None


class ERA5Daily(RasterDataset):
    filename_glob = "era5land_latin_america*.nc"
    filename_regex = r"era5land_latin_america_(?P<date>\d{4})\.nc$"
    date_format = "%Y"
    is_image = True
    separate_files = False

    def __init__(self, root, variables=None, transform=None):
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
        
        T_max = 32
        if patch.shape[0] < T_max:
            pad_len = T_max - patch.shape[0]
            pad = torch.zeros((pad_len, *patch.shape[1:]), dtype=patch.dtype)
            patch = torch.cat([pad, patch], dim=0)



        if self.transform:
            patch = self.transform(patch)

        return {"image": patch}
    
    def __len__(self):
        return len(self._aggregated_dataset["time"])

    
class StaticLayer(RasterDataset):
    filename_glob = "DEN_riskmap_wmean_masked.tif"
    is_image = True
    separate_files = False

# Base datasets
era5 = ERA5Daily(era5_path)
risk = StaticLayer(risk_raster_path)
viirs = VIIRSData(viirs_data_path)


class DengueDataset(Dataset):
    """
    Multi-resolution dataset: VIIRS (high), ERA5 (medium), Static maps (coarse)
    """
    def __init__(self, viirs, era5, static, bbox, patch_size=(1024,1024), transform=None):
        self.viirs = viirs
        self.era5 = era5
        self.static = static
        self.bbox = bbox
        self.patch_size = patch_size
        self.transform = transform
        self.num_times = len(viirs)

    def __len__(self):
        return self.num_times

    def __getitem__(self, idx):
        # Time slice
        t_current = viirs.index.iloc[idx].name.left
        if idx + 1 < len(viirs.index):
            t_next = viirs.index.iloc[idx+1].name.left - timedelta(seconds=1)
        else:
            t_next = viirs.index.iloc[idx].name.right

        # Spatial sampler
        sampler = GridGeoSampler(self.viirs, size=self.patch_size, roi=self.bbox, toi=pd.Interval(t_current, t_next))
        queries = list(sampler)
        x_slice, y_slice, _ = queries[0]
        query = (x_slice, y_slice, slice(t_current, t_next))

        # Extract patches
        x_high = self.viirs[query]["image"].float()
        x_med = self.era5[query]["image"].float()
        x_static = self.static[query]["image"].float()

        # Normalize
        x_high = (x_high - x_high.mean()) / (x_high.std() + 1e-6)
        x_med = (x_med - x_med.mean()) / (x_med.std() + 1e-6)
        x_static = (x_static - x_static.mean()) / (x_static.std() + 1e-6)

        if self.transform:
            x_high, x_med, x_static = self.transform(x_high, x_med, x_static)

        y = torch.tensor(0.0)  # placeholder label

        return x_high, x_med, x_static, y


bbox = latin_box()

#bbox = [xmin, ymin, xmax, ymax] in CRS coordinates
patch_size = (1024, 1024)

era5 = ERA5Daily(era5_path)
risk = StaticLayer(risk_raster_path)
viirs = VIIRSData(viirs_data_path)

dataset = DengueDataset(viirs=viirs, era5=era5, static=risk, bbox=bbox, patch_size=patch_size)
loader = DataLoader(dataset, batch_size=4, shuffle=False)


from src.models.transformer import DenguePredictor
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

# -------------------------
# Example batch
# -------------------------
for x_high, x_med, x_static, y in loader:
    print("High-res:", x_high.shape)
    print("Medium-res:", x_med.shape)
    print("Static:", x_static.shape)
    model(x_high, x_med, x_static)
    break

# all_batches = []
# for t_idx in range(len(viirs)):
#     sampler = GridGeoSampler(viirs, size=patch_size, roi=bbox, toi=t_idx)
#     # Determine next time boundary
#     t_current = viirs.index.iloc[t_idx].name.left
#     if t_idx + 1 < len(viirs.index):
#         t_next = viirs.index.iloc[t_idx+1].name.left - timedelta(seconds=1)
#     else:
#         t_next = viirs.index.iloc[t_idx].name.right - timedelta(seconds=1)
    
#     for query in sampler:
#         # Replace the time slice with [t_current, t_next)
#         x_slice, y_slice, _ = query
#         query = (x_slice, y_slice, slice(t_current, t_next))
        
#         viirs_patch = viirs[query]["image"]
#         era5_patch = era5[query]["image"]
#         static_patch = risk[query]["image"]

#         print(
#             "VIIRS:", viirs_patch.shape,
#             "ERA5:", era5_patch.shape,
#             "Static:", static_patch.shape,
#         )

