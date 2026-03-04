import os
from types import SimpleNamespace
from osgeo import gdal

class NotebookArgs:
    def __init__(self, **overrides):
        """
        Mimics argparse.Namespace for Jupyter notebooks.
        Pass keyword arguments to override defaults.
        """

        def getenv_cast(key, default, cast=str):
            val = os.getenv(key)
            if val is None:
                return default
            try:
                return cast(val)
            except Exception:
                return default

        # ----------------------------
        # Defaults (same as argparse)
        # ----------------------------

        self.product = overrides.get(
            "product",
            getenv_cast("product", "reflectance_250m", str)
        )

        self.reproj_lib = overrides.get(
            "reproj_lib",
            getenv_cast("REPROJ_LIB", "rioxarray", str)
        )
        if self.reproj_lib not in ["rioxarray", "xesmf"]:
            raise ValueError("reproj_lib must be 'rioxarray' or 'xesmf'")

        self.reproj_method = overrides.get(
            "reproj_method",
            getenv_cast("REPROJ_METHOD", "nearest", str)
        )
        if self.reproj_method not in ["nearest", "bilinear"]:
            raise ValueError("reproj_method must be 'nearest' or 'bilinear'")

        self.delete_temp = overrides.get("delete_temp", False)

        self.lon_min = overrides.get("lon_min", -70.0)
        self.lat_min = overrides.get("lat_min", -70.0)
        self.lon_max = overrides.get("lon_max", 70.0)
        self.lat_max = overrides.get("lat_max", 70.0)

        self.n_lon = overrides.get("n_lon", 7)
        self.n_lat = overrides.get("n_lat", 7)

        self.start_date = overrides.get(
            "start_date",
            getenv_cast("start_date", "2025-07-01", str)
        )

        self.end_date = overrides.get(
            "end_date",
            getenv_cast("end_date", "2025-07-05", str)
        )

        self.batch_days = overrides.get(
            "batch_days",
            getenv_cast("batch_days", 30, int)
        )

        self.store_cloud = overrides.get("store_cloud", False)
        self.majortom_grid = overrides.get("majortom_grid", False)

        self.output_format = overrides.get(
            "output_format",
            getenv_cast("output_format", "zarr", str)
        )
        if self.output_format not in ["tiff", "zarr"]:
            raise ValueError("output_format must be 'tiff' or 'zarr'")

    def __repr__(self):
        return f"NotebookArgs({self.__dict__})"

args = NotebookArgs(product="VIIRS_500m_night_monthly", 
    reproj_lib="rioxarray", 
    reproj_method="nearest", 
    delete_temp=False, 
    batch_days=1, 
    store_cloud=False, 
    majortom_grid=False, 
    output_format="tiff"
)

from pathlib import Path
import shutil
import os
from dotenv import load_dotenv
# from osgeo import gdal
import geopandas as gpd
from datetime import timedelta, datetime

from utils import init_logging, countries_to_bbox
from definitions import ROOT_DIR, DATA_PATH
from eo_data.modis import EarthAccessDownloader   # adjust import if needed

load_dotenv()

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

PRODUCT_KEY = "VIIRS_500m_night_monthly"

CORRUPTED_DATES = [
    "2014-03-01",
    "2014-09-01",
    "2015-05-01",
    "2016-01-01",
    "2016-09-01",
    "2017-05-01",
    "2017-12-01",
    "2018-08-01",
    "2019-04-01",
    "2019-12-01",
    "2020-08-01",
    "2021-04-01",
    "2021-12-01",
    "2022-10-01",
    "2023-08-01",
]

# CORRUPTED_DATES = ["2025-01-01"]

from definitions import DATA_PATH

NEW_OUTPUT_DIR = Path(
    DATA_PATH / "modis/VIIRS_nightlight/new_data"
)

# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

load_dotenv()

# gdal.SetConfigOption("GDAL_CACHEMAX", "512")
# gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

logger = init_logging(log_file="viirs_redownload.log", verbose=False)

products = {
    "VIIRS_500m_night_monthly": {
        "short_name": "VNP46A3",
        "variables": [
            "NearNadir_Composite_Snow_Free",
            "NearNadir_Composite_Snow_Free_Std",
            "NearNadir_Composite_Snow_Free_Quality",
        ],
        "raw_data_type": "h5",
        "resolution": 500,
    }
}

# ------------------------------------------------------------------
# BBOX (same as your original script)
# ------------------------------------------------------------------

gdf = gpd.read_file(DATA_PATH / "shapefiles" / "GAUL_2024.zip")

bbox, polygon = countries_to_bbox(
    ["Brazil", "Argentina", "Peru", "Colombia", "Panama"],
    gdf,
    col_name="gaul0_name",
)

# ------------------------------------------------------------------
# DOWNLOAD LOOP
# ------------------------------------------------------------------

NEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

short_name = products[PRODUCT_KEY]["short_name"]
variables = products[PRODUCT_KEY]["variables"]
raw_data_type = products[PRODUCT_KEY]["raw_data_type"]
resolution = products[PRODUCT_KEY]["resolution"]

for date_str in CORRUPTED_DATES:
    try:
        logger.info(f"Re-downloading {date_str}")

        # date_end = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        downloader = EarthAccessDownloader(args,
            short_name=short_name,
            bbox=bbox,
            variables=variables,
            resolution=resolution,
            date_range=(date_str, date_str),
            collection_name=f"{short_name}_061",
            output_format="tiff",
            raw_data_type=raw_data_type,
            data_dir=NEW_OUTPUT_DIR
        )

        # 🔥 override output directory WITHOUT touching API internals
        downloader.output_dir = NEW_OUTPUT_DIR

        # downloader.cleanup()
        downloader.run(batch_days=1)

        logger.info(f"Finished {date_str}")

    except Exception as e:
        logger.error(f"Failed on {date_str}: {e}")
        continue