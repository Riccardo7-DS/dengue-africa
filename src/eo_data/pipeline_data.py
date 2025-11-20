from eo_data.modis import EarthAccessDownloader
from utils import latin_box, generate_bboxes_fixed, init_logging
from osgeo import gdal
from definitions import ROOT_DIR
from pathlib import Path
from dotenv import load_dotenv
import argparse
import os
import shutil

# Load environment variables
load_dotenv(Path(ROOT_DIR)/ ".env")

# GDAL tuning
gdal.SetConfigOption('GDAL_CACHEMAX', '512')    # MB, tune to memory
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')

import sys
sys.dont_write_bytecode = True

# Argument parsing
parser = argparse.ArgumentParser(description="MODIS Downloader")
parser.add_argument('--product', type=str, default='reflectance_250m', help='MODIS product to download')
parser.add_argument('--reproj_lib', choices=['rioxarray', 'xesmf'], default=os.getenv('REPROJ_LIB', 'rioxarray'), help='Reprojection library to use')
parser.add_argument('--reproj_method', choices=['nearest', 'bilinear'], default=os.getenv('REPROJ_METHOD', 'nearest'), help='Reprojection method to use')    
parser.add_argument('-d', '--delete_temp', action='store_true', default=False, help='Delete temporary files after processing')
parser.add_argument('--lon_min', type=float, default=-70)
parser.add_argument('--lat_min', type=float, default=-70)
parser.add_argument('--lon_max', type=float, default=70)
parser.add_argument('--lat_max', type=float, default=70)
parser.add_argument('--n_lon', type=int, default=5)
parser.add_argument('--n_lat', type=int, default=5)
parser.add_argument('--start_date', type=str, default="2025-07-01")
parser.add_argument('--end_date', type=str, default="2025-07-05")
args = parser.parse_args()

# Define products
products = {
    "LST": {
        "short_name": "MOD11A1",
        "variables": ["MODIS_Grid_Daily_1km_LST:LST_Day_1km"]
    },
    "reflectance_250m": {
        "short_name": "MOD09GQ",
        "variables": ["sur_refl_b01", "sur_refl_b02", "QC_250m"]
    },
    "NDVI_1km_monthly": {
        "short_name": "MOD13A3",
        "variables": ["NDVI", "EVI", "SummaryQA"]
    }
}

variables = products[args.product]["variables"]
short_name = products[args.product]["short_name"]

# Initialize logger
logger = init_logging(log_file="modis_downloader.log", verbose=False)

# Generate bounding boxes
bboxes = generate_bboxes_fixed(
    lon_min=args.lon_min,
    lon_max=args.lon_max,
     lat_min=args.lat_min, 
    lat_max=args.lat_max,
    n_lon=args.n_lon, n_lat=args.n_lat
)

# Loop over each bounding box and download
for i, bbox in enumerate(bboxes):
    logger.info(f"Processing tile {i+1}/{len(bboxes)}: {bbox}")
    try:
        downloader = EarthAccessDownloader(
            short_name=short_name,
            bbox=bbox,
            variables=variables,
            date_range=(args.start_date, args.end_date),
            collection_name=f"{short_name}_061",
            reproj_lib=args.reproj_lib,
            reproj_method=args.reproj_method,
            output_format="tiff"
        )

        downloader.run()

        if args.delete_temp and downloader.temp_dir.exists():
            logger.warning("Deleting temporary directory as per user request.")
            shutil.rmtree(downloader.temp_dir)

        downloader.cleanup()

    except Exception as e:
        logger.error(f"Error processing tile {i+1}: {e}")
        downloader.cleanup()