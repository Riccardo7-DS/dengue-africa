from utils import process_gdf, countries_to_bbox
from definitions import DATA_PATH
from eo_data.era5 import Era5CDS, era5_variables
from pathlib import Path
import os

countries = ["Peru", "Brazil", "Colombia", "Panama", "Argentina"]

year_min = 2000
year_max = 2024

era5_downloader = Era5CDS(era5_variables, output_name="latin_america")
era5_downloader.run(year_min, year_max)

country = ["Taiwan"]
era5_downloader = Era5CDS(era5_variables, output_name="Taiwan")
era5_downloader.run(year_min, year_max)

country = ["Pakistan"]
era5_downloader = Era5CDS(era5_variables, output_name="Pakistan")
era5_downloader.run(year_min, year_max)


