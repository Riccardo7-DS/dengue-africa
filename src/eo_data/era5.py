import os
import cdsapi
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

#Replace UID:ApiKey with you UID and Api Key

era5_variables = ['total_precipitation', 
                  "temperature_2m", 
                  "skin_temperature", 
                  "soil_temperature_level_1", 
                  "soil_temperature_level_2", 
                  "soil_temperature_level_3",
                  "soil_temperature_level_4",
                  "potential_evaporation",
                  "total_evaporation",
                  "u_component_of_wind_10m",
                  "v_component_of_wind_10m"]


def query_era5_cdsapi(
        year_min: int,
        year_max: int,
        area: list,
        variables: list,
        output_name:str,
        output_path:str,
        product_name: str = "reanalysis-era5-land",
        format_output: str = "netcdf"
    ):
    """
    Download ERA5/ERA5-Land data from CDS API.
    """

    load_dotenv()  # your function to load .env variables

    api_url = os.environ["CDS_API_URL"]
    api_key = os.environ["CDS_API_KEY"]   # should be "uid:apikey"

    c = cdsapi.Client(url=api_url, key=api_key)

    years = range(year_min, year_max + 1)

    for year in years:
        output_file = f"era5land_{output_name}_{year}.nc"
        output_path_year = output_path / output_file

        c.retrieve(
            product_name,
            {
                "variable": variables,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day":   [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00"],
                "area": area,   # [North, West, South, East]
                "format": format_output,
            },
            output_path_year,
        )

        logger.info(f"{output_file} downloaded.")