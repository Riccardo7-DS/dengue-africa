import os
import cdsapi
import logging
from pathlib import Path
from dotenv import load_dotenv
from definitions import DATA_PATH
from utils import process_gdf, countries_to_bbox
import zipfile
import time

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

class Era5CDS():
    def __init__(self, variables:list, 
                 output_name:str):

        self.api_url = os.environ["CDS_API_URL"]
        self.api_key = os.environ["CDS_API_KEY"]
        self.zip_path = Path(DATA_PATH) / "shapefiles/GAUL_2024.zip"
        self.output_path = Path(DATA_PATH) / "ERA5" 
        os.makedirs(self.output_path, exist_ok=True)

        self.variables = variables
        self._output_name = output_name

    def run(self, countries:list, year_min:int, year_max:int):
        logger.info(f"Starting downloading data for countries {countries} from year {year_min} to {year_max}")
        bbox = self._select_area(countries)
        self.query_era5_cdsapi(
                  year_min=year_min, 
                  year_max=year_max, 
                  area=bbox, 
                  output_path=self.output_path, 
                  variables=self.variables, 
                  output_name=self._output_name)

    def _unzip_files(self, output_path, target_name):
        p = Path(output_path)

        # Ensure target_name becomes a full path
        target_path = p.parent / target_name

        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p, "r") as z:
                # Extract only .nc files if multiple exist inside
                members = [m for m in z.namelist() if m.endswith(".nc")]
                if not members:
                    raise RuntimeError("No .nc file found inside ZIP.")
                internal_nc = members[0]  # usually only one
                z.extract(internal_nc, p.parent)
            # p.unlink()  # remove zip if desired

            extracted_nc = p.parent / internal_nc
        else:
            # If it's not a ZIP, assume output_path is already an .nc file
            extracted_nc = p

        # Rename safely
        extracted_nc.rename(target_path.with_suffix(".nc"))

    def _select_area(self, countries:list):
        gdf = process_gdf(self.zip_path, countries=countries)
        gdf_admin0 = gdf.dissolve(by="adm_0_name", as_index=False)
        bbox, _ = countries_to_bbox(countries, gdf_admin0)
        west, south, east, north = bbox
        udpated_bbox = [north, west, south, east]
        return udpated_bbox

    def query_era5_cdsapi(
            self, 
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

        c = cdsapi.Client(url=self.api_url, key=self.api_key, wait_until_complete=False)

        years = range(year_min, year_max + 1)

        for year in years:
            output_file = f"era5land_{output_name}_{year}"
            output_path_year = str(output_path / output_file)

            
            r = c.retrieve(
                    product_name,
                    {
                        "variable": variables,
                        # "month": [f"{m:02d}" for m in range(1, 13)],
                        # "day":   [f"{d:02d}" for d in range(1, 32)],
                        "date": f"{year}-01-01/{year}-12-31",
                        "time": ["00:00"],
                        "area": area,   #   North, West, South, East
                        "format": format_output,
                    }
                )
            
            # sleep = 30

            # while True:
            #     r.update()
            #     reply = r.reply
            #     r.info("Request ID: %s, state: %s" % (reply["request_id"], reply["state"]))

            #     if reply["state"] == "completed":
            #         break
            #     elif reply["state"] in ("queued", "running"):
            #         r.info("Request ID: %s, sleep: %s", reply["request_id"], sleep)
            #         time.sleep(sleep)
            #     elif reply["state"] in ("failed",):
            #         r.error("Message: %s", reply["error"].get("message"))
            #         r.error("Reason:  %s", reply["error"].get("reason"))
            #         for n in (
            #             reply.get("error", {}).get("context", {}).get("traceback", "").split("\n")
            #         ):
            #             if n.strip() == "":
            #                 break
            #             r.error("  %s", n)
                    
            #         r.error(f"Skipping year {year} because of failure.")
            #         break

            r.download(output_path_year)
            self._unzip_files(output_path_year, output_file)

            logger.info(f"{output_file} downloaded.")

            