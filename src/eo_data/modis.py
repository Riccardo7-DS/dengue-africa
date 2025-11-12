# import ee
import logging
from pathlib import Path
from osgeo import gdal
from definitions import DATA_PATH
import tempfile
from collections import defaultdict
import gc 
import rioxarray
import os
import re
import glob
import shutil
import tempfile
import pandas as pd
from tqdm import tqdm
import earthaccess
from dotenv import load_dotenv
import numpy as np
import xarray as xr
from typing import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5netcdf


logger = logging.getLogger(__name__)




class EeModis():
    def __init__(self,
                 start_date:str, 
                 end_date:str,
                 name:Literal["ref_061", "LST_061"],
                 output_dir:str,
                 geometry=None,
                 output_resolution:int=1000,
                 download_collection:bool=True):
        
        ee.Authenticate()
        ee.Initialize(project="ee-querying-maps")
        
        valid_names = {"ref_061","LST_061"}
        assert name in valid_names, \
            "Invalid value for 'name'. It should be one of: 'ref_061', 'LST_061'"

        import ee
        import geemap
        ee.Initialize()

        self.start_date = start_date
        self.end_date = end_date
        self.out_dir = output_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.polygon = ee.Geometry.Rectangle(geometry) if type(geometry) is list else geometry
        self.output_resolution = output_resolution
        self.product = self._get_product_name(name)
        self.name = name
    
        # Import images
        images = ee.ImageCollection(self.product)\
                    .filterDate(start_date, end_date)\
                    .filterBounds(self.polygon)
        
        bands = self._get_bands(self.product)
        img_bands = images.select(bands)

        if name == "ref_061":
            img_bands = img_bands.map(lambda x: self._compute_ndvi(x, bands[1], bands[0],
                                                                   bands[2]))
        if name == "LST_061":
            img_bands = img_bands.map(self._scale_lst)

        if download_collection:
            self._collection_prepr_download(img_bands)
        else:
            self._image_prepr_download(img_bands)

    # -------- LST helpers -------- #
    def _scale_lst(self, img):
        """Convert LST from MOD11A1 to Celsius and mask by QA."""
        lst_day = img.select("LST_Day_1km").multiply(0.02).subtract(273.15)
        lst_night = img.select("LST_Night_1km").multiply(0.02).subtract(273.15)
        qa_day = img.select("QC_Day")
        qa_night = img.select("QC_Night")

        # simple QA mask (only keep qa==0)
        mask_day = qa_day.eq(0)
        mask_night = qa_night.eq(0)

        lst_day = lst_day.updateMask(mask_day).rename("LST_Day_C")
        lst_night = lst_night.updateMask(mask_night).rename("LST_Night_C")

        return img.addBands([lst_day, lst_night])

    def _get_product_name(self, name):
        if name == "ref_061":
            return "MODIS/061/MOD09GQ"
        elif name == "LST_061":
            return "MODIS/061/MOD11A1"
        else:
            raise NotImplementedError(f"Product {name} not implemented")

    def _get_bands(self, product_name):
        if product_name == "MODIS/061/MOD09GQ":
            return ["sur_refl_b01", "sur_refl_b02","QC_250m"]
        elif product_name == "MODIS/061/MOD11A1":
            return ["LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"]
        else:
            raise NotImplementedError(f"Product {product_name} not implemented")

    def _extract_qa(self, img, qaBand):
        qa_mask = img.select(qaBand)
        quality_mask = self._bitwise_extract(qa_mask, 0, 1).eq(0)
        return quality_mask

    def _bitwise_extract(self, input_image, from_bit, to_bit):
        import ee
        mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
        mask = ee.Number(1).leftShift(mask_size).subtract(1)
        return input_image.rightShift(from_bit).bitwiseAnd(mask)

    def _compute_ndvi(self, img, nirBand, redBand, qaBand=None):
        ndvi = img.normalizedDifference([nirBand, redBand])
        if qaBand is not None:
            qa = self._extract_qa(img, qaBand)
            return ndvi.updateMask(qa)
        else:
            return ndvi

    def _imreproj(self, image):
        return image.reproject(crs='EPSG:4326', scale=self.output_resolution)

    def ee_hoa_geometry(self):
        logger.info("Using default geometry for horn of Africa")
        import ee
        polygon = ee.Geometry.Polygon(
            [[[30.288233396779802,-5.949173816626356],
            [51.9972177717798,-5.949173816626356],
            [51.9972177717798,15.808293611760663],
            [30.288233396779802,15.808293611760663]]]
        )
        return  ee.Geometry(polygon, None, False)
        
    def _collection_prepr_download(self, images):
        import geemap
        clipped_img = images.map(lambda image: image.clip(self.polygon))
        reprojected_img = clipped_img.map(self._imreproj)
        reprojected_img.aggregate_array("system:index").getInfo()
        geemap.ee_export_image_collection(reprojected_img, self.out_dir)        



    def _split_roi(self, roi, nx=2, ny=2):
        """Split an ee.Geometry.Rectangle into nx*ny tiles."""
        coords = roi.coordinates().get(0).getInfo()
        xmin, ymin = coords[0][0], coords[0][1]
        xmax, ymax = coords[2][0], coords[2][1]
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
    
        rois = []
        for i in range(nx):
            for j in range(ny):
                x0, x1 = xmin + i*dx, xmin + (i+1)*dx
                y0, y1 = ymin + j*dy, ymin + (j+1)*dy
                rois.append(ee.Geometry.Rectangle([x0, y0, x1, y1]))
        return rois
    
    def _image_prepr_download(self, images, nx=2, ny=2):
        """Export each image in the collection, automatically splitting large ROIs."""
        import ee
        import logging
        logger = logging.getLogger(__name__)
    
        # Split ROI
        tiles = self._split_roi(self.polygon, nx=nx, ny=ny)
    
        # Get image IDs
        image_ids = images.aggregate_array("system:index").getInfo()
    
        for idx, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            img = ee.Image(image_id)
    
            for t_idx, tile in enumerate(tiles):
                img_tile = img.clip(tile)
                proj = img_tile.projection().getInfo()
                filename = f"{image_id.split('/')[-1]}_tile{t_idx}"
    
                task = ee.batch.Export.image.toDrive(
                    image=img_tile,
                    description=filename,
                    folder="MODIS_EXPORT",
                    region=tile.bounds(),
                    crs=proj["crs"],
                    crsTransform=proj["transform"],
                    maxPixels=1e13,
                    fileFormat="GeoTIFF"
                )
                task.start()
                logger.info(f"Exporting {filename}...")
    
    def _preprocess_file(self, ds:xr.Dataset):
        import pandas as pd
        time = ds.encoding['source'].split("/")[-1].replace("_","/")[:-4]
        date_xr = pd.to_datetime(time)
        ds = ds.assign_coords(time=date_xr)
        ds = ds.expand_dims(dim="time")
        return ds.isel(band=0)

    def xarray_preprocess(self):
        import xarray as xr
        import os
        files = [os.path.join(self.out_dir,f) for f in os.listdir(self.out_dir) if f.endswith(".tif")]
        dataset = xr.open_mfdataset(files, 
                                    preprocess=self._preprocess_file, 
                                    engine="rasterio")
        return dataset
    

# class ProcessModis:
#     """Process MODIS HDF files into mosaicked and reprojected NetCDF files."""

#     def __init__(self, 
#             folder_path: str, 
#             mosaics_folder: str):
        
#         self.folder_path = Path(folder_path)
#         self.modis_folder = Path(DATA_PATH) / "modis"
#         self.mosaics_folder = Path(mosaics_folder)
#         self.mosaics_folder.mkdir(parents=True, exist_ok=True)

#     @staticmethod
#     def _get_date(filename: str) -> datetime:
#         """Extract date from MODIS filename (assuming format MOD11A1.AYYYYDDD...)."""
#         try:
#             doy_str = filename.split(".")[1][1:]  # e.g. 'A2021205'
#             year, doy = int(doy_str[:4]), int(doy_str[4:])
#             return datetime(year, 1, 1) + timedelta(days=doy - 1)
#         except Exception as e:
#             raise ValueError(f"Cannot parse date from filename: {filename}") from e

#     def _group_files_by_date(self) -> dict:
#         """Group all .hdf files by date."""
#         files_by_date = defaultdict(list)
#         for file in self.folder_path.glob("*.hdf"):
#             date = self._get_date(file.name)
#             files_by_date[date].append(file)
#         return files_by_date

#     def process_all(self):
#         """Run mosaicking and processing for all available dates."""
#         files_by_date = self._group_files_by_date()
#         dates = sorted(files_by_date.keys())
#         self._process_dates(files_by_date, dates)

#     def _process_dates(self, files_by_date: dict, dates: list[datetime]):
#         """Process mosaics for each date."""
#         template_transform = None
#         template_shape = None

#         for i, date in enumerate(tqdm(dates, desc="Processing MODIS mosaics")):
#             tiles = files_by_date[date]

#             # Build list of MODIS subdatasets
#             subdatasets = [
#                 f'HDF4_EOS:EOS_GRID:"{f.resolve()}":MODIS_Grid_8Day_1km_LST:LST_Day_1km'
#                 for f in tiles
#             ]

#             # Temporary VRT path
#             with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as tmp_vrt:
#                 vrt_path = tmp_vrt.name

#             tif_path = self.mosaics_folder / f"{date.strftime('%Y%m%d')}.tif"

#             # Build and translate VRT → GeoTIFF
#             gdal.BuildVRT(vrt_path, subdatasets)
#             gdal.Translate(str(tif_path), vrt_path)
#             os.remove(vrt_path)  # clean temp file

#             # Load as xarray and process
#             da = (
#                 xr.open_dataset(tif_path, engine="rasterio", chunks={"x": 1000, "y": 1000})["band_data"]
#                 .rio.write_crs("EPSG:6933")
#             )
#             da = da.where(da != 0) * 0.02 - 273.15  # Convert to Celsius
#             da = da.rename("LST_Day_Celsius").expand_dims(time=[date]).isel(band=0)

#             # Reproject to EPSG:4326
#             if i == 0:
#                 da_latlon = da.rio.reproject("EPSG:4326")
#                 template_transform = da_latlon.rio.transform()
#                 template_shape = da_latlon.shape[-2:]
#             else:
#                 da_latlon = da.rio.reproject(
#                     "EPSG:4326",
#                     transform=template_transform,
#                     shape=template_shape,
#                 )

#             da_latlon = da_latlon.astype(np.float32)

#             # Export to NetCDF
#             out_nc = self.mosaics_folder / f"{date.strftime('%Y%m%d')}.nc"
#             da_latlon.to_netcdf(out_nc)
#             da.close()

#         logger.info(f"\n✅ Processing complete. Output saved to {self.mosaics_folder}")






class EarthAccessDownloader:
    """
    Generic downloader for NASA Earthdata (via earthaccess).
    Handles: authentication, search, download, mosaic, cleanup, and export to Zarr.
    """

    def __init__(
        self,
        short_name="MOD11A1",
        variables="MODIS_Grid_Daily_1km_LST:LST_Day_1km",
        date_range=None,
        bbox=None,
        data_dir=Path(DATA_PATH) / "modis",
        collection_name="lst_061",
        zarr_path=None,
    ):
        
        self.date_range = self._check_dates(date_range)

        self.short_name = short_name
        self.variable = variables
        self.bbox = bbox
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.granule_dir = self.data_dir / self.collection_name
        self.zarr_path = zarr_path or (self.data_dir / f"{self.short_name}_dataset.zarr")

        self.granule_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        self._login_earthaccess()

    def _login_earthaccess(self):
        load_dotenv()
        # Login once
        earthaccess.login(strategy="environment")

    def _check_dates(self, date_range):
        if date_range is None or len(date_range) != 2:
            raise ValueError("date_range must be a tuple of (start_date, end_date)")
        return date_range

    # ------------------------
    # Search and Download
    # ------------------------
    def _search_and_download(self):
        logger.info(f"🔍 Searching for {self.short_name} data...")
        results = earthaccess.search_data(
            short_name=self.short_name,
            bounding_box=(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]),
            temporal=self.date_range,
        )

        logger.info(f"⬇️ Downloading {len(results)} files to {self.granule_dir}")
        earthaccess.download(results, str(self.granule_dir))
        self.files = sorted(glob.glob(str(self.granule_dir / f"{self.short_name}*.hdf")))


    def _custom_preprocess(self, da):
        # ---------- PRODUCT-SPECIFIC TRANSFORMS ----------
        if self.short_name == "MOD11A1":
            # LST products: scale factor and conversion to Celsius
            da = da.where(da != 0) * 0.02 - 273.15
            da.name = "LST_Day_Celsius"
        elif "MOD13" in self.short_name:
            # NDVI products (MOD13Q1, MYD13A2, etc.)
            da = da.where(da != -3000) * 0.0001  # NDVI scale factor
            da.name = "NDVI"
        else:
            da.name = "band_data"
        # -------------------------------------------------

        return da

    def _get_date(self, filename):
        match = re.search(r'\.A(\d{7})\.', filename)
        return pd.to_datetime(match.group(1), format="%Y%j")

    # ------------------------
    # Mosaic and Convert
    # ------------------------
    # def _mosaic_daily(self):
    #     logger.info("🧩 Grouping and mosaicking tiles by date...")

    #     files_by_date = defaultdict(list)
    #     for f in self.files:
    #         files_by_date[self._get_date(f)].append(f)

    #     mosaics = []

    #     if isinstance(self.variable, str):
    #         variables = [self.variable]
    #     else:
    #         variables = self.variable

    #     for date, files in tqdm(sorted(files_by_date.items()), desc="Processing dates"):
    #         variable_data = {}

    #         for var in variables:
    #             subdatasets = []

    #             # --- Collect subdataset paths for this variable across all tiles ---
    #             for f in files:
    #                 ds = gdal.Open(str(f))
    #                 if ds is None:
    #                     logger.warning(f"⚠️ Could not open {f}, skipping.")
    #                     continue

    #                 # # Match the correct subdataset by variable name
    #                 matched = [
    #                     s[0] for s in ds.GetSubDatasets()
    #                     if var in s[0]  # e.g. 'sur_refl_b01'
    #                 ]
    #                 if not matched:
    #                     logger.warning(f"⚠️ Variable '{var}' not found in {f}")
    #                     continue

    #                 subdatasets.append(matched[0])  # one per tile

    #             if not subdatasets:
    #                 logger.warning(f"⚠️ No subdatasets found for variable '{var}' on {date:%Y-%m-%d}")
    #                 continue

    #             # --- Prepare output paths ---
    #             self.temp_dir.mkdir(parents=True, exist_ok=True)
    #             vrt_path = self.temp_dir / f"{self.short_name}_{var}_{date:%Y%m%d}.vrt"
    #             tif_path = self.temp_dir / f"{self.short_name}_{var}_{date:%Y%m%d}.tif"

    #             # --- Build the VRT mosaic for this variable ---
    #             try:
    #                 gdal.UseExceptions()
    #                 vrt = gdal.BuildVRT(str(vrt_path), subdatasets)
    #             except RuntimeError as e:
    #                 raise RuntimeError(f"❌ GDAL failed to build VRT for {var} on {date:%Y-%m-%d}: {e}")

    #             if vrt is None:  #or not vrt_path.exists():
    #                 raise RuntimeError(f"❌ GDAL failed to create VRT for {var} on {date:%Y-%m-%d}")
    #             vrt = None  # close handle

    #             # --- Translate to GeoTIFF ---
    #             gdal.Translate(str(tif_path), str(vrt_path))
    #             if not tif_path.exists():
    #                 raise RuntimeError(f"❌ GDAL failed to create GeoTIFF for {var} on {date:%Y-%m-%d}")

    #             # --- Open, reproject, and preprocess ---
    #             da = xr.open_dataset(tif_path, engine="rasterio")["band_data"]
    #             da = da.rio.write_crs("EPSG:6933")
    #             da = self._custom_preprocess(da)
    #             da = da.rio.reproject("EPSG:4326")
    #             da = da.expand_dims(time=[date])
    #             da.name = var

    #             variable_data[var] = da

    #         if variable_data:
    #             ds_date = xr.merge(variable_data.values())
    #             mosaics.append(ds_date)

    #     if not mosaics:
    #         raise RuntimeError("❌ No mosaics were generated. Check variable names or input files.")

    #     ds = xr.concat(mosaics, dim="time", join="override")
    #     return ds



    def _mosaic_daily(self, max_workers=1):
        logger.info("🧩 Grouping and mosaicking tiles by date...")

        # --- Group files by date ---
        files_by_date = defaultdict(list)
        for f in self.files:
            files_by_date[self._get_date(f)].append(f)

        variables = [self.variable] if isinstance(self.variable, str) else list(self.variable)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        def process_one_day(date_files):
            """Process a single date group (can run in parallel)."""
            date, files = date_files
            variable_data = {}

            # --- Cache subdatasets for all tiles once ---
            subdatasets_cache = {}
            for f in files:
                ds = gdal.Open(str(f))
                if ds is None:
                    logger.warning(f"⚠️ Could not open {f}, skipping.")
                    continue
                subdatasets_cache[f] = ds.GetSubDatasets()

            # --- Process each variable ---
            for var in variables:
                subdatasets = [
                    s[0]
                    for f in files
                    for s in subdatasets_cache.get(f, [])
                    if var in s[0]
                ]
                if not subdatasets:
                    logger.warning(f"⚠️ No subdatasets found for '{var}' on {date:%Y-%m-%d}")
                    continue

                # --- Output paths ---
                vrt_path = self.temp_dir / f"{self.short_name}_{var}_{date:%Y%m%d}.vrt"
                tif_path = self.temp_dir / f"{self.short_name}_{var}_{date:%Y%m%d}.tif"

                # --- Build mosaic only (no reprojection) ---
                try:
                    gdal.UseExceptions()
                    vrt = gdal.BuildVRT(str(vrt_path), subdatasets)
                    if vrt is None:
                        raise RuntimeError("BuildVRT returned None")

                    # Write GeoTIFF in original CRS (MODIS sinusoidal)
                    gdal.Translate(
                        str(tif_path),
                        vrt,
                        creationOptions=["TILED=YES", "BIGTIFF=YES", "COMPRESS=LZW"]
                    )
                    vrt = None
                except RuntimeError as e:
                    raise RuntimeError(f"❌ GDAL failed for {var} on {date:%Y-%m-%d}: {e}")

                if not tif_path.exists():
                    raise RuntimeError(f"❌ GeoTIFF not created for {var} on {date:%Y-%m-%d}")

                # --- Open lazily with rioxarray ---
                da = rioxarray.open_rasterio(tif_path, chunks=True).squeeze("band", drop=True)
                da = da.rio.write_crs("EPSG:6933")  # MODIS sinusoidal

                # --- MOD09 scaling and masking ---
                # da = da.where((da > 0) & (da <= 10000)) / 10000.0

                # --- Custom preprocessing ---
                da = self._custom_preprocess(da)

                # --- Lazy reprojection (rioxarray) ---
                da = da.rio.reproject(
                    dst_crs="EPSG:4326",   
                    resampling="nearest",
                    num_threads=1,
                    transform=None,
                    shape=None,
                ).chunk({"x": 1024, "y": 1024})

                # --- Add time dimension ---
                da = da.expand_dims(time=[date])
                da.name = var

                variable_data[var] = da

                del da
                gc.collect()

            # --- Merge variables for this date ---
            if variable_data:
                ds_date = xr.merge(variable_data.values())
                return ds_date
            return None

        # --- Parallel or sequential execution ---
        if max_workers > 1:
            mosaics = []
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process_one_day, item): item[0] for item in files_by_date.items()}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing dates"):
                    result = fut.result()
                    if result is not None:
                        mosaics.append(result)
        else:
            mosaics = []
            for item in tqdm(sorted(files_by_date.items()), desc="Processing dates"):
                ds_date = process_one_day(item)
                if ds_date is not None:
                    # mosaics.append(result)
                    path = self.granule_dir / f"{self.short_name}_{item[0]:%Y%m%d}.nc"
                    ds_date.to_netcdf(path, engine="h5netcdf")
                    del ds_date
                    gc.collect()

            
            ds = xr.open_mfdataset(str(path / "*.nc"), combine="by_coords", parallel=True, chunks="auto")


        if not mosaics:
            raise RuntimeError("❌ No mosaics were generated. Check variable names or input files.")

        # --- Concatenate all dates ---
        ds = xr.combine_by_coords(mosaics, combine_attrs="override")

        logger.info("✅ Daily mosaics successfully completed.")
        return ds


    # ------------------------
    # Export to Zarr
    # ------------------------
    def _export_to_zarr(self, ds):
        logger.info(f"💾 Appending data to {self.zarr_path}")
        if not self.zarr_path.exists():
            ds.to_zarr(self.zarr_path, mode="w")
        else:
            ds.to_zarr(self.zarr_path, mode="a", append_dim="time")

    # ------------------------
    # Cleanup
    # ------------------------
    def cleanup(self):
        logger.info(f"🧹 Cleaning up temporary files in {self.temp_dir}")
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------
    # Full pipeline
    # ------------------------
    def run(self):
        self._search_and_download()
        ds = self._mosaic_daily()
        self._export_to_zarr(ds)
        self.cleanup()
        logger.info("✅ Done!")


if __name__ == "__main__":
    from utils import latin_box, init_logging

    products = {
        "LST": {
            "short_name": "MOD11A1",
            "variables": ["MODIS_Grid_Daily_1km_LST:LST_Day_1km"]
        },
        "reflectance_250m": {
            "short_name": "MOD09GQ",
            # "variables": [
            #     "MODIS_Grid_250m_2D:sur_refl_b01",
            #     "MODIS_Grid_250m_2D:sur_refl_b02"
            # ],
            "variables": ["sur_refl_b01", 
                           "sur_refl_b02", 
                           "QC_250m", 
                           "obscov"
            ]
        }
    }

    product = "reflectance_250m"

    variables = products[product]["variables"]
    short_name = products[product]["short_name"]

    logger = init_logging(log_file="modis_downloader.log", verbose=False)
    bbox = latin_box(True)
    
    downloader = EarthAccessDownloader(
        short_name=short_name,
        bbox= bbox,
        variables=variables,
        date_range=("2020-12-27", "2020-12-31"),
    )

    downloader.cleanup()
    downloader.run()