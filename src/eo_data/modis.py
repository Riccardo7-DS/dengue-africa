# import ee
import logging
from pathlib import Path
from osgeo import gdal
from definitions import DATA_PATH
import tempfile
from collections import defaultdict
import gc 
import rasterio
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
from utils import prepare, minio_client, setup_minio_config, extract_object_from_minio
from pyproj import Transformer
import ee, geemap 
from datetime import datetime, timedelta
from pystac_client import Client
from odc.stac import load
import pyproj
from majortom import Grid
import shapely.geometry
import pickle

logger = logging.getLogger(__name__)

class EeModis():
    def __init__(self,
                 start_date:str, 
                 end_date:str,
                 name:Literal["ref_250m_061","NDVI_1km_monthly_061", "LST_061"],
                 output_dir:str,
                 geometry=None,
                 output_resolution:int=1000,
                 crs='EPSG:4326',
                 format:Literal["GeoTIFF","COG"]="GeoTIFF"):
        
        
        ee.Authenticate()
        ee.Initialize(project=os.environ["EE_PROJECT"])
        
        valid_names = {"ref_250m_061","LST_061", "NDVI_1km_monthly_061"}
        assert name in valid_names, \
            "Invalid value for 'name'. It should be one of: 'ref_250m_061', 'LST_061', 'NDVI_1km_monthly_061'"

        ee.Initialize(project=os.environ.get("EE_PROJECT"))

        self.start_date = start_date
        self.end_date = end_date
        self.out_dir = output_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.polygon = ee.Geometry.Rectangle(geometry) if type(geometry) is list else geometry
        self.output_resolution = None if output_resolution is None else output_resolution
        self.product = self._get_product_name(name)
        self.name = name
        self._format = format
        self._crs = crs


    def run(self, 
            download_collection:bool=False,
            compute_ndvi:bool=False):
        
        """
        download_collection: if True, download the entire image collection,
                                otherwise download each image separately.
        """
        import ee
    
        # Import images
        images = ee.ImageCollection(self.product)\
                    .filterDate(self.start_date, self.end_date)\
                    .filterBounds(self.polygon)
        
        self.bands = self._get_bands(self.product)
        img_bands = images.select(self.bands)

        img_bands = self._preprocess(img_bands, compute_ndvi=compute_ndvi)
        
        if download_collection:
            self._collection_prepr_download(img_bands)
        else:
            self._export_images_to_local(img_bands, 
                self.out_dir, 
                scale=self.output_resolution)

    def _preprocess(self, img_bands, compute_ndvi:bool=False):

        if self.name == "ref_250m_061" and compute_ndvi:
            img_bands = img_bands.map(lambda x: self._compute_ndvi(x, self.bands[1], self.bands[0],
                                                                   self.bands[2]))
        if self.name == "LST_061":
            img_bands = img_bands.map(self._scale_lst)

        images_preprocessed = img_bands.map(lambda image: image.clip(self.polygon))

        if self.output_resolution is not None:
            images_preprocessed = images_preprocessed.map(self._imreproj)
        #     images_preprocessed.aggregate_array("system:index").getInfo()

        return images_preprocessed



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
        # Example for 500m MODIS pixel
        # meters_per_degree = 111_319  # approximate at equator
        # scale_deg = resolution / meters_per_degree  # ~0.0045°
        return image.resample("bilinear")

    def ee_hoa_geometry(self):
        logger.info("Using default geometry for horn of Africa")

        polygon = ee.Geometry.Polygon(
            [[[30.288233396779802,-5.949173816626356],
            [51.9972177717798,-5.949173816626356],
            [51.9972177717798,15.808293611760663],
            [30.288233396779802,15.808293611760663]]]
        )
        return  ee.Geometry(polygon, None, False)
        
    def _collection_prepr_download(self, images):
        geemap.ee_export_image_collection(images, self.out_dir, self._format, crs=self._crs)        


    """Export each MODIS image locally as a Cloud Optimized GeoTIFF."""
    def _export_images_to_local(self, images, output_dir, scale=None):

        image_list = images.toList(images.size())
        n = images.size().getInfo()

        for i in range(n):
            image = ee.Image(image_list.get(i))
            image_id = image.get("system:index").getInfo()
            filename = Path(output_dir) / f"{image_id}.tif"

            logger.debug(f"Exporting {image_id} → {filename}")
            geemap.ee_export_image(
                image,
                filename=str(filename),
                scale=scale,
                region=image.geometry(),
                file_per_band=False,
                format=self._format,
            )

    def _split_roi(self, roi, nx=2, ny=2):
        """Split an ee.Geometry.Rectangle into nx*ny tiles."""
        import ee
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
    

class EarthAccessDownloader:
    """
    Generic downloader for NASA Earthdata (via earthaccess).
    Handles: authentication, search, download, mosaic, cleanup, and export to Zarr.
    """

    def __init__(
        self,
        args,
        short_name="MOD11A1",
        variables="MODIS_Grid_Daily_1km_LST:LST_Day_1km",
        resolution=None,
        date_range=None,
        bbox=None,
        data_dir=None,
        collection_name="lst_061",
        crs="EPSG:6933",
        zarr_path=None,
        output_format:Literal["tiff","zarr"]="zarr",
        raw_data_type=".hdf"
    ):  
        
        if resolution is None:
            raise ValueError("resolution must be provided.")
        else:
            self._resolution = resolution
        
        self.date_range = self._check_dates(date_range)

        self.short_name = short_name
        self.variable = variables
        self.bbox = bbox
        self.collection_name = collection_name
        
        self._check_cloud_or_local(data_dir, args.store_cloud)
        
        self._crs = crs
        if self.bbox:
            self._crs = self._get_utm_crs()
        self._reproj_lib = self._check_reproj_lib(args.reproj_lib)
        self._reproj_method = self._check_reproj_method(args.reproj_method)
        
        self.raw_data_type = raw_data_type
        self._output_format = output_format

        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Initialize MinIO configuration if cloud storage is enabled
        if self._store_cloud and os.getenv('AWS_ACCESS_KEY_ID') is None:
            setup_minio_config()
        
        self._login_earthaccess()

    def _check_cloud_or_local(self, data_dir, store_cloud, zarr_path=None):
        """
        Set up data directories for local or cloud storage.

        Parameters
        ----------
        data_dir : str | Path | None
            Local path or /vsis3/ path for cloud storage.s
        store_cloud : bool
            True for cloud storage, False for local filesystem.
        """
        self._store_cloud = store_cloud

        # Set data_dir
        if store_cloud:
            if not data_dir:
                raise ValueError("For cloud storage, data_dir must be provided as a /vsis3/ path.")
            self.data_dir = str(data_dir)  # keep as string for GDAL
        else:
            self.data_dir = Path(data_dir) if data_dir else Path(DATA_PATH) / "modis"

        logger.info(f"Data directory set to: {self.data_dir} with cloud option {self._store_cloud}")

        # Build granule_dir
        collection_path = (
            f"{self.data_dir}/{self.collection_name}" if store_cloud else self.data_dir / self.collection_name
        )

        if self.bbox:
            bbox_str = "_".join(format(x, ".1f") for x in self.bbox)
            collection_path = f"{collection_path}/{bbox_str}" if store_cloud else collection_path / bbox_str

        self.granule_dir = f"{collection_path}/raw_data" if store_cloud else collection_path / "raw_data"

        if not store_cloud:
            self.granule_dir.mkdir(parents=True, exist_ok=True) 

        if zarr_path and not store_cloud:
            self.zarr_path = zarr_path
        elif not store_cloud:
            self.zarr_path = self.data_dir / f"{self.short_name}_dataset.zarr"
        elif store_cloud and zarr_path:
            raise NotImplementedError("Zarr output not implemented for cloud storage.")
        
        self._minio_client = minio_client() if store_cloud else None
        
    def _check_reproj_lib(self, reproj_lib):
        valid_libs = ["rioxarray", "xesmf"]
        if reproj_lib not in valid_libs:
            raise ValueError(f"Invalid reproj_lib '{reproj_lib}'. Must be one of {valid_libs}.")
        logger.info(f"Using reproj_lib: {reproj_lib}")
        return reproj_lib
    
    def _check_reproj_method(self, reproj_method):
        valid_methods = ["nearest", "bilinear"]
        if reproj_method not in valid_methods:
            raise ValueError(f"Invalid reproj_method '{reproj_method}'. Must be one of {valid_methods}.")
        logger.info(f"Using reproj_method: {reproj_method} with {self._crs}")
        return reproj_method

    def _get_utm_crs(self):
        if not self.bbox:
            return "EPSG:6933"  # fallback
        # bbox is [minx, miny, maxx, maxy] in EPSG:4326
        lon = (self.bbox[0] + self.bbox[2]) / 2
        lat = (self.bbox[1] + self.bbox[3]) / 2
        zone = int((lon + 180) / 6) + 1
        if lat >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone
        return f"EPSG:{epsg}"

    def _login_earthaccess(self):
        load_dotenv()
        # Login once
        earthaccess.login(strategy="environment")

    def _check_dates(self, date_range):
        if date_range is None or len(date_range) != 2:
            raise ValueError("date_range must be a tuple of (start_date, end_date)")
        return date_range
    

    def _missing_date_ranges(self, start_date, end_date, processed_dates):
        """
        Returns a list of (start, end) date tuples (YYYY-MM-DD)
        representing contiguous missing date intervals.
        """

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        processed = set(
            datetime.strptime(d, "%Y-%m-%d") for d in processed_dates
        )

        all_dates = []
        d = start
        while d <= end:
            if d not in processed:
                all_dates.append(d)
            d += timedelta(days=1)

        if not all_dates:
            return []

        ranges = []
        range_start = all_dates[0]
        prev = all_dates[0]

        for d in all_dates[1:]:
            if d == prev + timedelta(days=1):
                prev = d
            else:
                ranges.append((range_start, prev))
                range_start = d
                prev = d

        ranges.append((range_start, prev))

        return [
            (r[0].strftime("%Y-%m-%d"), r[1].strftime("%Y-%m-%d"))
            for r in ranges
        ]

    # ------------------------
    # Search and Download
    # ------------------------
    def _search_and_download(self, date_range=None):
        """
        Search and download data for missing dates within a given date range.
        """
    
        if date_range is None:
            date_range = self.date_range
    
        start_date, end_date = date_range
    
        processed_dates = self._get_processed_tiffs()

        done_dates_range = [d for d in date_range if d in processed_dates]
        
        if len(done_dates_range) > 0:
            logger.info(f"Skipping {len(done_dates_range)} already processed dates.")
    
        missing_ranges = self._missing_date_ranges(
            start_date, end_date, processed_dates
        )
    
        if not missing_ranges:
            logger.info("✅ No missing dates to download.")
            self.files = None
            return
    
        for sub_range in missing_ranges:
            logger.info(
                f"🔍 Searching for {self.short_name} data from {sub_range[0]} to {sub_range[1]}..."
            )
    
            results = earthaccess.search_data(
                short_name=self.short_name,
                bounding_box=(
                    self.bbox[0], self.bbox[1],
                    self.bbox[2], self.bbox[3]
                ),
                temporal=sub_range,
            )
    
            logger.info(f"⬇️ Downloading {len(results)} files to {self.granule_dir}")
    
            if results:
                if self._store_cloud:
                    earthaccess.download(results, self.temp_dir)
                else:
                    earthaccess.download(results, str(self.granule_dir))

        # After download, refresh file list
        if not self._store_cloud:
            self.files = sorted(glob.glob(str(self.granule_dir / f"{self.short_name}*.{self.raw_data_type}")))
        else:
            self.files = sorted(glob.glob(str(self.temp_dir / f"{self.short_name}*.{self.raw_data_type}")))


    def _custom_preprocess(self, da):

        # ---------- PRODUCT-SPECIFIC TRANSFORMS ----------
        if self.short_name == "MOD11A1":
            # LST products: scale factor and conversion to Celsius
            da = da.where(da > 0).astype("float32") * 0.02 - 273.15  # LST scale factor
            da.name = "LST_Day_Celsius"
        elif "MOD13" in self.short_name:
            # NDVI products (MOD13Q1, MYD13A2, etc.)
            da = da.where(da != -3000) * 0.0001  # NDVI scale factor
            da.name = "NDVI"
        elif "VNP46A" in self.short_name:
            da = da.rio.write_nodata(-9999)  # int16-safe nodata
            da = da.where(da != 65535, -9999)
        else:
            da.name = "band_data"
            da = da.rio.write_nodata(-9999)  # Set nodata for int16 data
            da = da.where(da != 65535, -9999)
        # -------------------------------------------------

        return da

    def _get_date(self, filename):
        match = re.search(r'\.A(\d{7})\.', filename)
        return pd.to_datetime(match.group(1), format="%Y%j")

    def _get_hv(self, filename):
        match = re.search(r'\.h(\d+)v(\d+)\.', filename)
        return int(match.group(1)), int(match.group(2))
    
    def _validate_latlon_bounds(self, lat, lon, tol=1e-6):
        """
        Validate that the lat/lon arrays define a meaningful bounding box.

        Parameters
        ----------
        lat : np.ndarray
            2D array of latitude values.
        lon : np.ndarray
            2D array of longitude values.
        tol : float
            Minimum required span; values smaller are treated as collapsed.

        Raises
        ------
        ValueError
            If the lat/lon bounding box is degenerate (collapsed to a line/point).
        """

        min_lat, max_lat = np.nanmin(lat), np.nanmax(lat)
        min_lon, max_lon = np.nanmin(lon), np.nanmax(lon)

        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon

        errors = []

        if lat_span < tol:
            errors.append(
                f"Latitude range is too small: span={lat_span:.6f}. "
                "This suggests a collapsed latitude dimension or swapped axes."
            )

        if lon_span < tol:
            errors.append(
                f"Longitude range is too small: span={lon_span:.6f}. "
                "This suggests a collapsed longitude dimension or swapped axes."
            )

        if errors:
            raise ValueError("\n".join(errors))

        return True

    def _regridding_with_xe_modis(self,
        ds_src: xr.Dataset,
        method: str = "bilinear",
        lon_res: float | None = None,
        lat_res: float | None = None,
        ):

        import xesmf as xe
        """
        Regrid a Dataset from EPSG:6933 to EPSG:4326 using xESMF.
        If lon_res / lat_res are None, infer from rio.resolution().
            """
        # Ensure CRS metadata
        if hasattr(ds_src, "rio"):
            ds_src = ds_src.rio.write_crs(self._crs)

        # ---- Create 2D lon/lat coordinates for xESMF ----
        if "x" in ds_src.coords and "y" in ds_src.coords:
            # Projected coordinates
            x = ds_src.coords["x"].values
            y = ds_src.coords["y"].values

            # Clip x/y to valid EPSG:6933 range to avoid lat warning
            x_min, x_max = -3.5e7, 3.5e7
            y_min, y_max = -3.5e7, 3.5e7
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)

            # Build 2D meshgrid (shape y,x)
            X, Y = np.meshgrid(x, y, indexing="xy")

            # Transform to geographic coordinates
            transformer = Transformer.from_crs("EPSG:6933", self._crs, always_xy=True)
            lon2d, lat2d = transformer.transform(X, Y)

            # Optional safety clip
            lon2d = np.clip(lon2d, -180, 180)
            lat2d = np.clip(lat2d, -90, 90)

            # Assign CF-compliant coordinates
            ds_src = ds_src.assign_coords(
                lon=(("y", "x"), lon2d),
                lat=(("y", "x"), lat2d)
            )

        # Infer resolution if not provided
        if lon_res is None or lat_res is None:
            res_x, res_y = ds_src.rio.resolution()
            lon_res = abs(lon_res or res_x)
            lat_res = abs(lat_res or res_y)

        # Define target lat/lon grid based on source bounds
        bounds = ds_src.rio.bounds()
        lon_target = np.arange(bounds[0], bounds[2] + lon_res, lon_res)
        lat_target = np.arange(bounds[1], bounds[3] + lat_res, lat_res)

        ds_out = xr.Dataset(
            coords={
                "lon": (["lon"], lon_target),
                "lat": (["lat"], lat_target)
            }
        )

        # Regrid using xESMF
        regridder = xe.Regridder(ds_src, ds_out, method=method, reuse_weights=False)
        ds_regridded = regridder(ds_src)

        # ---- Mask: keep only coordinates with non-null data ----
        src_mask = xr.where(ds_src.isnull(), 0, 1).mean(dim=list(ds_src.data_vars)).fillna(0)
        mask_regridded = regridder(src_mask)
        ds_regridded = ds_regridded.where(mask_regridded > 0)

        # Attach CRS metadata
        ds_regridded.attrs["crs"] = self._crs

        return ds_regridded
    
    def _get_processed_tiffs(self):
        """
        Get all processed TIFF dates from local or cloud storage.
        Works with both local filesystem and MinIO/S3 storage via /vsis3/.
        """
        processed_dates = set()

        if self._store_cloud:
            # GDAL /vsis3/ path
            tif_out_dir = f"{self.granule_dir.rsplit('/', 1)[0]}/tiffs"
            files = gdal.ReadDir(tif_out_dir)
            if files:
                for filename in files:
                    if filename.endswith(".tif"):
                        parts = filename.split("_")
                        if len(parts) >= 2:
                            date_str = parts[-1].replace(".tif", "")
                            try:
                                date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                                processed_dates.add(date_fmt)
                            except ValueError:
                                logger.debug(f"Skipping S3 file with non-matching date: {filename}")
            logger.info(f"Found {len(processed_dates)} processed dates in MinIO: {tif_out_dir}")

        else:
            # Local filesystem
            tif_out_dir = Path(self.granule_dir).parent / "tiffs"
            if tif_out_dir.exists():
                for tif_file in tif_out_dir.glob("*.tif"):
                    date_str = tif_file.stem.split('_')[-1]
                    try:
                        date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                        processed_dates.add(date_fmt)
                    except ValueError:
                        logger.debug(f"Skipping local file with non-matching date: {tif_file.name}")
            logger.info(f"Found {len(processed_dates)} processed dates locally: {tif_out_dir}")

        return processed_dates
    
    def _exclude_processed_files(self, files):
        processed_dates = self._get_processed_tiffs()

        # Step 2: Filter self.files
        filtered_files = []
        for h5_file in files:
            h5_stem = Path(h5_file).stem
            # Extract the AYYYYDDD part
            doy_part = [part for part in h5_stem.split('.') if part.startswith('A')]
            if not doy_part:
                continue  # skip if no AYYYYDDD found
            doy_str = doy_part[0][1:]  # remove the 'A', e.g., '2012001'
            # Convert YYYYDDD to YYYYMMDD
            date_obj = datetime.strptime(doy_str, "%Y%j")
            date_str = date_obj.strftime("%Y-%m-%d")

            if date_str not in processed_dates:
                filtered_files.append(h5_file)
        
        pattern = re.compile(r'\.h(\d+)v(\d+)\.')
        files_sorted = sorted(filtered_files, key=lambda x: tuple(map(int, pattern.search(x).groups())))
        
        return files_sorted


    def _mosaic_daily(self, max_workers=1):
        """
        Process all tiles, mosaic and reproject them per day.
        Export either as Zarr or TIFF based on self.arg.output.
        """
        logger.info("🧩 Grouping tiles by date...")

        # --- Group files by date ---
        files_by_date = defaultdict(list)
        for f in self.files:
            files_by_date[self._get_date(f)].append(f)

        variables = [self.variable] if isinstance(self.variable, str) else list(self.variable)

        if self._store_cloud is False:
            self.tif_out_dir = self.granule_dir.parent / "tiffs"
        else:
            self.tif_out_dir = self.temp_dir / "tiffs"

        self.tif_out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------
        # Helper: process one day
        # ------------------------
        def process_one_day(date_files):
            date, files = date_files
            variable_data = {}

            # --- Cache subdatasets for all tiles ---
            subdatasets_cache = {}
            for f in files:
                ds = gdal.Open(str(f))
                if ds is None:
                    logger.warning(f"⚠️ Could not open {f}, skipping.")
                    if f.endswith(self.raw_data_type):
                        Path(f).unlink(missing_ok=True)
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
                    logger.warning(f"⚠️ No subdatasets for '{var}' on {date:%Y-%m-%d}")
                    continue

                da = self._build_mosaic(subdatasets, var)
                if da is None:
                    continue

                # --- Custom preprocessing ---
                da = self._custom_preprocess(da)
                da = da.expand_dims(time=[date])
                da.name = var
                variable_data[var] = da

            # --- Merge variables ---
            if not variable_data:
                return None

            ds_date = self._merge_dataarrays(list(variable_data.values()))
            logger.info(f"Reprojecting dataset for {date:%Y-%m-%d}...")
            # ds_date = self._reproject(ds_date)

            # --- Export tiff---
            if self._output_format.lower() == "tiff":
                self._export_data(ds_date, date)
            return ds_date

        # ------------------------
        # Parallel or sequential execution
        # ------------------------
        mosaics = []
        items = sorted(files_by_date.items())
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process_one_day, item): item[0] for item in items}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing dates"):
                    result = fut.result()
                    if result is not None:
                        mosaics.append(result)
        else:
            for item in tqdm(items, desc="Processing dates"):
                ds_date = process_one_day(item)
                if ds_date is not None:
                    mosaics.append(ds_date)
                    del ds_date
                    gc.collect()

        # --- If Zarr is requested, merge all days into a single dataset ---
        if self._output_format.lower() == "zarr":
            ds = xr.open_mfdataset(str(self.granule_dir / "*.nc"), combine="by_coords", parallel=True, chunks="auto")
            ds= prepare(ds)
            ds = ds.chunk({"time": 1, "lat": 1024, "lon": 1024})
            self._export_to_zarr(ds)


    # ------------------------
    # Helper: build mosaic for one variable
    # ------------------------
    def _build_mosaic(self, subdatasets, var):
        from osgeo import gdal
        import rioxarray

        vrt_path = self.temp_dir / f"{self.short_name}_{var}.vrt"
        tif_path = self.temp_dir / f"{self.short_name}_{var}.tif"

        try:
            gdal.UseExceptions()
            
            # Reproject each subdataset to UTM before mosaicking
            temp_tifs = []
            for i, subdataset in enumerate(subdatasets):
                da = rioxarray.open_rasterio(subdataset, chunks=True).squeeze("band", drop=True)
                dest_crs = self._get_utm_crs(da.rio.bounds())
                
                if not dest_crs.startswith("EPSG"):
                    raise ValueError(f"Invalid UTM CRS returned: {dest_crs}")
                da_utm = da.rio.reproject(
                    dst_crs=dest_crs,
                    resampling=self._reproj_method,
                    resolution=self._resolution
                )
                temp_tif = self.temp_dir / f"temp_{var}_{i}.tif"
                da_utm.rio.to_raster(str(temp_tif), driver="GTiff")
                temp_tifs.append(str(temp_tif))
            
            vrt = gdal.BuildVRT(str(vrt_path), temp_tifs, creationOptions=["NUM_THREADS=ALL_CPUS"])
            if vrt is None:
                raise RuntimeError("BuildVRT returned None")

            gdal.Translate(
                str(tif_path),
                vrt,
                creationOptions=["TILED=YES", "BIGTIFF=YES", "COMPRESS=LZW", "NUM_THREADS=ALL_CPUS"]
            )
            vrt = None

            da_mosaic = rioxarray.open_rasterio(tif_path, chunks=True).squeeze("band", drop=True).astype("int16")
            return da_mosaic 

        except RuntimeError as e:
            logger.warning(f"GDAL mosaic failed for {var}: {e}")
            return None
        

    def _drop_empty_coordinates(self, ds:xr.Dataset):
        # Only drop if coordinates exist
        if "lat" in ds.coords:
            ds = ds.dropna(dim="lat", how="all")
        if "lon" in ds.coords:
            ds = ds.dropna(dim="lon", how="all")
        if "x" in ds.coords:
            ds = ds.dropna(dim="x", how="all")
        if "y" in ds.coords:
            ds = ds.dropna(dim="y", how="all")
        return ds


    def _export_multiband_raster(
        self,
        ds: xr.Dataset | xr.DataArray,
        date,
        tile_dir: str = None,
        dtype: str = "int16",
    ):
        """
        Export a multiband GeoTIFF either locally or directly to MinIO.
        Storage location is determined by self._store_cloud flag.

        Parameters
        ----------
        ds : xr.Dataset or xr.DataArray
            Dataset to export
        date : datetime
            Date for the filename
        tile_dir : str, optional
            Local directory path (required if store_cloud=False)
        dtype : str
            Data type for output, default "int16"
        """

        # Construct output path
        filename = f"{self.short_name}_{date:%Y%m%d}.tif"
        
        if self._store_cloud:
            # Cloud: full /vsis3/ path as string
            out_tif = f"{self.tif_out_dir}/{filename}"
        else:
            # Local filesystem
            if tile_dir is None:
                tile_dir = self.tif_out_dir

            tile_dir = Path(tile_dir)
            tile_dir.mkdir(parents=True, exist_ok=True)
            out_tif = tile_dir / filename

        var_names = list(ds.data_vars)
        first = ds[var_names[0]]

        time, height, width = first.shape
        transform = first.rio.transform()
        crs = first.rio.crs

        is_float = np.issubdtype(np.dtype(dtype), np.floating)
        predictor = 3 if is_float else 2

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(var_names),
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "DEFLATE",
            "ZLEVEL": 5,
            "TILED": True,
            "BLOCKXSIZE": 256,
            "BLOCKYSIZE": 256,
            "PREDICTOR": predictor,
            "BIGTIFF": "YES" if (height * width * len(var_names) * 2 > 4e9) else "NO",
        }

        with rasterio.open(str(out_tif), "w", **profile) as dst:
            for i, var in enumerate(var_names, start=1):
                arr = ds[var].data.squeeze()
                dst.write(arr.astype(dtype), i)
                dst.set_band_description(i, var)

        if self._store_cloud:
            self._minio_client.fput_object(
            os.getenv('MINIO_BUCKET'),
            f"{extract_object_from_minio(self.granule_dir)}/tiffs/{filename}",
            str(out_tif),
        )
                    

    def _export_single_bands(self, ds_date, date, tile_dir=None):
        """
        Export single-band rasters, either locally or to MinIO.
        """
        if tile_dir is None and not self._store_cloud:
            tile_dir = self.tif_out_dir
        
        for var in ds_date.data_vars:
            filename = f"{self.short_name}_{var}_{date:%Y%m%d}.tif"
            
            if not self._store_cloud:
                out_tif = Path(tile_dir) / filename
                Path(tile_dir).mkdir(parents=True, exist_ok=True)
            
            da = ds_date[var]
            
            # Determine appropriate predictor based on data type
            # PREDICTOR=3 (float prediction) only for float dtypes
            # PREDICTOR=2 (horizontal differencing) for integer dtypes
            is_float = np.issubdtype(da.dtype, np.floating)
            predictor = 3 if is_float else 2
            
            # Optimized GeoTIFF options for single-band export
            rio_kwargs = {
                "driver": "GTiff",
                "compress": "DEFLATE",
                "ZLEVEL": 5,              # Balance compression vs speed (~2-3x faster than ZLEVEL=9)
                "TILED": True,
                "BLOCKXSIZE": 256,
                "BLOCKYSIZE": 256,
                "PREDICTOR": predictor,   # 3 for float, 2 for integer
                "dtype": str(da.dtype)
            }
            
            da.rio.to_raster(str(out_tif), **rio_kwargs)
            
            storage_type = "MinIO" if self._store_cloud else "local"
            logger.debug(f"✅ Exported {var} to {storage_type}: {out_tif}") 

    # ------------------------
    # Helper: export dataset according to arg.output
    # ------------------------
    def _export_data(self, ds_date:xr.Dataset | xr.DataArray, date:str): 
        """ Export daily dataset according to self._output_format """ 
        if self._output_format.lower() == "tiff": 
            self._export_multiband_raster(ds_date, date)
        else: 
            raise ValueError(f"Unknown output format '{self._output_format}'")
        
    def _merge_dataarrays(self, dataarrays: list[xr.DataArray]) -> xr.Dataset:
        """
        Merge a list of xarray DataArrays into a Dataset.
        Checks for differing spatial resolutions and issues a warning.
        For MOD09GA, resamples state_1km to match the resolution of other bands (500m).
        """
        if not dataarrays:
            return xr.Dataset()
        
        # Check spatial resolutions
        resolutions = []
        for da in dataarrays:
            if hasattr(da, 'rio') and da.rio.crs is not None:
                res = da.rio.resolution()
                resolutions.append(res)
            else:
                resolutions.append(None)
        
        unique_res = set(tuple(r) if r is not None else None for r in resolutions)
        if len(unique_res) > 1:
            logger.warning(f"Different spatial resolutions detected among DataArrays: {unique_res}")
        
        # For MOD09GA, resample state_1km to 500m if necessary
        if self.short_name == "MOD09GA":
            target_da = None
            for da in dataarrays:
                if da.name != "state_1km":
                    target_da = da
                    break
            if target_da is not None:
                for i, da in enumerate(dataarrays):
                    if da.name == "state_1km":
                        # Resample state_1km to match target_da resolution
                        dataarrays[i] = da.rio.reproject_match(target_da, resampling=self._reproj_method)
                        logger.info("Resampled state_1km to match other bands' resolution.")
                        break
        
        # Merge the DataArrays
        return xr.merge(dataarrays)

    def _convert_crs(self, ds: xr.Dataset | xr.DataArray, dst_crs: str, bbox_deg: tuple[float, float, float, float]) -> xr.Dataset | xr.DataArray:
        """
        Transform bounding box to dataset CRS and clip the dataset.
        """
        if isinstance(ds, xr.DataArray):
            ds_repr = ds.to_dataset(name=ds.name)
        else:
            ds_repr = ds

        project = pyproj.Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True).transform
        minx, miny = project(bbox_deg[0], bbox_deg[1])
        maxx, maxy = project(bbox_deg[2], bbox_deg[3])

        return [minx, miny, maxx, maxy]

    def _reproject(self, ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:

        if isinstance(ds, xr.DataArray):
            ds_repr = ds.to_dataset(name=ds.name)
        else:
            ds_repr = ds

        if self._reproj_lib == "rioxarray":
            from rasterio.enums import Resampling
            resampling_dict = {
                "nearest": Resampling.nearest,
                "bilinear": Resampling.bilinear,
            }

            reproject = resampling_dict[self._reproj_method]

            ds_repr = ds_repr.rio.reproject(
                dst_crs=self._crs,   
                resampling=reproject,
                resolution=self._resolution,
            )#.chunk({"x": 1024, "y": 1024})

            if (self._crs == "EPSG:6933" or self._crs.startswith("EPSG:326") or self._crs.startswith("EPSG:327")) and self.bbox is not None:
                # Convert bbox to dataset CRS
                bbox_projected = self._convert_crs(ds_repr, self._crs, self.bbox)
                minx, miny, maxx, maxy = bbox_projected
            elif self._crs == "EPSG:4326" and self.bbox is not None:
                minx, miny, maxx, maxy = self.bbox
            else:
                minx, miny, maxx, maxy = ds_repr.rio.bounds()

            ds_tmp = ds_repr.rio.clip_box(
                    minx=minx,
                    miny=miny,
                    maxx=maxx,
                    maxy=maxy,
                    allow_one_dimensional_raster=True,
                )

            # Drop degenerate clips
            if ds_tmp.sizes.get("x", 0) <= 1 or ds_tmp.sizes.get("y", 0) <= 1:
                logger.warning(
                    f"⚠️ Clipped raster collapsed to 1D "
                    f"(x={ds_tmp.sizes.get('x')}, y={ds_tmp.sizes.get('y')}), skipping."
                )
                return ds_repr

        elif self._reproj_lib == "xesmf":
            ds_repr = self._regridding_with_xe_modis(
                ds_repr, 
                method=self._reproj_method
            )
        else:
            raise ValueError(f"Unknown reproj_lib '{self._reproj_lib}'")
        
        if isinstance(ds, xr.DataArray):
            return ds_repr[ds.name]
        else:
            return ds_repr
        
    # ------------------------
    # Export to Zarr
    # ------------------------
    def _export_to_zarr(self, ds):
        """
        Export dataset to Zarr format, either locally or to MinIO.
        """
        if self._store_cloud:
            zarr_path = f"/vsis3/{self.minio_bucket}/{self.minio_prefix}/{self.collection_name}_dataset.zarr"
        else:
            zarr_path = self.zarr_path
        
        logger.info(f"💾 Appending data to {zarr_path}")
        
        if self._store_cloud:
            # For cloud storage, we need to check existence differently
            try:
                ds_existing = xr.open_zarr(zarr_path)
                # Append to existing
                ds.to_zarr(zarr_path, mode="a", append_dim="time")
            except (FileNotFoundError, KeyError, ValueError):
                # Create new
                ds.to_zarr(zarr_path, mode="w")
        else:
            if not Path(zarr_path).exists():
                ds.to_zarr(zarr_path, mode="w")
            else:
                ds.to_zarr(zarr_path, mode="a", append_dim="time")

    # ------------------------
    # Cleanup
    # ------------------------
    def _cleanup_raw_files(self):
        """Delete raw downloaded files (HDF/H5) to free disk space."""
        raw_extensions = (self.raw_data_type.lstrip("."), self.raw_data_type)
        deleted_count = 0
        
        if self.granule_dir.exists():
            for f in os.listdir(self.granule_dir):
                if f.endswith(raw_extensions):
                    file_path = os.path.join(self.granule_dir, f)
                    try:
                        Path(file_path).unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"🧹 Deleted {deleted_count} raw data files from {self.granule_dir}")

    def cleanup(self, clean_nontemp=False):
        logger.info(f"🧹 Cleaning up temporary files in {self.temp_dir}")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if clean_nontemp:
            self._cleanup_raw_files()

    # ------------------------
    # Full pipeline with batch support
    # ------------------------
    def _split_date_range(self, start_date, end_date, batch_days=30):
        """
        Split a date range into batches of specified days.
        
        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        batch_days : int
            Number of days per batch
            
        Returns
        -------
        list of tuples
            List of (batch_start, batch_end) date ranges
        """
        from datetime import datetime, timedelta
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        batches = []
        
        current = start
        while current < end:
            batch_end = min(current + timedelta(days=batch_days), end)
            batches.append((current.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")))
            current = batch_end + timedelta(days=1)
        
        return batches
    
    def build_tile_to_grid_lookup(self, grid, cache_path):
        """
        Build mapping:
            (h, v) -> list of (grid_id, lat, lon)
        using majortom.Grid points.
        """
        import pickle
        from tqdm import tqdm
        from transform.majortom import latlon_to_sinu, sinu_to_tile

        tile_lookup = {}

        points = grid.points  # GeoDataFrame

        for _, row in tqdm(points.iterrows(), total=len(points)):
            grid_id = row["name"]
            lon = row.geometry.x
            lat = row.geometry.y

            # lat/lon → MODIS sinusoidal
            x, y = latlon_to_sinu(lat, lon)
            h, v = sinu_to_tile(x, y)

            tile_lookup.setdefault((h, v), []).append(
                (grid_id, lat, lon)
            )

        with open(cache_path, "wb") as f:
            pickle.dump(tile_lookup, f)

        return tile_lookup

    def _generate_global_aoi(self, generate_global=False):
        import pickle

        self.grid = Grid(dist=100)

        if not generate_global:
            return None

        grid_file = Path(DATA_PATH) / "tile_to_grid_global.pkl"

        if grid_file.exists():
            with open(grid_file, "rb") as f:
                return pickle.load(f)

        tile_to_grid = self.build_tile_to_grid_lookup(
            self.grid,
            cache_path=grid_file,
        )
        return tile_to_grid
    
    def run(self, batch_days=30, majortom_grid: bool = False, pixel_size=250):
        if majortom_grid:
            from transform import CalculationsMajorTom
            self.calculations =  CalculationsMajorTom(pixel_size=pixel_size)
            tile_to_grid = self._generate_global_aoi(generate_global=True)
        else:
            tile_to_grid = None

        start_date, end_date = self.date_range
        batches = self._split_date_range(start_date, end_date, batch_days=batch_days)

        if not self._store_cloud:
            self._cleanup_raw_files()

        logger.info(f"📦 Processing {len(batches)} batches of ~{batch_days} days")

        for i, (batch_start, batch_end) in enumerate(batches, 1):
            logger.info(f"Batch {i}/{len(batches)}: {batch_start} → {batch_end}")

            self._search_and_download(date_range=(batch_start, batch_end))

            if not self.files:
                logger.info("No new files — skipping batch")
                continue

            if majortom_grid:
                self.build_or_update_majortom_zarr(
                    tile_to_grid=tile_to_grid,
                    patch_size=64,
                )
            else:
                self._mosaic_daily()

            if not self._store_cloud:
                self._cleanup_raw_files()
            else:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("✅ All batches completed")

    def _init_or_open_zarr(self, grid_ids, patch_size):

        HALF = patch_size
        PATCH_SIZE = 2 * HALF

        grid_ids = pd.Index(
            [str(g) for g in grid_ids],
            name="grid_id",
        )

        variables = (
            self.variable if isinstance(self.variable, (list, tuple))
            else [self.variable]
        )

        zarr_path = Path(self.zarr_path)

        if not zarr_path.exists():
            logger.info("🆕 Creating new MajorTOM Zarr store")

            data_vars = {
                var: (
                    ("grid_id", "time", "y", "x"),
                    np.empty(
                        (len(grid_ids), 0, PATCH_SIZE, PATCH_SIZE),
                        dtype=np.float16,
                    ),
                )
                for var in variables
            }

            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "grid_id": grid_ids,
                    "time": pd.DatetimeIndex([], name="time"),
                    "y": np.arange(PATCH_SIZE),
                    "x": np.arange(PATCH_SIZE),
                },
                attrs={
                    "grid": "MajorTOM",
                    "projection": "MODIS sinusoidal",
                    "resolution_m": self._resolution,
                },
            )

            encoding = {
                var: {"chunks": (1, 1, PATCH_SIZE, PATCH_SIZE)}
                for var in variables
            }

            ds.to_zarr(zarr_path, mode="w", encoding=encoding)
            return ds

        logger.info("📦 Opening existing Zarr store")
        return xr.open_zarr(zarr_path)

    def build_or_update_majortom_zarr(
        self,
        tile_to_grid,
        patch_size=64,
        ):

        if not tile_to_grid:
            logger.warning("No MajorTOM tile mapping provided")
            return

        # collect unique grid_ids
        grid_ids = sorted({
            grid_id
            for cells in tile_to_grid.values()
            for grid_id, _, _ in cells
        })

        logger.info(f"📍 Using {len(grid_ids)} MajorTOM grid cells")

        all_times = sorted({self._get_date(fp) for fp in self.files})
        logger.info(f"📅 Processing {len(all_times)} unique dates")

        ds = self._init_or_open_zarr(
            grid_ids=grid_ids,
            patch_size=patch_size,
        )

        ds, time_index = self._append_times(
            zarr_path=self.zarr_path,
            new_times=all_times,
        )

        if not time_index:
            logger.info("⏭️ All times already present")

        self._stream_tiles_into_zarr(
            zarr_path=self.zarr_path,
            tile_to_grid=tile_to_grid,
            time_index=time_index,
            patch_size=patch_size,
        )

    def _stream_tiles_into_zarr(
        self,
        zarr_path,
        tile_to_grid,
        time_index,
        patch_size=64,
    ):

        HALF = patch_size
        PATCH_SIZE = 2 * HALF

        variables = (
            self.variable if isinstance(self.variable, (list, tuple))
            else [self.variable]
        )

        ds = xr.open_zarr(zarr_path)
        grid_index = {gid: i for i, gid in enumerate(ds.grid_id.values)}

        logger.info(f"🧬 Streaming {len(self.files)} MODIS tiles into Zarr")

        for file_path in tqdm(self.files, desc="MODIS tiles"):
            try:
                date = self._get_date(file_path)
                if date not in time_index:
                    continue
                t_idx = time_index[date]

                fname = Path(file_path).name
                m = re.search(r"h(\d+)v(\d+)", fname)
                if not m:
                    continue
                h, v = int(m.group(1)), int(m.group(2))

                cells = tile_to_grid.get((h, v))
                if not cells:
                    continue

                hdf = gdal.Open(str(file_path))
                if hdf is None:
                    continue

                for var in variables:
                    var_path = None
                    for sd_path, sd_name in hdf.GetSubDatasets():
                        if var in sd_name:
                            var_path = sd_path
                            break
                    if var_path is None:
                        continue

                    with rasterio.open(var_path) as src:
                        data = src.read(1).astype(np.float16)
                        ny, nx = data.shape

                        patches, grid_idxs = [], []

                        for grid_id, lat, lon in cells:
                            g_idx = grid_index.get(grid_id)
                            if g_idx is None:
                                continue

                            x, y = self.calculations.latlon_to_sinu(lat, lon)
                            px, py = self.calculations.xy_to_pixel(x, y, h, v)

                            if (
                                py - HALF < 0 or px - HALF < 0 or
                                py + HALF >= ny or px + HALF >= nx
                            ):
                                continue

                            patch = data[
                                py - HALF : py + HALF,
                                px - HALF : px + HALF,
                            ]

                            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                                continue

                            patches.append(patch)
                            grid_idxs.append(g_idx)

                        if not patches:
                            continue

                        patches = np.stack(patches, axis=0)

                        # SAFE, backend-agnostic write
                        ds[var].values[
                            grid_idxs,
                            t_idx,
                            :,
                            :
                        ] = patches

            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}", exc_info=True)

        logger.info("✅ MajorTOM Zarr streaming completed")

    def _append_times(self, zarr_path, new_times):
        # normalize variables
        variables = (
            self.variable if isinstance(self.variable, (list, tuple))
            else [self.variable]
        )

        ds = xr.open_zarr(zarr_path)

        # normalize time types
        existing_times = pd.to_datetime(ds.time.values)
        new_times = pd.to_datetime(new_times)

        times_to_add = [t for t in new_times if t not in set(existing_times)]

        if not times_to_add:
            logger.info("⏭️ No new times to append")
            return ds, {}

        logger.info(f"➕ Appending {len(times_to_add)} new timesteps")

        n_grid = len(ds.grid_id)
        n_time = len(times_to_add)
        ny = len(ds.y)
        nx = len(ds.x)

        # build empty data_vars for ALL variables
        data_vars = {}
        for var in variables:
            data_vars[var] = (
                ("grid_id", "time", "y", "x"),
                np.full(
                    (n_grid, n_time, ny, nx),
                    np.nan,
                    dtype=np.float16,
                ),
            )

        append_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": times_to_add,
                "grid_id": ds.grid_id,
                "y": ds.y,
                "x": ds.x,
            },
        )

        append_ds.to_zarr(
            zarr_path,
            append_dim="time",
        )

        # compute time → index mapping
        start_idx = len(existing_times)
        time_index = {
            t: start_idx + i
            for i, t in enumerate(times_to_add)
        }

        return xr.open_zarr(zarr_path), time_index

class StacModisTileProcessor:
    def __init__(self, 
                 stac_url, 
                 collection, 
                 time_range, 
                 bands, 
                 crs="EPSG:6933", 
                 resolution=1000, 
                 tile_size=256, 
                 stride=256, 
                 max_missing=0.5):
        
        self.stac_url = stac_url
        self.collection = collection
        self.time_range = time_range
        self.bands = bands
        self.crs = crs
        self.resolution = resolution
        self.tile_size = tile_size
        self.stride = stride
        self.max_missing = max_missing
        self.catalog = Client.open(stac_url)
        self.ds = None
        self.out_dir = DATA_PATH / "modis"  # Place under data/ per project conventions
        self.out_dir.mkdir(exist_ok=True)

    def load_data(self):
        search = self.catalog.search(collections=[self.collection], datetime=self.time_range)
        items = list(search.get_items())
        if not items:
            raise ValueError("No items found for the given search parameters.")
        
        self.ds = load(
            items,
            bands=self.bands,
            crs=self.crs,
            resolution=self.resolution,
            groupby="solar_day",
            chunks={"time": 5, "x": 1024, "y": 1024},
        )
        logger.info(f"Loaded dataset: {self.ds}")

    def process_tiles(self):
        if self.ds is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        x_coords = self.ds.x.values
        y_coords = self.ds.y.values
        x_tiles = range(0, len(x_coords) - self.tile_size + 1, self.stride)
        y_tiles = range(0, len(y_coords) - self.tile_size + 1, self.stride)
        
        tile_id = 0
        for ix in x_tiles:
            for iy in y_tiles:
                tile = self.ds.isel(
                    x=slice(ix, ix + self.tile_size),
                    y=slice(iy, iy + self.tile_size),
                )
                
                # Skip tiles with too much missing data
                if tile[self.bands[0]].isnull().mean() > self.max_missing:
                    continue
                
                # Convert to numpy (time, bands, y, x)
                tile_np = np.stack([tile[band].values for band in self.bands], axis=1)
                
                np.save(self.out_dir / f"tile_{tile_id:06d}.npy", tile_np)
                tile_id += 1
        logger.info(f"Processed {tile_id} tiles.")


if __name__ == "__main__":
    from utils import init_logging, countries_to_bbox
    from osgeo import gdal
    import argparse
    from dotenv import load_dotenv
    import geopandas as gpd 

    from definitions import ROOT_DIR
    load_dotenv(Path(ROOT_DIR)/ ".env")

    gdal.SetConfigOption('GDAL_CACHEMAX', '512')    # MB, tune to memory
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    import sys
    sys.dont_write_bytecode = True

    parser = argparse.ArgumentParser(description="MODIS Downloader")
    parser.add_argument('--product', type=str, default='VIIRS_500m_night_monthly', help='MODIS product to download')
    parser.add_argument('--start_date', type=str, help='date to start collection')
    parser.add_argument('--end_date', type=str, help='date to end collection')
    parser.add_argument('--reproj_lib', choices=['rioxarray', 'xesmf'], default=os.getenv('REPROJ_LIB', 'rioxarray'), help='Reprojection library to use (rioxarray or xesmf)')
    parser.add_argument('--reproj_method', choices=['nearest', 'bilinear'], default=os.getenv('REPROJ_METHOD', 'nearest'), help='Reprojection method to use')    
    parser.add_argument('-d', '--delete_temp', action='store_true', default=False, help='Delete temporary files after processing')
    args = parser.parse_args()

    products = {
        "LST": {
            "short_name": "MOD11A1",
            "variables": ["MODIS_Grid_Daily_1km_LST:LST_Day_1km"]
        },
        "reflectance_250m": {
            "short_name": "MOD09GQ",
            "variables": ["sur_refl_b01", 
                           "sur_refl_b02", 
                           "QC_250m"
            ],
            "raw_data_type" : "hdf",
            "crs": "EPSG:6933",
        },
        "reflectance_1000m": {
            "short_name": "MOD09GA",
            "variables": ["sur_refl_b01",
                          "sur_refl_b02",
                          "state_1km"
            ],
            "raw_data_type" : "hdf",
            "crs": "EPSG:6933",
        },
        "NDVI_1km_monthly": {
            "short_name": "MOD13A3",
            "variables": ["NDVI",
                          "EVI",
                          "SummaryQA"
            ],
            "raw_data_type" : "hdf",
            "crs": "EPSG:6933",
        },
        "VIIRS_500m_night_monthly": {
            "short_name": "VNP46A3",
            "variables": ["NearNadir_Composite_Snow_Free",
                          "NearNadir_Composite_Snow_Free_Std",
                          "NearNadir_Composite_Snow_Free_Quality"
            ],
            "raw_data_type" : "h5"
        },
        
        "VIIRS_500m_night_daily": {
            "short_name": "VNP46A2",
            "variables": ["Gap_Filled_DNB_BRDF_Corrected_NTL",
                          "DNB_BRDF_Corrected_NTL",
                          "QF_Cloud_Mask"
            ],
            "raw_data_type" : "h5"
        }
    }

    variables = products[args.product]["variables"]
    short_name = products[args.product]["short_name"]
    raw_data_type = products[args.product]["raw_data_type"]

    logger = init_logging(log_file="modis_downloader.log", verbose=False)
    
    gdf = gpd.read_file(DATA_PATH/ "shapefiles"/ "GAUL_2024.zip")
    bbox, polygon = countries_to_bbox(["Brazil", "Argentina", "Peru", "Colombia", "Panama"], gdf, col_name="gaul0_name")

    try:
        
        downloader = EarthAccessDownloader(
        short_name=short_name,
        bbox= bbox,
        variables=variables,
        date_range=(args.start_date, args.end_date),
        collection_name=f"{short_name}_061",
        reproj_lib=args.reproj_lib,
        reproj_method=args.reproj_method,
        output_format="tiff",
        raw_data_type=raw_data_type
    )   
        if args.delete_temp and downloader.temp_dir.exists():
            logger.warning("Deleting temporary directory as per user request.")
            shutil.rmtree( downloader.temp_dir)

        downloader.cleanup()
        downloader.run(batch_days=10)

    except Exception as e:
        # downloader.cleanup()
        # if downloader.granule_dir.exists():
        #     shutil.rmtree( downloader.granule_dir)
        logger.error(e)
        raise e