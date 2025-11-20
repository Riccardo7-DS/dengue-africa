# import ee
import logging
from pathlib import Path
# from osgeo import gdal
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
from utils import prepare
from pyproj import Transformer
# import geemap 



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
        short_name="MOD11A1",
        variables="MODIS_Grid_Daily_1km_LST:LST_Day_1km",
        date_range=None,
        bbox=None,
        data_dir=Path(DATA_PATH) / "modis",
        collection_name="lst_061",
        zarr_path=None,
        reproj_lib:Literal["rioxarray","xesmf"]="xesmf",
        reproj_method:Literal["nearest","bilinear"]="nearest",
        output_format:Literal["tiff","zarr"]="zarr",
    ):
        
        self.date_range = self._check_dates(date_range)

        self.short_name = short_name
        self.variable = variables
        self.bbox = bbox
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.granule_dir = self.data_dir / self.collection_name
        self.zarr_path = zarr_path or (self.data_dir / f"{self.short_name}_dataset.zarr")
        self._reproj_lib = self._check_reproj_lib(reproj_lib)
        self._reproj_method = self._check_reproj_method(reproj_method)

        self._output_format = output_format

        self.granule_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        self._login_earthaccess()

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
        logger.info(f"Using reproj_method: {reproj_method}")
        return reproj_method

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
            da = da.where(da > 0).astype("float32") * 0.02 - 273.15  # LST scale factor
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
            ds_src = ds_src.rio.write_crs("EPSG:6933")

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
            transformer = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
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
        ds_regridded.attrs["crs"] = "EPSG:4326"

        return ds_regridded
    

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
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.tif_out_dir = self.granule_dir / "tiffs"
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

            ds_date = xr.merge(variable_data.values())
            logger.info(f"Reprojecting dataset for {date:%Y-%m-%d}...")
            ds_date = self._reproject(ds_date)

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
            vrt = gdal.BuildVRT(str(vrt_path), subdatasets, creationOptions=["NUM_THREADS=ALL_CPUS"])
            if vrt is None:
                raise RuntimeError("BuildVRT returned None")

            gdal.Translate(
                str(tif_path),
                vrt,
                creationOptions=["TILED=YES", "BIGTIFF=YES", "COMPRESS=LZW", "NUM_THREADS=ALL_CPUS"]
            )
            vrt = None

            da = rioxarray.open_rasterio(tif_path, chunks=True).squeeze("band", drop=True).astype("int16")
            return da

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

    def _export_multiband_raster(self, ds:xr.Dataset | xr.DataArray, date:str,  tile_dir:str, dtype="int16"):

        out_tif = Path(tile_dir) / f"{self.short_name}_{date:%Y%m%d}.tif"  
        var_names = list(ds.data_vars)
        first = ds[var_names[0]]

        time, height, width = first.shape
        transform = first.rio.transform()
        crs = first.rio.crs

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(var_names),
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "LZW"
        }

        with rasterio.open(out_tif, "w", **profile) as dst:
            for i, var in enumerate(var_names, start=1):
                arr = ds[var].data
                arr = arr.squeeze()     # pulls only one band into RAM
                dst.write(arr.astype(dtype), i)
                dst.set_band_description(i, var)

    def _export_single_bands(self, ds_date, date, tile_dir):
        for var in ds_date.data_vars:
            out_tif = Path(tile_dir) / f"{self.short_name}_{var}_{date:%Y%m%d}.tif"  
            da = ds_date[var]             
            da.rio.to_raster( str(out_tif), 
                            driver="GTiff", 
                            # standard GeoTIFF 
                            compress="LZW", 
                            dtype="int16" ) 

    # ------------------------
    # Helper: export dataset according to arg.output
    # ------------------------
    def _export_data(self, ds_date:xr.Dataset | xr.DataArray, date:str): 
        """ Export daily dataset according to self.arg.output """ 
        if self._output_format.lower() == "tiff": 
            tile_dir = self.tif_out_dir / '_'.join(format(x, ".1f") for x in self.bbox)
            os.makedirs(tile_dir, exist_ok=True) 
            self._export_multiband_raster(ds_date, date, tile_dir)
        else: 
            raise ValueError(f"Unknown output format '{self._output_format}'")
        
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
                dst_crs="EPSG:4326",   
                resampling=reproject,
            )#.chunk({"x": 1024, "y": 1024})

            ds_repr = ds_repr.rio.clip_box(minx=self.bbox[0], 
                                           miny=self.bbox[1], 
                                           maxx=self.bbox[2], 
                                           maxy=self.bbox[3])

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
        [os.remove(os.path.join(self.granule_dir, x)) for x in os.listdir(self.granule_dir) if x.endswith(".hdf")]

    # ------------------------
    # Full pipeline
    # ------------------------
    def run(self):
        self._search_and_download()
        self._mosaic_daily()
        
        self.cleanup()
        logger.info("✅ Done!")




if __name__ == "__main__":
    from utils import init_logging
    from osgeo import gdal
    import argparse
    from dotenv import load_dotenv
    from definitions import ROOT_DIR
    load_dotenv(Path(ROOT_DIR)/ ".env")

    gdal.SetConfigOption('GDAL_CACHEMAX', '512')    # MB, tune to memory
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    import sys
    sys.dont_write_bytecode = True

    parser = argparse.ArgumentParser(description="MODIS Downloader")
    parser.add_argument('--product', type=str, default='reflectance_250m', help='MODIS product to download')
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
            ]
        },
        "NDVI_1km_monthly": {
            "short_name": "MOD13A3",
            "variables": ["NDVI",
                          "EVI",
                          "SummaryQA"
            ]
        }
    }

    variables = products[args.product]["variables"]
    short_name = products[args.product]["short_name"]

    logger = init_logging(log_file="modis_downloader.log", verbose=False)
    bbox = [-70.0, -70.0, -62.22222222222222, -62.22222222222222]

    try:
        
        downloader = EarthAccessDownloader(
        short_name=short_name,
        bbox= bbox,
        variables=variables,
        date_range=("2010-01-01", "2010-01-05"),
        collection_name=f"{short_name}_061",
        reproj_lib=args.reproj_lib,
        reproj_method=args.reproj_method,
        output_format="tiff"
    )   
        if args.delete_temp and downloader.temp_dir.exists():
            logger.warning("Deleting temporary directory as per user request.")
            shutil.rmtree( downloader.temp_dir)

        downloader.cleanup()
        downloader.run()

    except Exception as e:
        downloader.cleanup()
        if downloader.granule_dir.exists():
            shutil.rmtree( downloader.granule_dir)
        raise e