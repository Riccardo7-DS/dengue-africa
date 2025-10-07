import ee, geemap, xarray as xr
import logging
from tqdm.auto import tqdm

ee.Authenticate()
ee.Initialize(project="ee-querying-maps")
logger = logging.getLogger(__name__)


import os
import xarray as xr
from typing import Literal

class EeModis():
    def __init__(self,
                 start_date:str, 
                 end_date:str,
                 name:Literal["ref_061", "LST_061"],
                 output_dir:str,
                 geometry=None,
                 output_resolution:int=1000,
                 download_collection:bool=True):
        
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



if __name__ == "__main__":
    EeModis(
        start_date="2013-09-08", 
        end_date= "2023-02-17",
        name= "NDVI_06", 
        output_resolution=1000,
        download_collection=True)