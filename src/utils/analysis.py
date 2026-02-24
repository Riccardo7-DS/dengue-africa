import xarray as xr
import numpy as np
import pandas as pd
from definitions import DATA_PATH
import logging
from utils.general import process_gdf
from tqdm import tqdm

logger = logging.getLogger(__name__)

def compute_ndvi(band1: xr.DataArray,
                 band2: xr.DataArray) -> xr.DataArray:
    """
    Compute NDVI given two reflectance bands:
    band1 = RED
    band2 = NIR
    """

    denom = band2 + band1
    num   = band2 - band1

    # Mask zero denominators BEFORE division → no warnings
    safe_denom = xr.where(denom == 0, np.nan, denom)

    ndvi = num / safe_denom

    return ndvi.astype("float32")


class DengueRasterizer():
    def __init__(self, 
                 resolution_degree:float=None, 
                 countries:list = ["Brazil", "Peru", "Colombia", "Argentina"],
                 output_file:str=None,
                 reproject_geopandas:bool=True,
                 extra_columns:list=None):
        
        self.extra_columns = extra_columns
        
        if output_file is None:
            countries_str = "_".join(countries)
            self.output_file = DATA_PATH / "dengue_cases"/ f"dengue_admin2_cases_{countries_str}.nc"
        else:
            self.output_file = output_file

        if resolution_degree is None:
            logger.info("No resolution_degree provided, using default FAO resolution of ~5km (0.041666666666666664 degrees)")
            resolution_degree = 0.041666666666666664  # ~5km at the equator

        self.reproject_geopandas = reproject_geopandas
        self.resolution_degree = resolution_degree
        self.countries = countries
        self.admin_data = self._load_admin_raster()
        self.raster = self._orechestrate_geometry(reproject_geopandas=self.reproject_geopandas)

    def _load_admin_raster(self):
        from utils import load_admin_data
        path = DATA_PATH / "Spatial_extract_V1_3.csv"
        return load_admin_data(path, temporal_resolution="Week", spatial_resolution="Admin2")
    
    def _orechestrate_geometry(self,  reproject_geopandas=True):
        if reproject_geopandas:
            zip_path = DATA_PATH / "shapefiles" / "GAUL_2024.zip"
            gdf = process_gdf(zip_path, countries=self.countries)
            # dissolve all admin2 into admin0 boundaries
            # raster = gdf.dissolve(by="adm_0_name", as_index=False)
            self.lat_min, self.lat_max, self.lon_min, self.lon_max = gdf.total_bounds[[1, 3, 0, 2]]
        else:
            raise NotImplementedError("Only reprojection with geopandas is implemented for now")
            tif_path = DATA_PATH / "riskmaps_public main data/admin_rasters/Admin0_5k_raster.tif"
            raster = xr.open_dataarray(tif_path, engine="rasterio").isel(band=0)

        
        self.lats = np.arange(self.lat_max, self.lat_min, -self.resolution_degree)
        self.lons = np.arange(self.lon_min, self.lon_max, self.resolution_degree)

        gdf["region_id"] = gdf["adm_0_name"] + "_" + gdf["adm_2_name"]
        gdf["region_id"] = gdf["region_id"].str.lower()

        return gdf
    
    def _rasterize_static(self, da, shapefile, region_col="region_id", data_col="gaul2_code"):
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        H, W = len(self.lats), len(self.lons)
        lon_min, lon_max = self.lons.min(), self.lons.max()
        lat_min, lat_max = self.lats.min(), self.lats.max()
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)

        valid_regions = set(shapefile[region_col].values)

        region_vals = {
            region: da[data_col].sel(region=region).item()
            for region in da.region.values
            if region in valid_regions
        }

        shapes_list = [
            (geom, float(val))
            for geom, region_id in zip(shapefile.geometry, shapefile[region_col])
            if (val := region_vals.get(region_id)) is not None and not np.isnan(float(val))
        ]

        raster_arr = rasterize(
            shapes_list,
            out_shape=(H, W),
            transform=transform,
            fill=-1,
            dtype="int32"
        )

        return xr.DataArray(
            raster_arr,
            coords={"lat": self.lats, "lon": self.lons},
            dims=("lat", "lon"),
            name=data_col
        )

    def pipeline_yearly(self):

        ds_admin = df_to_xarray(
            self.admin_data,
            countries=self.countries,
            fill_value=np.nan,
            extra_cols=self.extra_columns
        )

        time_vars = [v for v in ds_admin.data_vars if "time" in ds_admin[v].dims]
        static_vars = [v for v in ds_admin.data_vars if "time" not in ds_admin[v].dims]

        logger.info(f"Time-varying variables: {time_vars}")
        logger.info(f"Static variables: {static_vars}")

        years = np.unique(ds_admin.time.dt.year.values)

        for year in years:

            ds_year  = ds_admin.sel(time=ds_admin.time.dt.year == year)
            print(ds_year.time.size)

            data_vars = {}

            # ---- Rasterize time-varying ----
            for var in time_vars:
                data_vars[var] = self._rasterize_timeseries(
                    ds_year,
                    self.raster,
                    data_col=var,
                )

            # ---- Rasterize static once ----
            static_rasters = {}
            for var in static_vars:
                static_rasters[var] = self._rasterize_static(
                    ds_admin,
                    self.raster,
                    data_col=var,
                )

            # ---- Broadcast static across time ----
            for var, static_da in static_rasters.items():

                broadcasted = static_da.expand_dims(time=ds_year.time) \
                    .assign_coords(
                        time=ds_year.time,
                        lat=data_vars[time_vars[0]].lat,
                        lon=data_vars[time_vars[0]].lon,
                    ).transpose("time", "lat", "lon")

                data_vars[var] = broadcasted

            yearly_ds = xr.Dataset(data_vars)

            output_path = self.output_file.parent / f"{self.output_file.stem}_{year}.nc"

            yearly_ds.to_netcdf(output_path)

            logger.info(f"Saved {output_path}")

            del yearly_ds  # free memory

    def _rasterize_timeseries(self, ds, shapefile, data_col="dengue_total", region_col="region_id"):
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        H, W = len(self.lats), len(self.lons)
        
            # Direct pixel-value mapping — no vectorization needed
        transform = from_bounds(self.lon_min, self.lat_min, self.lon_max, self.lat_max, W, H) 
        
        valid_regions = set(shapefile[region_col].values)

        rasters = []
        for t in tqdm(range(ds.time.size)):
            week_vals = {
                region: ds[data_col].isel(time=t).sel(region=region).item()
                for region in ds.region.values
                if region in valid_regions
            }

            shapes_list = [
                (geom, float(val))
                for geom, region_id in zip(shapefile.geometry, shapefile[region_col])
                if (val := week_vals.get(region_id)) is not None and not np.isnan(float(val))
            ]

            raster_arr = rasterize(
                shapes_list,
                out_shape=(H, W),
                transform=transform,
                fill=np.nan,
                dtype="float32"
            )
            rasters.append(raster_arr)

        return xr.DataArray(
            np.stack(rasters),
            coords={"time": ds.time.values, "lat": self.lats, "lon": self.lons},
            dims=("time", "lat", "lon"),
            name=data_col
        )
    
    def pipeline(self): 
        da = df_to_xarray(self.admin_data, countries=self.countries, fill_value=np.nan, extra_cols=self.extra_columns) 
        time_vars = [v for v in da.data_vars if "time" in da[v].dims] 
        static_vars = [v for v in da.data_vars if "time" not in da[v].dims] 
        logger.info(f"Time-varying variables: {time_vars}") 
        logger.info(f"Static variables: {static_vars}") 
        # Rasterize time-varying variables 
        data_vars = { var: self._rasterize_timeseries(da, self.raster, data_col=var) for var in time_vars } 
        for var in static_vars: 
            data_vars[var] = self._rasterize_static(da, self.raster, data_col=var) 
        # raster_da = self._rasterize_timeseries(da, raster, res=self.resolution_degree, region_col="region_id",) 
        ds = xr.Dataset(data_vars) 
        logger.info(f"Rasterization complete, saving to {self.output_file}") 
        ds.to_netcdf(self.output_file) 
        return ds
    

# -----------------
# Preprocessing
# -----------------


def df_to_xarray(df, countries: str = None, fill_value=0, extra_cols: list = None):
    """
    extra_cols: list of columns from df to attach as region-level metadata
                e.g. ["adm_2_code", "adm_1_name"]
    """
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["adm_0_name"] = df["adm_0_name"].str.title()
    df["adm_2_name"] = df["adm_2_name"].str.title()

    if countries is not None:
        df = df[df["adm_0_name"].isin(countries)]

    # Collapse duplicates by mean
    agg_cols = ["adm_0_name", "adm_2_name", "start_dt"]
    if extra_cols:
        agg_cols += extra_cols
    temp_df = df.groupby(agg_cols).agg({"dengue_total": "mean"}).reset_index()

    # Create a unique region ID
    temp_df["region_id"] = temp_df["adm_0_name"] + "_" + temp_df["adm_2_name"]
    temp_df["region_id"] = temp_df["region_id"].str.lower()
        
    # Pivot: time x region
    pivoted_0 = pivoted_0 = temp_df.pivot_table(
        index="start_dt",
        columns="region_id",
        values="dengue_total",
        aggfunc="sum"
    )
    pivoted_0.index = pd.to_datetime(pivoted_0.index)
    pivoted_0.index = pivoted_0.index.to_period("W").to_timestamp()

    full_index = pd.date_range(pivoted_0.index.min(), pivoted_0.index.max(), freq="W-MON")
    pivoted = pivoted_0.reindex(full_index, fill_value=fill_value)

    # Build main DataArray
    da = xr.DataArray(
        pivoted.values,
        coords={"time": full_index, "region": pivoted.columns.values},
        dims=("time", "region"),
        name="dengue_total"
    )

    # Build Dataset
    ds = da.to_dataset()

    # Attach extra metadata as region-level coordinates
    if extra_cols:
        # One row per region_id, taking first value (these should be static per region)
        meta = (
            temp_df.groupby("region_id")[extra_cols]
            .first()
            .reindex(pivoted.columns)  # align to region order
        )
        for col in extra_cols:
            ds[col] = xr.DataArray(meta[col].values, dims=["region"], coords={"region": pivoted.columns.values})

    return ds