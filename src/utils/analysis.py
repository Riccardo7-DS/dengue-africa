import xarray as xr
import numpy as np
import pandas as pd
from definitions import DATA_PATH
import logging
from utils.general import process_gdf
from tqdm import tqdm
import os 

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


from dataclasses import dataclass, field
from typing import Optional, Callable
import os
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


@dataclass
class RasterizerConfig:
    """
    Configuration for the flexible rasterizer.
    
    Parameters
    ----------
    resolution_degree : float
        Spatial resolution in degrees. Defaults to ~5km (FAO standard).
    output_file : str or Path
        Path to save the output NetCDF file.
    time_col : str
        Column name for the time dimension.
    geometry_col : str
        Column name for the geometry.
    region_col : str
        Column used to uniquely identify each region (used as the join key).
    value_cols : list[str]
        Columns to rasterize as data variables.
    static_cols : list[str]
        Columns that are time-invariant (rasterized once and broadcast).
    region_id_builder : Callable[[GeoDataFrame], Series]
        Optional function to build a composite region_id from the GeoDataFrame.
        Defaults to using `region_col` directly.
    bounds : tuple[float, float, float, float] | None
        (lat_min, lat_max, lon_min, lon_max). Inferred from geometry if None.
    save_yearly : bool
        If True, saves one NetCDF per year instead of a single file.
    """
    resolution_degree: float = 0.041666666666666664
    output_file: str = "output_raster.nc"
    time_col: str = "time"
    geometry_col: str = "geometry"
    region_col: str = "region_id"
    value_cols: list = field(default_factory=list)
    static_cols: list = field(default_factory=list)
    region_id_builder: Optional[Callable] = None
    bounds: Optional[tuple] = None
    save_yearly: bool = False


class FlexibleRasterizer:
    """
    Converts a (Geo)DataFrame with polygon geometries and time series data
    into an xarray raster Dataset saved as NetCDF.

    Example — new Buenos Aires dataset
    ------------------------------------
    config = RasterizerConfig(
        output_file="ba_cases.nc",
        time_col="time",
        geometry_col="geometry",
        region_col="gid",
        value_cols=["cantidad_casos"],
        static_cols=["in1"],
        region_id_builder=lambda gdf: gdf["gid"].astype(str),
        save_yearly=True,
    )
    rasterizer = FlexibleRasterizer(df=your_geodataframe, config=config)
    ds = rasterizer.pipeline()

    Example — original dengue pattern (geometry from external shapefile)
    ---------------------------------------------------------------------
    config = RasterizerConfig(
        output_file="dengue_cases.nc",
        region_col="region_id",
        value_cols=["dengue_total"],
        static_cols=["gaul2_code"],
        region_id_builder=lambda gdf: (
            gdf["adm_0_name"] + "_" + gdf["adm_2_name"]
        ).str.lower(),
    )
    rasterizer = FlexibleRasterizer(df=admin_df, config=config, shapefile=gaul_gdf)
    ds = rasterizer.pipeline()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: RasterizerConfig,
        shapefile: Optional[gpd.GeoDataFrame] = None,
    ):
        self.config = config
        self.df = df.copy()

        # ── Geometry source ───────────────────────────────────────────────
        # Priority: explicit shapefile arg > geometry embedded in df
        if shapefile is not None:
            self.gdf = shapefile.copy()
        elif config.geometry_col in df.columns:
            self.gdf = gpd.GeoDataFrame(df, geometry=config.geometry_col)
            # Keep one row per region (geometry is static)
            self.gdf = self.gdf.drop_duplicates(subset=[config.region_col])
        else:
            raise ValueError(
                "No geometry source found. Either pass a `shapefile` GeoDataFrame "
                f"or ensure `df` has a '{config.geometry_col}' column."
            )

        # ── Build region_id ───────────────────────────────────────────────
        if config.region_id_builder is not None:
            self.gdf[config.region_col] = config.region_id_builder(self.gdf)
            self.df[config.region_col] = config.region_id_builder(
                gpd.GeoDataFrame(self.df, geometry=config.geometry_col)
                if config.geometry_col in self.df.columns
                else self.df
            )
        else:
            # Assume region_col already exists in both df and gdf
            if config.region_col not in self.gdf.columns:
                raise ValueError(f"region_col '{config.region_col}' not found in GDF.")

        # ── Spatial grid ──────────────────────────────────────────────────
        if config.bounds is not None:
            lat_min, lat_max, lon_min, lon_max = config.bounds
        else:
            lon_min, lat_min, lon_max, lat_max = self.gdf.total_bounds

        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max
        res = config.resolution_degree
        self.lats = np.arange(lat_max, lat_min, -res)
        self.lons = np.arange(lon_min, lon_max, res)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_transform(self):
        from rasterio.transform import from_bounds
        H, W = len(self.lats), len(self.lons)
        return from_bounds(
            self.lon_min, self.lat_min,
            self.lon_max, self.lat_max,
            W, H
        ), H, W

    def _build_xarray(self) -> xr.Dataset:
        """
        Pivot the flat DataFrame into an xr.Dataset with dims (region, time).
        Only value_cols and static_cols are included.
        """
        cfg = self.config
        time_col = cfg.time_col
        region_col = cfg.region_col
        all_value_cols = cfg.value_cols + cfg.static_cols

        # Coerce time column
        self.df[time_col] = pd.to_datetime(self.df[time_col])

        ds_dict = {}
        regions = self.df[region_col].unique()
        times = np.sort(self.df[time_col].unique())

        for col in cfg.value_cols:
            pivot = (
                self.df
                .pivot_table(index=region_col, columns=time_col, values=col, aggfunc="sum")
                .reindex(index=regions, columns=times)
            )
            ds_dict[col] = xr.DataArray(
                pivot.values,
                coords={"region": regions, "time": times},
                dims=("region", "time"),
                name=col,
            )

        for col in cfg.static_cols:
            # Take the first (static) value per region
            static_vals = self.df.groupby(region_col)[col].first().reindex(regions)
            ds_dict[col] = xr.DataArray(
                static_vals.values,
                coords={"region": regions},
                dims=("region",),
                name=col,
            )

        return xr.Dataset(ds_dict)

    def _rasterize_timeseries(self, ds: xr.Dataset, data_col: str) -> xr.DataArray:
        from rasterio.features import rasterize

        transform, H, W = self._get_transform()
        cfg = self.config
        valid_regions = set(self.gdf[cfg.region_col].values)

        rasters = []
        for t in tqdm(range(ds.time.size), desc=f"Rasterizing {data_col}"):
            region_vals = {
                region: ds[data_col].isel(time=t).sel(region=region).item()
                for region in ds.region.values
                if region in valid_regions
            }
            shapes_list = [
                (geom, float(val))
                for geom, rid in zip(self.gdf.geometry, self.gdf[cfg.region_col])
                if (val := region_vals.get(rid)) is not None and not np.isnan(float(val))
            ]
            raster_arr = rasterize(
                shapes_list,
                out_shape=(H, W),
                transform=transform,
                fill=np.nan,
                dtype="float32",
            )
            rasters.append(raster_arr)

        return xr.DataArray(
            np.stack(rasters),
            coords={"time": ds.time.values, "lat": self.lats, "lon": self.lons},
            dims=("time", "lat", "lon"),
            name=data_col,
        )

    def _rasterize_static(self, ds: xr.Dataset, data_col: str) -> xr.DataArray:
        from rasterio.features import rasterize

        transform, H, W = self._get_transform()
        cfg = self.config
        valid_regions = set(self.gdf[cfg.region_col].values)

        region_vals = {
            region: ds[data_col].sel(region=region).item()
            for region in ds.region.values
            if region in valid_regions
        }
        shapes_list = [
            (geom, float(val))
            for geom, rid in zip(self.gdf.geometry, self.gdf[cfg.region_col])
            if (val := region_vals.get(rid)) is not None and not np.isnan(float(val))
        ]
        raster_arr = rasterize(
            shapes_list,
            out_shape=(H, W),
            transform=transform,
            fill=-1,
            dtype="int32",
        )
        return xr.DataArray(
            raster_arr,
            coords={"lat": self.lats, "lon": self.lons},
            dims=("lat", "lon"),
            name=data_col,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def pipeline(self) -> xr.Dataset:
        """Run rasterization over all time steps and save to NetCDF."""
        ds = self._build_xarray()

        if self.config.save_yearly:
            return self._pipeline_yearly(ds)

        return self._pipeline_full(ds)

    def _pipeline_full(self, ds: xr.Dataset) -> xr.Dataset:
        data_vars = {}

        for col in self.config.value_cols:
            data_vars[col] = self._rasterize_timeseries(ds, col)

        for col in self.config.static_cols:
            static_da = self._rasterize_static(ds, col)
            # Broadcast static var across time
            time_ref = data_vars[self.config.value_cols[0]]
            data_vars[col] = (
                static_da
                .expand_dims(time=time_ref.time)
                .assign_coords(time=time_ref.time, lat=self.lats, lon=self.lons)
                .transpose("time", "lat", "lon")
            )

        out_ds = xr.Dataset(data_vars)
        out_ds.to_netcdf(self.config.output_file)
        print(f"Saved → {self.config.output_file}")
        return out_ds

    def _pipeline_yearly(self, ds: xr.Dataset) -> dict:
        """Save one NetCDF per year."""
        years = np.unique(pd.to_datetime(ds.time.values).year)
        results = {}

        for year in years:
            ds_year = ds.sel(time=ds.time.dt.year == year)
            data_vars = {}

            for col in self.config.value_cols:
                data_vars[col] = self._rasterize_timeseries(ds_year, col)

            for col in self.config.static_cols:
                static_da = self._rasterize_static(ds, col)  # use full ds for static
                time_ref = data_vars[self.config.value_cols[0]]
                data_vars[col] = (
                    static_da
                    .expand_dims(time=time_ref.time)
                    .assign_coords(time=time_ref.time, lat=self.lats, lon=self.lons)
                    .transpose("time", "lat", "lon")
                )

            yearly_ds = xr.Dataset(data_vars)
            output_path = str(self.config.output_file).replace(".nc", f"_{year}.nc")
            yearly_ds.to_netcdf(output_path)
            print(f"Saved → {output_path}")
            results[year] = yearly_ds
            del yearly_ds

        return results
    

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

