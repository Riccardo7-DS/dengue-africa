import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
# from rasterio.features import rasterize
from tqdm.auto import tqdm
import logging 

logger = logging.getLogger(__name__)

def latin_box(invert:bool=False):
    if invert:
        return [-86.308594,-35.317366, -34.277344, 13.111580]
    else:
        return [-35.317366,-86.308594,13.111580,-34.277344]

african_countries = [ 
    'ANGOLA', 'BENIN', 'BURKINA FASO', 'CABO VERDE', 'CAMEROON', 
    'CENTRAL AFRICAN REPUBLIC', 'CHAD', "COTE D'IVOIRE", 'ERITREA', 
    'ETHIOPIA', 'GHANA', 'GUINEA', 'KENYA', 'MALI', 'MAURITANIA', 
    'MAURITIUS', 'MAYOTTE', 'NIGER', 'REUNION', 'SAO TOME AND PRINCIPE',
    'SENEGAL','SEYCHELLES', 'SUDAN', 'TOGO', 'UNITED REPUBLIC OF TANZANIA']

# Your list of countries/territories
other_places = [
    'AFGHANISTAN', 'AMERICAN SAMOA', 'ANGUILLA', 'ANTIGUA AND BARBUDA',
    'ARGENTINA', 'ARUBA', 'BAHAMAS', 'BARBADOS', 'BELIZE', 'BERMUDA',
    'BOLIVIA', 'BONAIRE, SAINT EUSTATIUS AND SABA', 'BRAZIL',
    'CAMBODIA', 'CAYMAN ISLANDS', 'CHILE', 'COLOMBIA', 'COOK ISLANDS',
    'COSTA RICA', 'CUBA', 'CURACAO', 'DOMINICA', 'DOMINICAN REPUBLIC',
    'ECUADOR', 'EL SALVADOR', 'FIJI', 'FRENCH GUIANA',
    'FRENCH POLYNESIA', 'GRENADA', 'GUADELOUPE', 'GUAM', 'GUATEMALA',
    'GUYANA', 'HAITI', 'HONDURAS', 'JAMAICA', 'JAPAN', 'KIRIBATI',
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 'MALAYSIA', 'MARSHALL ISLANDS',
    'MARTINIQUE', 'MEXICO', 'MICRONESIA (FEDERATED STATES OF)',
    'MONTSERRAT', 'NAURU', 'NEPAL', 'NEW CALEDONIA', 'NICARAGUA',
    'NIUE', 'NORTHERN MARIANA ISLANDS', 'PAKISTAN', 'PALAU', 'PANAMA',
    'PAPUA NEW GUINEA', 'PARAGUAY', 'PERU', 'PHILIPPINES', 'PITCAIRN',
    'PUERTO RICO', 'SAINT BARTHELEMY', 'SAINT KITTS AND NEVIS',
    'SAINT LUCIA', 'SAINT MARTIN', 'SAINT VINCENT AND THE GRENADINES',
    'SAMOA', 'SAUDI ARABIA', 'SINGAPORE', 'SINT MAARTEN',
    'SOLOMON ISLANDS', 'SRI LANKA', 'SUDAN', 'SURINAME', 'TAIWAN',
    'TOKELAU', 'TONGA', 'TRINIDAD AND TOBAGO',
    'TURKS AND CAICOS ISLANDS', 'TUVALU', 'UNITED STATES OF AMERICA',
    'URUGUAY', 'VANUATU', 'VENEZUELA', 'VIET NAM',
    'VIRGIN ISLANDS (UK)', 'VIRGIN ISLANDS (US)', 'WALLIS AND FUTUNA',
    'YEMEN'
]

def rasterize_timeseries(da:xr.DataArray, shapefile:gpd.GeoDataFrame, region_col:str="region_id", res:float=0.05):
    """
    Convert xarray DataArray (time x region) into raster cube (time x H x W).
    
    da: xarray.DataArray with dims (time, region)
    shapefile: GeoDataFrame with polygons for admin2
    region_col: column in shapefile matching da.region
    res: grid resolution in degrees (~0.05 ≈ 5km)
    """
    # Compute bounds from shapefile
    lat_min, lat_max, lon_min, lon_max = shapefile.total_bounds[[1,3,0,2]]
    lats = np.arange(lat_max, lat_min, -res)
    lons = np.arange(lon_min, lon_max, res)
    H, W = len(lats), len(lons)
    
    # Create transform
    transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)
    
    rasters = []
    for t in tqdm(range(da.time.size)):
        week_vals = {region: da.isel(time=t).sel(region=region).item() 
                     for region in da.region.values}
        # Filter out regions not in shapefile
        week_vals = {k: v for k, v in week_vals.items() if k in shapefile[region_col].values}
        
        shapes = [(geom, val) for geom, val in zip(shapefile.geometry, shapefile[region_col].map(week_vals.get)) 
                  if not np.isnan(val)]
        
        raster = rasterize(
            shapes,
            out_shape=(H, W),
            transform=transform,
            fill=np.nan,
            dtype='float32'
        )
        rasters.append(raster)
    
    # Stack into xarray
    raster_da = xr.DataArray(
        np.stack(rasters),
        coords={"time": da.time.values, "lat": lats, "lon": lons},
        dims=("time", "lat", "lon"),
        name="dengue_raster"
    )
    return raster_da



# -----------------
# Preprocessing
# -----------------





def df_to_xarray(df, freq="W-MON", countries:str=None, fill_value=0):

    # Convert to datetime
    df["start_dt"] = pd.to_datetime(df["start_dt"])

    df["adm_0_name"] = df["adm_0_name"].str.title()  # Standardize country names
    df["adm_2_name"] = df["adm_2_name"].str.title()  # Standardize Admin2 names

    if countries is not None:
        df = df[df["adm_0_name"].isin(countries)]
            
    # Collapse duplicates by mean
    temp_df = df.groupby(["adm_0_name", "adm_2_name", "start_dt"]).agg({"dengue_total":"mean"}).reset_index()
    
    # Create a unique region ID
    temp_df["region_id"] = temp_df["adm_0_name"] + "_" + temp_df["adm_2_name"]

    # Pivot directly: time x region
    pivoted_0 = temp_df.pivot(index="start_dt", columns="region_id", values="dengue_total")

    pivoted_0.index = pd.to_datetime(pivoted_0.index)
    pivoted_0.index = pivoted_0.index.to_period("W").to_timestamp()

    # Reindex to full weekly timeline
    full_index = pd.date_range(
        pivoted_0.index.min(), pivoted_0.index.max(), freq="W-MON"
    )

    pivoted = pivoted_0.reindex(full_index, fill_value=fill_value)
    
    # Convert to xarray
    da = xr.DataArray(
        pivoted.values,
        coords={"time": full_index, "region": pivoted.columns.values},
        dims=("time", "region"),
        name="dengue_total"
    )
    return da

def build_timeseries(da, n_weeks=12):

    """
    Convert xarray DataArray (time x region) into dataset [S, T, F].
    Drops windows where all values are NaN.
    """
    values = da.values  # shape (time, region)
    n_time, n_regions = values.shape
    
    sequences = []
    for r in tqdm(range(n_regions)):
        series = values[:, r]  # time series for region r
        for t in range(n_time - n_weeks + 1):
            window = series[t : t + n_weeks]
            if np.isnan(window).all():   # 👈 skip windows with only NaNs
                continue
            sequences.append(window)
    
    # Convert to numpy array: [S, T, F]
    sequences = np.array(sequences)
    sequences = sequences[..., np.newaxis]  # add feature dimension
    return sequences

def process_gdf(zip_path, countries:list=None):
    # Read the shapefile directly from the zip
    gdf = gpd.read_file(f"zip://{zip_path}")

    # Optional: standardize column names to match your table
    gdf = gdf.rename(columns={
        "gaul0_name": "adm_0_name",
        "gaul1_name": "adm_1_name",
        "gaul2_name": "adm_2_name",
        "GAUL_CODE": "GAUL_code_2"
    })

    gdf["region_id"] = gdf["adm_0_name"].str.title() + "_" + gdf["adm_2_name"].str.title()

    if countries is not None:
        gdf = gdf[gdf["adm_0_name"].str.title().isin(countries)]
        return gdf.reset_index(drop=True)
    else:
        return gdf



def convert_data_to_yearly_presence(data:pd.DataFrame, filter_year:int=2000, spatial_resolution:str="Admin2"):
    

    data["start_dt"] = pd.to_datetime(data["calendar_start_date"])
    data["year"] = data["start_dt"].dt.year

    data = data.loc[data["S_res"]==spatial_resolution]
    data = data.loc[data["year"] >= filter_year]  # <- filter years if needed

    # Build presence/absence table
    presence = (
        data.groupby(["adm_0_name", "year"])
        .size()                # count rows
        .reset_index(name="count")
    )

    presence["exists"] = (presence["count"] > 0).astype(int)

    # Get full year range
    all_years = pd.Series(range(data["year"].min(), data["year"].max() + 1))

    # Pivot
    heatmap_data = (
        presence.pivot(index="adm_0_name", columns="year", values="exists")
        .reindex(columns=all_years, fill_value=0)  # <- add missing years
        .fillna(0)
        .T
    )

    names = data["adm_0_name"].unique()
    
    return heatmap_data, names


def admin2_aggregate(pred_weekly, admin2_mask):
    B, weeks, H, W = pred_weekly.shape
    num_admin2 = admin2_mask.shape[0]
    pred_flat = pred_weekly.view(B, weeks, H*W)
    mask_flat = admin2_mask.view(num_admin2, H*W).float()
    agg = torch.einsum("bwh,nh->bwn", pred_flat, mask_flat)  # (B, weeks, num_admin2)
    return agg

def aggregate_to_admin(pred, admin2_mask):
    # pred: (B, weeks, H, W)
    # admin2_mask: (num_admin2, H, W)
    B, weeks, H, W = pred.shape
    num_admin2 = admin2_mask.shape[0]
    pred_flat = pred.view(B, weeks, H*W)
    mask_flat = admin2_mask.view(num_admin2, H*W).float()
    agg = torch.einsum("bwh,nh->bwn", pred_flat, mask_flat)  # (B, weeks, num_admin2)
    return agg


def init_logging(log_file=None, verbose=False):
    import os
    # Determine the logging level
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Define the logging format
    formatter = "%(asctime)s : %(levelname)s : [%(filename)s:%(lineno)s - %(funcName)s()] : %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # Setup basic configuration for logging
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.FileHandler(log_file, "w"),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.StreamHandler()
            ]
        )

    logger = logging.getLogger()
    return logger
