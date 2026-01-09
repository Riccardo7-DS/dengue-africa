from definitions import DATA_PATH
from typing import Optional , Literal
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

load_dotenv()

def load_land_mask():
    import ee 
    return ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select('water_mask')

def load_admin_data(data_path, 
            temporal_resolution:Literal[None, "Week", "Month"]=None, 
            spatial_resolution:Literal[None, "Admin0", "Admin1", "Admin2"]=None, 
            filter_year:int=2000
    ):

    import pandas as pd
    data = pd.read_csv(data_path)
    data["start_dt"] = pd.to_datetime(data["calendar_start_date"])
    data["year"] = data["start_dt"].dt.year

    if spatial_resolution is not None:
        if spatial_resolution not in ["Admin0", "Admin1", "Admin2"]:
            raise ValueError("spatial_resolution must be one of None, 'Admin0', 'Admin1', 'Admin2'")
        data = data.loc[data["S_res"]==spatial_resolution]

    if temporal_resolution is not None:
        if temporal_resolution not in ["Week", "Month"]:
            raise ValueError("temporal_resolution must be one of None, 'Week', 'Month'")    
        data = data.loc[data["T_res"]==temporal_resolution]
        
    if filter_year is not None:            
        data = data.loc[data["year"] >= filter_year]  # <- filter years if needed
    return data.reset_index(drop=True)


def bbox_size_km(bbox, pixel_size_m):
    import math
    """
    Compute the approximate width and height of a bounding box in km,
    given pixel size in meters, accounting for latitude distortion.
    
    Parameters
    ----------
    bbox : list or tuple
        [lon_min, lat_min, lon_max, lat_max]
    pixel_size_m : float
        Pixel size in meters
    
    Returns
    -------
    width_km, height_km : float
        Approximate size of bbox in km
    n_pixels_x, n_pixels_y : int
        Number of pixels along x and y at the given pixel size
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    
    # Latitude/longitude extents
    width_deg = lon_max - lon_min
    height_deg = lat_max - lat_min
    
    # Latitude center for accurate longitude -> meters conversion
    lat_center = (lat_min + lat_max)/2
    
    # Approximate conversion: 1° latitude ~ 111.319 km
    height_m = height_deg * 111_319
    # 1° longitude ~ 111.319 km * cos(lat)
    width_m = width_deg * 111_319 * math.cos(math.radians(lat_center))
    
    # Number of pixels
    n_pixels_x = int(width_m / pixel_size_m)
    n_pixels_y = int(height_m / pixel_size_m)
    
    # Size in km
    width_km = n_pixels_x * pixel_size_m / 1000
    height_km = n_pixels_y * pixel_size_m / 1000

    print("Width (km):", width_km)
    print("Height (km):", height_km)
    print("Pixels X:", n_pixels_x)
    print("Pixels Y:", n_pixels_y)
    
    return width_km, height_km, n_pixels_x, n_pixels_y

def tile_has_land(bbox, scale=250):
    
    import ee 

    land_mask = load_land_mask()

    """Return True if the bounding box contains land pixels."""
    lon_min, lat_min, lon_max, lat_max = bbox
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    
    water_stats = land_mask.reduceRegion(
        reducer=ee.Reducer.minMax(),  # check min and max
        geometry=region,
        scale=scale,  # MODIS resolution
        maxPixels=1e9
    ).getInfo()
    
    if water_stats is None:
        # GEE could not compute: treat as ocean (or skip)
        return False
    
    water_min = water_stats.get('water_mask_min')
    water_max = water_stats.get('water_mask_max')
    
    if water_min is None or water_max is None:
        return False
    
    # Return True if there is at least one land pixel
    return water_min < 1


def tile_has_min_land_fraction(bbox, min_percentage=10, scale=500, debug=False):
    """
    Return True if at least `min_percentage` of the pixels in the bbox are land.
    
    Parameters
    ----------
    bbox : list
        [lon_min, lat_min, lon_max, lat_max]
    min_percentage : float
        Minimum percentage of land required (0-100)
    scale : int
        Pixel size in meters (MODIS resolution)
    debug : bool
        If True, print debug info about what's being computed
    """
    import ee
    ee.Authenticate()
    ee.Initialize(project=os.environ["EE_PROJECT"])

    land_mask = load_land_mask()

    lon_min, lat_min, lon_max, lat_max = bbox
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    
    if debug:
        print(f"\n=== Debugging tile_has_min_land_fraction ===")
        print(f"Bbox: {bbox}")
        print(f"Looking for min_percentage >= {min_percentage}%")
    
    # Convert water_mask -> land_mask: 0=land -> 1, 1=water -> 0
    # This creates a binary image where 1 represents land pixels
    land_binary = land_mask.eq(0).toFloat()
    
    # Compute total pixels and land pixels
    stats = land_binary.reduceRegion(
        reducer=ee.Reducer.sum().combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        ),
        geometry=region,
        scale=scale,
        maxPixels=1e9
    ).getInfo()
    
    if debug:
        print(f"Raw stats dict: {stats}")
    
    if stats is None:
        if debug:
            print("Stats is None, returning False")
        return False
    
    # Try to find the keys in the returned dictionary
    land_sum = None
    total_count = None
    
    # List of possible key names to try
    possible_sum_keys = ['constant_sum', 'water_mask_sum', 'sum', 'constant']
    possible_count_keys = ['constant_count', 'water_mask_count', 'count']
    
    for key in possible_sum_keys:
        if key in stats:
            land_sum = stats[key]
            if debug:
                print(f"Found land_sum with key '{key}': {land_sum}")
            break
    
    for key in possible_count_keys:
        if key in stats:
            total_count = stats[key]
            if debug:
                print(f"Found total_count with key '{key}': {total_count}")
            break
    
    if land_sum is None or total_count is None or total_count == 0:
        if debug:
            print(f"Missing values: land_sum={land_sum}, total_count={total_count}")
        return False
    
    land_percentage = (land_sum / total_count) * 100  # percentage of land
    
    if debug:
        print(f"Computed: {land_sum} land pixels / {total_count} total = {land_percentage:.2f}%")
        print(f"Result: {land_percentage >= min_percentage}")
    
    return land_percentage >= min_percentage

def plot_boxes_on_map(bboxes):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot bounding boxes
    for bbox in bboxes:
        lon_min, lat_min, lon_max, lat_max = bbox
        ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                [lat_min, lat_min, lat_max, lat_max, lat_min],
                color='red', linewidth=2, transform=ccrs.PlateCarree())

    # ax.set_extent([-90, -30, -40, 20], crs=ccrs.PlateCarree())
    ax.set_extent([-70, 70, -70, 70], crs=ccrs.PlateCarree())

    plt.title("Bounding Boxes")
    plt.savefig("bounding_boxes_map.png", dpi=300)
    bbox = bboxes[0]
    bbox_size_km(bbox, pixel_size_m=250)