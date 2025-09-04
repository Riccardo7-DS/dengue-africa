from definitions import DATA_PATH
from typing import Optional , Literal

def load_data(data_path, 
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