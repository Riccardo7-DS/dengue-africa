

def run_pipeline():

    from models import DengueConvLSTM
    import torch
    from utils import admin2_aggregate, aggregate_to_admin, load_data
    from torch.nn import MSELoss
    from definitions import DATA_PATH

    data_path = DATA_PATH / "Spatial_extract_V1_3.csv"
    weekly_tabular_data = load_data(data_path, 
        temporal_resolution="Week", 
        spatial_resolution="Admin2", 
        filter_year=2000
    )

    criterion = MSELoss()
    model = DengueConvLSTM(raster_channels=3, tab_features=20, hidden_dim=64, weeks_out=4)
    raster_seq = torch.randn(8, 10, 3, 32, 32)  # (B, T, C, H, W)

    
    # Training loop
    weekly_pred = model(raster_seq, tabular_data)  # (B, weeks, H, W)
    agg_pred = aggregate_to_admin(weekly_pred, admin2_mask)
    loss = criterion(agg_pred, weekly_admin2_cases)  # MSE / Poisson / etc.