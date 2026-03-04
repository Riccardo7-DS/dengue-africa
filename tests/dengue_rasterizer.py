from definitions import DATA_PATH
from utils import FlexibleRasterizer, RasterizerConfig, load_admin_data
import logging
logger = logging.getLogger(__name__)

config = RasterizerConfig(static_cols=["FAO_GAUL_code"],
                          save_yearly=False,
                          )

path = DATA_PATH / "Spatial_extract_V1_3.csv"
df =load_admin_data(path, temporal_resolution="Week", spatial_resolution="Admin2")
rasterizer = FlexibleRasterizer(df, config=config)
ds = rasterizer.pipeline()
