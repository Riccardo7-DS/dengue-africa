from utils import DengueRasterizer
import logging
logger = logging.getLogger(__name__)

rasterizer = DengueRasterizer(
    reproject_geopandas=True, 
    extra_columns=["FAO_GAUL_code"],
    # countries=["Peru"]
)
da = rasterizer.pipeline_yearly()