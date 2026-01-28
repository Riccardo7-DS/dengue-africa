import os
from utils import get_tiles, get_days_for_tile
from dotenv import load_dotenv
from utils import minio_client

load_dotenv()

# --- Initialize MinIO client ---
client = minio_client()
bucket = os.getenv("MINIO_BUCKET")

# -------------------------
# Example usage
# -------------------------
collection_path_ga = "modis/MOD09GA_061/"
collection_path_gq = "modis/MOD09GQ_061/"

# tiles = get_tiles(client, bucket, collection_path_ga)
# print("Tiles:", tiles)

# tile = tiles[0]
# days = get_days_for_tile(client, bucket, collection_path_ga, tile)
# print(f"Days available for {tile}:", days)

from transform import CloudAdapterPipeline

CloudAdapterPipeline(client, 
    bucket, 
    collection_ga=collection_path_ga, 
    collection_gq=collection_path_gq
)