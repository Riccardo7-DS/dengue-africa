from definitions import DATA_PATH, ROOT_DIR
import matplotlib.pyplot as plt
from transform import CloudAdapterPipeline, extract_modis_cube, extract_image
path_modis_500 = DATA_PATH / "modis/data/modis" / "MOD09GA_dataset.zarr"
path_modis_250 = DATA_PATH / "modis/data/modis" / "MOD09GQ_dataset.zarr"

test_time = 10
test_patch = 50

cloud_pipe = CloudAdapterPipeline(
        model_name=("facebookresearch/dinov2", "dinov2_vits14"))

cloud_pipe.load_checkpoint(ROOT_DIR / "checkpoints/best.pt")
cube_250 = extract_modis_cube(path_modis_250, "mod09_250")
test_array = extract_image(path_modis_250, "mod09_250", test_time, test_patch)
cloud_pipe.predict(test_array)