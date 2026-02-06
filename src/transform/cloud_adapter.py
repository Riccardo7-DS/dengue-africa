import numpy as np
from rasterio.warp import reproject, Resampling
from utils import compute_ndvi
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from dataclasses import dataclass
from utils import get_tiles, get_days_for_tile
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import xarray as xr 
import zarr 
from definitions import DATA_PATH
import torch.nn.functional as F


logger = logging.getLogger(__name__)

@dataclass
class PatchConfig:
    patch_size: int = 256
    stride: int = 256
    min_clear_fraction: float = 0.7 


class MODISCloudDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.xs[idx], dtype=torch.float32),
            torch.tensor(self.ys[idx], dtype=torch.long)
        )

class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.ReLU(),
            nn.Conv2d(bottleneck, dim, 1)
        )

    def forward(self, x):
        return x + self.adapter(x)

BAND_CONFIG = {
    "mod09_250": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "qc": "QC_250m",
    },
    "mod09_500": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "state": "state_1km",
    },
}

def resize_to_multiple_of_patch(x,y, patch_size=14):
    S, C, H, W = x.shape
    # Compute largest multiple of patch_size <= H, W
    H_new = (H // patch_size) * patch_size
    W_new = (W // patch_size) * patch_size

    if H_new != H or W_new != W:
        # Crop x
        x = x[:, :, :H_new, :W_new]
        if y is not None:
            y = y[:, :H_new, :W_new]

    return (x, y) if y is not None else x


class CloudAdapterModel(nn.Module):
    def __init__(self, backbone, feat_dim=384):
        super().__init__()
        self.backbone = backbone
        self.adapter = Adapter(feat_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1)
        )   

    def forward(self, x):
        B, C, H, W = x.shape

        feats = self.backbone.forward_features(x)["x_norm_patchtokens"]
        h = w = int(feats.shape[1] ** 0.5)
        feats = feats.permute(0, 2, 1).reshape(B, -1, h, w)

        feats = self.adapter(feats)
        logits = self.decoder(feats)

        return nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )

class CloudAdapterPipeline():
    def __init__(self,
                client=None,
                bucket=None,
                collection_gq=None,
                collection_ga=None,
                model_name=("facebookresearch/dinov2", "dinov2_vits14"), 
                resampling=Resampling.nearest):
        

        if client:
            self._data_preparation(client, bucket, collection_gq, collection_ga)
        
        self.model_name = model_name
        self._resampling_method = resampling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        
    def _data_preparation(self, client, bucket, collection_gq, collection_ga):
        """Download paired MODIS tiffs."""

        X, y = self._download_paired_modis_tiffs(
            client,
            bucket,
            collection_gq=collection_gq,
            collection_ga=collection_ga,
        )
        self.dataset = MODISCloudDataset(X, y)

    def train(self, 
            epochs=10):
        
        logger.info("Starting training...")
        
        for epoch in range(epochs):
            self.model.train()
            for batch in self.loader:
                if batch is None:
                    continue

                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                x, y = resize_to_multiple_of_patch(x, y, patch_size=14)
                logits = self.model(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            logger.info(f"Epoch {epoch} | Loss {loss.item():.4f}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = nn.functional.softmax(logits, dim=1)
            cloud_mask = probs[0, 1].cpu().numpy()
        return cloud_mask

    def _load_fm(self):
        # Download the model weights

        model_weights = torch.hub.load(
            self.model_name[0],
            self.model_name[1],
            pretrained=True,
            trust_repo=True,
        )

        for p in model_weights.parameters():
            p.requires_grad = False

        self.model = CloudAdapterModel(model_weights)


    def _extract_cloud_mask_500m(self, state_500m):
        """
        state_500m: np.ndarray (H, W), reprojected state_1km
        """
        cloud_state = state_500m & 0b11
        cloud_mask = np.isin(cloud_state, [1, 2]).astype(np.uint8)
        return cloud_mask
    
    

    def _upsample_mask_to_250m(self, mask_500m, src, ref_250m):
        
        mask_250m = np.zeros((ref_250m.height, ref_250m.width), dtype=np.uint16)

        reproject(
            source=mask_500m,
            destination=mask_250m,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_250m.transform,
            dst_crs=ref_250m.crs,
            resampling=self._resampling_method
        )
        return mask_250m

    def _load_mod09gq_input(self, ds):
        red = ds.read(1).astype(np.float32) / 10000.0
        nir = ds.read(2).astype(np.float32) / 10000.0
        ndvi = compute_ndvi(nir, red)
        x = np.stack([red, nir, ndvi], axis=0)
        return x

    def _extract_patches(self, x, y, size=256, stride=256):
        patches_x, patches_y = [], []
        _, H, W = x.shape

        for i in range(0, H - size, stride):
            for j in range(0, W - size, stride):
                patches_x.append(x[:, i:i+size, j:j+size])
                patches_y.append(y[i:i+size, j:j+size])

        return patches_x, patches_y
    


    def process_tile_day(self,
        mod09gq_file,
        mod09ga_state_file,
        out_dir,
        patch_cfg,
        batch_id,
    ):
        X, y = self.build_patch_datacube(
            mod09gq_file,
            mod09ga_state_file,
            patch_cfg,
        )

        if len(X) > 0:
            self._save_patch_batch(X, y, out_dir, batch_id)

    def _save_patch_batch(X, y, out_path, batch_id):
        import numpy as np
        out_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_path / f"patches_{batch_id}.npz",
            X=np.stack(X),
            y=np.stack(y),
        )

    def _build_patch_datacube(
        self,
        mod09gq_path,
        mod09ga_state_path,
        patch_cfg: PatchConfig,
    ):
        """
        Returns:
            X_patches: list[np.ndarray]  (3, H, W)
            y_patches: list[np.ndarray]  (H, W)
        """

        import rasterio
        import numpy as np
        from rasterio.warp import reproject, Resampling

        X_patches, y_patches = [], []

        # --- Load MOD09GQ (250 m) ---
        with rasterio.open(mod09gq_path) as gq:
            X_high = self._load_mod09gq_input(gq)
            H, W = X_high.shape[1], X_high.shape[2]

        # --- Load MOD09GA state_1km (already reprojected to 500 m) ---
        with rasterio.open(mod09ga_state_path) as ga:
            cloud_500m = ga.read(3).astype(np.uint8)

        cloud_250m = self._upsample_mask_to_250m(cloud_500m, ga, gq)

        # --- Sliding window ---
        ps, st = patch_cfg.patch_size, patch_cfg.stride

        for i in range(0, H - ps + 1, st):
            for j in range(0, W - ps + 1, st):

                x_patch = X_high[:, i:i+ps, j:j+ps]
                y_patch = cloud_250m[i:i+ps, j:j+ps]

                # --- Cloud filtering ---
                clear_fraction = 1.0 - y_patch.mean()

                if clear_fraction < patch_cfg.min_clear_fraction:
                    continue

                X_patches.append(x_patch)
                y_patches.append(y_patch)

        if len(X_patches)>0:
            X_patches = np.concatenate(X_patches, axis=0)
            y_patches = np.concatenate(y_patches, axis=0)

        return X_patches, y_patches
    
    def _get_dates_from_filenames(self, files):
        return [
            datetime.strptime(f.split('_')[1].split('.')[0], "%Y%m%d").strftime("%Y-%m-%d")
            for f in files
        ]


    def _download_paired_modis_tiffs(
        self,
        client,
        bucket: str,
        collection_gq: str,
        collection_ga: str,
        local_dir: str=None,
    ):
        """
        Download paired MOD09GQ (250m) and MOD09GA (500m) TIFFs
        for the same tile/day combination.

        Args:
            client: MinIO client
            bucket (str): bucket name
            collection_gq (str): e.g. "modis/MOD09GQ_061/"
            collection_ga (str): e.g. "modis/MOD09GA_061/"
            local_dir (str): local root directory

        """
        if local_dir is None:
            local_dir = "/tmp/modis_paired/"

        X_patches, y_patches = [], []

        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # tiles are assumed identical across products
        tiles_1 = get_tiles(client, bucket, collection_gq)
        tiles_2 = get_tiles(client, bucket, collection_ga)
        tiles = sorted(set(tiles_1) & set(tiles_2))

        summary = {}

        for tile in tqdm(tiles, desc="Processing tiles"):
            tile_dir = local_path / tile
            (tile_dir / "MOD09GQ").mkdir(parents=True, exist_ok=True)
            (tile_dir / "MOD09GA").mkdir(parents=True, exist_ok=True)

            # days available for each product
            days_gq = set(get_days_for_tile(client, bucket, collection_gq, tile))
            days_ga = set(get_days_for_tile(client, bucket, collection_ga, tile))
            days_gq = self._get_dates_from_filenames(days_gq)
            days_ga = self._get_dates_from_filenames(days_ga) 

            common_days = sorted(set(days_ga) & set(days_gq))

            summary[tile] = []

            for day in tqdm(common_days, desc=f"{tile}", leave=False):
                obj_gq = f"{collection_gq}{tile}/tiffs/MOD09GQ_{day.replace("-","")}.tif"
                obj_ga = f"{collection_ga}{tile}/tiffs/MOD09GA_{day.replace("-","")}.tif"

                out_gq = tile_dir / "MOD09GQ" / day
                out_ga = tile_dir / "MOD09GA" / day

                try:
                    client.fget_object(bucket, obj_gq, str(out_gq))
                    client.fget_object(bucket, obj_ga, str(out_ga))
                    X, y = self._build_patch_datacube(out_gq, out_ga, PatchConfig())
                    if len(X) > 0:
                        X_patches.extend(X)
                        y_patches.extend(y)

                except Exception as e:
                    logger.warning(
                        f"Failed {tile} {day}: {e}"
                    )
        logger.info(f"Downloaded {len(X_patches)} patches from paired MODIS data.")

        return X_patches, y_patches
    
    def _extract_single_ndvi_zarr(self, store, date: str, grid_id:str=None) -> xr.DataArray:
        patches = store["patches"]
        if grid_id is None:
            logger.warning("No grid_id provided, using first available grid_id for date.")
            grid_id = list(patches["sur_refl_b01"][date].keys())[0]

        ref1 = patches["sur_refl_b01"][date][grid_id][:, :]
        ref2 = patches["sur_refl_b02"][date][grid_id][:, :]

        ndvi = compute_ndvi(ref1, ref2)
        list_keys = list(patches["sur_refl_b01"][date ].keys())
        return ndvi, list_keys
    
    def _extract_ndvi_all_dates(self, store):
        patches = store["patches"]

        dates = sorted(patches["sur_refl_b01"].keys())
        grid_ids = sorted(patches["sur_refl_b01"][dates[0]].keys())

        ndvi_time = []

        for date in tqdm(dates):
            ndvi_grids = []

            for grid_id in grid_ids:
                try:
                    ref1 = patches["sur_refl_b01"][date][grid_id][:]
                    ref2 = patches["sur_refl_b02"][date][grid_id][:]

                    ndvi = compute_ndvi(ref1, ref2)  # (y, x)
                    ndvi_grids.append(ndvi)
                except KeyError:
                    # If the grid_id is not available for a given date, append a masked array
                    ndvi_grids.append(np.ma.masked_array(np.zeros((ref1.shape[0], ref1.shape[1])), mask=True))  
            # shape → (grid_id, y, x)
            ndvi_grids = np.stack(ndvi_grids, axis=0)
            ndvi_time.append(ndvi_grids)

        # shape → (time, grid_id, y, x)
        ndvi = np.stack(ndvi_time, axis=0)

        # reorder to (grid_id, y, x, time)
        ndvi = np.moveaxis(ndvi, 0, -1).astype(np.float32)

        ds = xr.Dataset(
            data_vars={
                "ndvi": (("grid_id", "y", "x", "time"), ndvi)
            },
            coords={
                "grid_id": grid_ids,
                "time": dates
            }
        )

        return ds
    
    # def extract_modis_cube(self, path, product: str, samples:int=None):

    #     store = zarr.open(path, mode="r")
    #     patches = store["patches"]
    #     cfg = BAND_CONFIG[product]

    #     # --- discover coordinates ---
    #     dates = sorted(patches[cfg["red"]].keys())
    #     grid_ids = sorted(patches[cfg["red"]][dates[0]].keys())

    #     if samples is not None:
    #         grid_ids = grid_ids[:samples]

    #     n_time = len(dates)
    #     n_grid = len(grid_ids)

    #     # infer spatial shape once
    #     sample = patches[cfg["red"]][dates[0]][grid_ids[0]]
    #     y, x = sample.shape

    #     # --- preallocate ---
    #     ndvi = np.empty((n_grid, y, x, n_time), dtype=np.float32)

    #     aux_vars = {
    #         k: np.empty((n_grid, y, x, n_time), dtype=np.uint16)
    #         for k in cfg if k not in ("red", "nir")
    #     }

    #     # --- main loop ---
    #     for t, date in tqdm(enumerate(dates)):
    #         red_grp = patches[cfg["red"]][date]
    #         nir_grp = patches[cfg["nir"]][date]

    #         aux_grps = {
    #             k: patches[v][date]
    #             for k, v in cfg.items()
    #             if k not in ("red", "nir")
    #         }
    #         try:
    #             for g, grid_id in enumerate(grid_ids):
    #                 red = red_grp[grid_id][:]
    #                 nir = nir_grp[grid_id][:]

    #                 ndvi[g, :, :, t] = compute_ndvi(red, nir)

    #                 for aux_name, aux_grp in aux_grps.items():
    #                     aux_vars[aux_name][g, :, :, t] = aux_grp[grid_id][:]
    #         except KeyError:
    #             # If the grid_id is not available for a given date, fill with masked values
    #             ndvi[g, :, :, t] = np.ma.masked_array(np.zeros((y, x)), mask=True)
    #             for aux_name in aux_vars:
    #                 aux_vars[aux_name][g, :, :, t] = np.ma.masked_array(np.zeros((y, x)), mask=True)
    #     # --- build xarray ---
    #     data_vars = {
    #         "ndvi": (("grid_id", "y", "x", "time"), ndvi)
    #     }

    #     for name, arr in aux_vars.items():
    #         data_vars[name] = (("grid_id", "y", "x", "time"), arr)

    #     ds = xr.Dataset(
    #         data_vars=data_vars,
    #         coords={
    #             "grid_id": grid_ids,
    #             "time": dates,
    #         },
    #     )

    #     return ds

    def pipeline(self, 
        path_mod09gq, 
        path_mod09ga,
        epochs=10,
        batch_size=16
        ):


        logger.info("Extracting MODIS cubes...")
        cube_500 = self.extract_modis_cube(path_mod09ga, "mod09_500")
        cube_250 = self.extract_modis_cube(path_mod09gq, "mod09_250")

        logger.info("Building dataset...")
        dataset = MODISZarrPatchDataset(
            cube_250,
            cube_500,
            min_clear_fraction=0.7,
        )

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=safe_collate,
            persistent_workers=True,
        )

        logger.info(f"Loading GFM {self.model_name}...")
        self._load_fm()
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
                lr=1e-4, 
                weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train(epochs=epochs)

    def extract_modis_cube(self, path, product: str, samples: int = None):
        store = zarr.open(path, mode="r")
        patches = store["patches"]
        cfg = BAND_CONFIG[product]

        dates = sorted(patches[cfg["red"]].keys())
        grid_ids = sorted(patches[cfg["red"]][dates[0]].keys())

        if samples is not None:
            grid_ids = grid_ids[:samples]

        return {
            "store": store,
            "patches": patches,
            "cfg": cfg,
            "dates": dates,
            "grid_ids": grid_ids,
            "product": product,
        }


    def _upsample_nn_np(self, arr):
        # arr: (..., y, x)
        return np.repeat(
            np.repeat(arr, 2, axis=-2),
            2, axis=-1
        )

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # create empty tensors with batch=0 so DataLoader doesn't return None
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.default_collate(batch)

def upsample_500_to_250(mask_500):
    return np.repeat(np.repeat(mask_500, 2, axis=0), 2, axis=1)
    
class MODISZarrPatchDataset(Dataset):
    def __init__(
        self,
        cube_250,
        cube_500,
        min_clear_fraction=0.05,
    ):
        self.cube_250 = cube_250
        self.cube_500 = cube_500
        self.min_clear_fraction = min_clear_fraction

        self.dates = cube_250["dates"]
        self.grid_ids = cube_250["grid_ids"]

        self.index = [
            (t, g)
            for t in range(len(self.dates))
            for g in range(len(self.grid_ids))
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        t, g = self.index[idx]
        date = self.dates[t]
        grid_id = self.grid_ids[g]
    
        try:
            # --- MOD09GQ (250 m input) ---
            p250 = self.cube_250["patches"]
            cfg250 = self.cube_250["cfg"]
    
            red = p250[cfg250["red"]][date][grid_id][:].astype(np.float32) / 10000.0
            nir = p250[cfg250["nir"]][date][grid_id][:].astype(np.float32) / 10000.0
            ndvi = compute_ndvi(red, nir)
    
            x = np.stack([red, nir, ndvi], axis=0)  # (3, 512, 512)
    
            # --- MOD09GA (500 m cloud mask → 250 m) ---
            p500 = self.cube_500["patches"]
            cfg500 = self.cube_500["cfg"]
    
            mask_500 = p500[cfg500["state"]][date][grid_id][:]
            mask_250 = upsample_500_to_250(mask_500)
    
            # --- filter cloudy patches ---
            clear_fraction = 1.0 - mask_250.mean()
            if clear_fraction < self.min_clear_fraction:
                return None  # skip patch
    
            return (
                torch.from_numpy(x),
                torch.from_numpy(mask_250).long(),
            )
    
        except KeyError:
            # Patch missing in either cube → skip
            return None


if __name__ == "__main__":
    import xarray as xr
    from definitions import DATA_PATH

    path_modis_500 = DATA_PATH / "modis" / "MOD09GA_dataset.zarr"
    path_modis_250 = DATA_PATH / "modis" / "MOD09GQ_dataset.zarr"

    cloud_pipe = CloudAdapterPipeline(
        model_name=("facebookresearch/dinov2", "dinov2_vits14"))
    
    cloud_pipe.pipeline(
        path_modis_250, 
        path_modis_500,
        batch_size=64,
        )