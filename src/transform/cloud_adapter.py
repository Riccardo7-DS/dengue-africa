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
                client,
                bucket,
                collection_gq,
                collection_ga,
                model_name=("facebookresearch/dinov2", "dinov2_vits14"), 
                resampling=Resampling.nearest):
        

        self._data_preparation(client, bucket, collection_gq, collection_ga)
        self.model_name = model_name
        self._resampling_method = resampling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_fm()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
                lr=1e-4, 
                weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
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
            epochs=10, 
            batch_size=16):
        
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

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


    def _extract_cloud_mask_500m(self, state_500m):
        """
        state_500m: np.ndarray (H, W), reprojected state_1km
        """
        cloud_state = state_500m & 0b11
        cloud_mask = np.isin(cloud_state, [1, 2]).astype(np.uint8)
        return cloud_mask

    def _upsample_mask_to_250m(self, mask_500m, src, ref_250m):
        mask_250m = np.zeros((ref_250m.height, ref_250m.width), dtype=np.uint8)

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
    
    def _load_fm(self):
        model_weights = torch.hub.load(
            self.model_name[0],
            self.model_name[1], pretrained=True
        )

        for p in model_weights.parameters():
            p.requires_grad = False

        self.model = CloudAdapterModel(model_weights)

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

            # upsample to 250m
            cloud_250m = np.zeros((H, W), dtype=np.uint8)

            reproject(
                source=cloud_500m,
                destination=cloud_250m,
                src_transform=ga.transform,
                src_crs=ga.crs,
                dst_transform=gq.transform,
                dst_crs=gq.crs,
                resampling=Resampling.nearest,
            )

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