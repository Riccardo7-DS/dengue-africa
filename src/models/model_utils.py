"""
Functions for DL models
"""

import math
import os 
import logging
import time
import numpy as np 
import torch
from torch.utils.data import Subset
import pandas as pd
from torchgeo.samplers import GridGeoSampler
import xarray as xr
import torch
import torch.nn.functional as F
import hashlib
import pickle
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import BoundingBox as TorchGeoBoundingBox
from torchgeo.samplers import GridGeoSampler
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
from typing import List, Optional, Literal

logger = logging.getLogger(__name__)

def export_batches(batch_idx, epoch, batch, run_dir,
                   n_export_batches=3, export_epoch=1):
    """
    Export the first n batches of training data for inspection.

    Args:
        batch_idx: current batch index
        epoch: current epoch
        batch: tuple from dataloader (x_high, x_med, x_static, y_batch)
        run_dir: Path to run directory
        n_export_batches: how many batches to export
        export_epoch: which epoch to export from
    """
    if epoch != export_epoch or batch_idx >= n_export_batches:
        return

    x_high, x_med, x_static, x_cond, y_batch = batch

    export_dir = run_dir / "batch_exports" / f"epoch_{epoch:03d}"
    export_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        export_dir / f"batch_{batch_idx:03d}.npy",
        {
            "x_high": x_high.cpu().numpy(),
            "x_med": x_med.cpu().numpy(),
            "x_static": x_static.cpu().numpy(),
            "x_cond": x_cond.cpu().numpy(),
            "y_batch": y_batch.cpu().numpy(),
        }
    )

    logger.info(f"Exported batch {batch_idx} → {export_dir}")

def nan_checks_replace(datasets, replace_nan=0.0):
    processed = []
    for i, data in enumerate(datasets):
        data = data.float()

        # Single-pass sentinel + nan + inf replacement
        # Handles -3.4e38 ESRI nodata, NaN, and Inf in one operation
        sentinel_mask = data.abs() > 3.3e38
        if sentinel_mask.any():  # .any() stays on GPU, no .item() sync
            data.masked_fill_(sentinel_mask, replace_nan)

        # One call handles all remaining NaN/Inf
        torch.nan_to_num_(data, nan=replace_nan, posinf=replace_nan, neginf=replace_nan)

        processed.append(data)

    return processed

def standardize_tensor(datasets, replace_nan=0.0):
    processed = []

    for i, data in enumerate(datasets):
        ndim = data.ndim

        if ndim == 5:      # [B, T, C, H, W]
            channel_dim = 2
            reduce_dims = (0, 1, 3, 4)
        elif ndim == 4:    # [B, C, H, W]
            channel_dim = 1
            reduce_dims = (0, 2, 3)
        else:              # [B, D] or anything else — global normalization
            channel_dim = None
            reduce_dims = None

        if channel_dim is not None:
            mean = data.mean(dim=reduce_dims, keepdim=True)
            std = data.std(dim=reduce_dims, keepdim=True)

            safe_std = torch.where(std > 1e-6, std, torch.ones_like(std))
            data = (data - mean) / safe_std

            zero_var = (std <= 1e-6).squeeze()
            if zero_var.any():
                logger.debug(
                    f"Dataset {i}: channels "
                    f"{zero_var.nonzero().squeeze().tolist()} have zero variance"
                )
        else:
            mean = data.mean()
            std = data.std()
            if std > 1e-6:
                data = (data - mean) / std

        # Safety check after normalization
        if torch.isnan(data).any():
            logger.warning(f"Dataset {i}: NaNs introduced during normalization, replacing")
            data = torch.nan_to_num(data, nan=replace_nan)

        processed.append(data)

    return processed

def debug_nan(tensors, names):
    for t, name in zip(tensors, names):
        nan_count = torch.isnan(t).sum().item()
        inf_count = torch.isinf(t).sum().item()
        
        if nan_count > 0:
            valid_vals = t[~torch.isnan(t)]
            if len(valid_vals) > 0:
                logger.debug(f"NaN detected in {name} ({nan_count}/{t.numel()} = {100*nan_count/t.numel():.1f}%) — valid min: {valid_vals.min()}, max: {valid_vals.max()}, mean: {valid_vals.mean()}")
            else:
                logger.warning(f"NaN detected in {name} — ALL VALUES ARE NAN!")
        
        if inf_count > 0:
            valid_vals = t[~torch.isinf(t)]
            if len(valid_vals) > 0:
                logger.debug(f"Inf detected in {name} ({inf_count}/{t.numel()} = {100*inf_count/t.numel():.1f}%) — valid min: {valid_vals.min()}, max: {valid_vals.max()}, mean: {valid_vals.mean()}")
            else:
                logger.warning(f"Inf detected in {name} — ALL VALUES ARE INF!")


def rolling_split(dataset, train_end, horizon_days=365):
    times = dataset.get_target_times()

    val_start = train_end + pd.Timedelta(days=1)
    val_end = train_end + pd.Timedelta(days=horizon_days)

    train_idx = np.where(times <= train_end)[0]
    val_idx = np.where((times >= val_start) & (times <= val_end))[0]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def temporal_split(dataset, train_end, val_end):
    times = dataset.get_target_times()

    train_idx = np.where(times <= train_end)[0]
    val_idx = np.where((times > train_end) & (times <= val_end))[0]
    test_idx = np.where(times > val_end)[0]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )

class MetricsRecorder:
    def __init__(self):
        self.train_mape = []
        self.train_rmse = []
        self.train_loss = []
        self.val_mape = []
        self.val_rmse = []
        self.val_loss = []
        self.lr = []
        self.epoch = []

    def add_train_metrics(self, metrics, epoch):
        self.train_mape.append(np.mean(metrics['mape']))
        self.train_rmse.append(np.mean(metrics["rmse"]))
        self.train_loss.append(np.mean(metrics["loss"]))
        self.epoch.append(epoch)

    def add_val_metrics(self, metrics):
        self.val_mape.append(np.mean(metrics['mape']))
        self.val_rmse.append(np.mean(metrics["rmse"]))
        self.val_loss.append(np.mean(metrics["loss"]))
        self.lr.append(metrics["lr"][-1])

def update_tensorboard_scalars(writer, recorder:MetricsRecorder):
    writer.add_scalars('loss', {'train': recorder.train_loss[-1]}, recorder.epoch[-1])
    #writer.add_scalars('LossTR', {'trainD': train.lossd[-1]}, train.epoch[-1])
    writer.add_scalars('loss', {'valid': recorder.val_loss[-1]}, recorder.epoch[-1])
    #writer.add_scalars('LossVAL', {'validD': valid.lossd[-1]}, valid.epoch[-1])
    
    writer.add_scalars('rmse', {'train': recorder.train_rmse[-1]}, recorder.epoch[-1])
    writer.add_scalars('rmse', {'valid': recorder.val_rmse[-1]}, recorder.epoch[-1])
    writer.add_scalars('mape', {'train': recorder.train_mape[-1]}, recorder.epoch[-1])
    writer.add_scalars('mape', {'valid': recorder.val_mape[-1]}, recorder.epoch[-1])
    writer.add_scalars('lr', {'learning rate': recorder.lr[-1]}, recorder.epoch[-1])



class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, config, patience=None, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, logger.infos a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience if patience is not None else config.patience
        self.min_patience =  getattr(config, "min_patience", 0)
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.current_checkpoint = None

    def __call__(self, val_loss, model_dict, epoch, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if (self.counter >= self.patience) & (epoch > self.min_patience):
                self.early_stop = True
                self._cleanup_checkpoints(save_path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict, epoch, save_path)
            self.counter = 0
    
    def find_checkpoints(self, directory, pattern):
        """Finds files matching the given pattern."""
        try:
            directories = [
                os.path.join(directory, entry) 
                for entry in os.listdir(directory)
            ]
            files = [
                os.path.join(directory, file) 
                for directory in directories 
                for file in os.listdir(directory) 
                if pattern in file
            ]
            return files
        except Exception as e:
            logger.error(f"Error while scanning directory: {e}")
            return []
        
    def save_checkpoint(self, val_loss, model_dict, epoch, save_path, n_save=3):
        """Saves model when validation loss decreases and removes older checkpoints."""
        
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)

        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                f"Saving model trained with {epoch} epochs..."
            )
        self.val_loss_min = val_loss
        
        
        for key, value in model_dict.items():
            if key == "epoch":
                continue
            temp_save_path = os.path.join(checkpoint_path, f"{key}_epoch_{model_dict['epoch']}.pth")
            torch.save(value, temp_save_path)

        metadata = {
            'epoch': model_dict['epoch'],
            'components': {
                key: os.path.join(checkpoint_path, f"{key}_epoch_{model_dict['epoch']}.pth")
                for key in model_dict if key != 'epoch'
            }
        }
        dest_path = os.path.join(checkpoint_path, f"metadata_epoch_{model_dict['epoch']}.pth")
        self.current_checkpoint = dest_path
        torch.save(metadata, dest_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints(save_path, n_save)
    
    def _cleanup_checkpoints(self, save_path, n_save=3):
        """Remove older checkpoints, keeping only the n_save most recent ones."""
        try:
            # Get all checkpoint directories
            checkpoint_dirs = [
                d for d in os.listdir(save_path)
                if os.path.isdir(os.path.join(save_path, d)) and d.startswith("checkpoint_epoch_")
            ]
            
            if len(checkpoint_dirs) <= n_save:
                return
            
            # Sort by epoch number
            checkpoint_dirs.sort(key=lambda x: int(x.split("_")[-1]))
            
            # Remove oldest checkpoints
            for old_dir in checkpoint_dirs[:-n_save]:
                old_path = os.path.join(save_path, old_dir)
                import shutil
                shutil.rmtree(old_path)
                logger.info(f"Removed old checkpoint: {old_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up checkpoints: {e}")


def collate_skip_none(batch):
    # filter out None items
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)




def pad_tensor(x, target_h, target_w):
    """
    Pads a tensor on the bottom-right to (target_h, target_w).
    Assumes x shape is (..., H, W)
    """
    h, w = x.shape[-2:]
    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Target size smaller than tensor size")

    # Pad format: (left, right, top, bottom)
    return F.pad(x, (0, pad_w, 0, pad_h))


def collate_pad(batch):
    """
    Collate function that:
    - skips None
    - pads tensors to max H, W in batch
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    x_highs, x_meds, x_statics, ys = zip(*batch)

    # Find max spatial size in batch
    max_h = max(x.shape[-2] for x in x_highs)
    max_w = max(x.shape[-1] for x in x_highs)

    # Pad all tensors
    x_highs = torch.stack([
        pad_tensor(x, max_h, max_w) for x in x_highs
    ])

    x_meds = torch.stack([
        pad_tensor(x, max_h, max_w) for x in x_meds
    ])

    x_statics = torch.stack([
        pad_tensor(x, max_h, max_w) for x in x_statics
    ])

    ys = torch.stack([
        pad_tensor(y, max_h, max_w) for y in ys
    ])

    return x_highs, x_meds, x_statics, ys


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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        # if args.ensamble is True:
        #     state_dict = load_model_in_DDP(checkpoint['state_dict'])
        # else:
        #     state_dict = checkpoint['state_dict']
        
        model.load_state_dict( checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        #ema.load_state_dict(checkpoint["ema"])
        checkp_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {checkp_epoch}")

    start_epoch = 0 if checkpoint_path is None else checkp_epoch

    return model, optimizer, scheduler, start_epoch#, ema

def compute_ensamble_rmse(y, x, mask):
    """
    Computes RMSE with spatial mask.
    
    Args:
        y: ground truth, shape (batch, height, width)
        x: predictions, shape (ensemble, batch, height, width)
        mask: boolean mask of shape (height, width)

    Returns:
        Scalar RMSE averaged over batch, using valid (masked) pixels only
    """
    x_mean = x.mean(axis=0)  # (batch, height, width)
    sq_error = (y - x_mean) ** 2  # (batch, height, width)

    mask_broadcast = mask[None, :, :]  # (1, height, width)
    sq_error_masked = np.where(mask_broadcast, sq_error, np.nan)

    rmse_per_sample = np.sqrt(np.nanmean(sq_error_masked, axis=(1, 2)))  # (batch,)
    rmse = np.nanmean(rmse_per_sample)  # scalar
    return rmse


def compute_ensamble_spread(x, mask):
    """
    Computes ensemble spread (standard deviation) with spatial mask.
    
    Args:
        x: predictions, shape (ensemble, batch, height, width)
        mask: boolean mask of shape (height, width)

    Returns:
        Scalar spread averaged over batch and spatial dimensions (with mask)
    """
    ens, batch, h, w = x.shape
    x_mean = x.mean(axis=0)  # (batch, height, width)
    sq_diff = (x - x_mean[None, ...]) ** 2  # (ensemble, batch, height, width)
    sample_var = np.sum(sq_diff, axis=0) / (ens - 1)  # (batch, height, width)

    mask_broadcast = mask[None, :, :]  # (1, height, width)
    sample_var_masked = np.where(mask_broadcast, sample_var, np.nan)

    spread_per_sample = np.sqrt(np.nanmean(sample_var_masked, axis=(1, 2)))  # (batch,)
    spread = np.nanmean(spread_per_sample)  # scalar
    return spread

def compute_ssr(y, x, mask):
    """
    Computes spread-skill ratio (SSR) as spread / RMSE.

    Args:
        y: ground truth, shape (batch, height, width)
        x: predictions, shape (ensemble, batch, height, width)
        mask: boolean mask of shape (height, width)

    Returns:
        SSR (scalar)
    """
    rmse = compute_ensamble_rmse(y, x, mask)
    spread = compute_ensamble_spread(x, mask)
    return spread / rmse if rmse != 0 else np.nan


"""
Metrics with null values
"""

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))

def masked_bce(preds, labels, null_val=np.nan, pos_weight=None):
    """
    preds:  (B, H, W) logits
    labels: (B, H, W) float binary mask (can contain NaN)
    pos_weight: scalar or tensor for positive class weighting
    """
    preds = preds.squeeze(1) if preds.dim() == 4 else preds

    # ── build mask ─────────────────────────────
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    # removed: mask = mask / mask.mean().clamp(min=1e-8)
    mask = torch.nan_to_num(mask, nan=0.0)

    # ── clean labels ───────────────────────────
    labels_clean = torch.nan_to_num(labels, nan=0.0).float()

    # ── pos_weight handling ────────────────────
    if pos_weight is not None:
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, device=preds.device)
        pos_weight = pos_weight.to(preds.device)

    # ── BCE with logits ────────────────────────
    loss = F.binary_cross_entropy_with_logits(
        preds,
        labels_clean,
        reduction="none",
        pos_weight=pos_weight
    )

    # ── apply mask and average over valid pixels only ──
    loss = loss * mask
    loss = torch.nan_to_num(loss, nan=0.0)

    return loss.sum() / mask.sum().clamp(min=1)



def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    # normalize the mask by the mean
    mask /=  torch.mean(mask)
    # Replace any NaNs in the mask with zeros
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # Calculate the percentage error (with small epsilon to avoid division by zero)
    epsilon = 1e-10
    loss = torch.abs(preds - labels) / (labels + epsilon)
    # Apply the mask to the loss
    loss = loss * mask
    # Replace any NaNs in the loss with zeros
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # Return the mean of the masked loss
    return torch.mean(loss)


def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def corrcoef(x):
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

def tensor_corr(prediction, label):
    x = label.squeeze().reshape(-1)
    y = prediction.squeeze().reshape(-1)
    return pearsonr(x, y)

"""
Metrics with custom mask
"""

def mask_mae(preds, labels, mask, return_value=True):
    loss = torch.abs(preds-labels)
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(mask), torch.tensor(0.0), null_loss)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = np.where(np.isnan(mask), 0, null_loss)
    elif not mask:
        null_loss = loss
        full_mask = torch.ones(loss.shape)

    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss

def mask_rmse(preds, labels, mask=None):
    mse = mask_mse(preds=preds, labels=labels, mask=mask)
    return torch.sqrt(mse)

from typing import Union

class CustomMetrics:

    def __init__(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        metric_list: Union[List[str], str],
        mask: Union[None, torch.Tensor, np.ndarray] = None,
        masked: bool = False,
        reduction: str = "mean"
    ):

        self.preds = self._to_tensor(preds)
        self.labels = self._to_tensor(labels)

        self.mask = None
        self.masked = masked
        self.reduction = reduction

        if mask is not None:
            self.mask = self._to_tensor(mask).float()

        if masked and self.mask is None:
            raise ValueError("Masked=True but no mask provided")

        if masked:
            self._count_masked_pixels(self.mask)

        self.losses, self.metrics = self._get_metrics(metric_list)

    # -------------------------------------------------------
    # Utilities
    # -------------------------------------------------------

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    # -------------------------------------------------------
    # Metric dispatcher
    # -------------------------------------------------------

    def _get_metrics(self, metric_list):

        if isinstance(metric_list, str):
            metric_list = [metric_list]

        results = []
        metrics = []

        for metric in metric_list:

            loss = self._apply_metric(metric)

            if self.masked:
                loss = self._apply_mask(loss)

            results.append(loss)
            metrics.append(metric)

        return results, metrics

    # -------------------------------------------------------
    # Metric definitions
    # -------------------------------------------------------

    def _apply_metric(self, metric):

        preds = self.preds
        labels = self.labels

        if metric == "rmse":
            loss = torch.sqrt((preds - labels) ** 2)

        elif metric == "bias":
            loss = preds - labels

        elif metric == "mse":
            loss = (preds - labels) ** 2

        elif metric == "mae":
            loss = torch.abs(preds - labels)

        elif metric == "mape":
            eps = 1e-8
            loss = torch.abs((preds - labels) / (labels + eps))

        elif metric == "cross_entropy":
            # expects preds: (B,C,W,H), labels: (B,W,H)
            loss = torch.nn.functional.cross_entropy(
                preds,
                labels.long(),
                reduction="none"
            )

        else:
            raise ValueError(f"Metric {metric} not recognized")

        return loss

    # -------------------------------------------------------
    # Mask utilities
    # -------------------------------------------------------

    def _count_masked_pixels(self, mask):

        good_pixels = mask.sum()
        tot_pixels = mask.numel()

        logger.info(
            f"{(1 - good_pixels / tot_pixels):.2%} of pixels masked in loss"
        )

    def _apply_mask(self, loss):

        mask = self.mask

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = mask.expand_as(loss)

        loss = loss * mask
        loss = torch.nan_to_num(loss, nan=0.0)

        if self.reduction == "mean":
            return loss.sum() / mask.sum()

        elif self.reduction == "none":
            return loss

        else:
            raise ValueError("Unsupported reduction")

def mask_mape(preds, labels, mask, return_value=True):
    loss = torch.abs((preds-labels)/labels)
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(mask), torch.tensor(0.0), null_loss)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = np.where(np.isnan(mask), 0, null_loss)
    elif not mask:
        full_mask = torch.ones(loss.shape)

    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss

def mask_mse(preds, labels, mask, return_value=True):
    loss = (preds-labels)**2
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(mask), torch.tensor(0.0), null_loss)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = np.where(np.isnan(mask), 0, null_loss)
    elif not mask:
        null_loss = loss
        full_mask = torch.ones(loss.shape)

    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss
            

def masked_custom_loss(criterion, preds, labels, mask=None, eps=1e-8, return_value=True):
    # Remove NaNs and Infs from predictions and labels before loss computation
    # Create mask for finite values (not NaN or Inf)
    finite_mask = torch.isfinite(preds) & torch.isfinite(labels)
    
    if not finite_mask.any():
        logger.warning("[masked_custom_loss] All predictions/labels are NaN or Inf! Returning zero loss.")
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    
    # Filter out non-finite values
    preds_clean = preds[finite_mask]
    labels_clean = labels[finite_mask]
    
    loss = criterion(preds_clean, labels_clean)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.to(loss.device)
        # Broadcast and filter mask
        if mask.shape != finite_mask.shape:
            try:
                mask = torch.broadcast_to(mask, finite_mask.shape)
            except RuntimeError:
                logger.warning(f"[masked_custom_loss] Could not broadcast mask shape {mask.shape} to finite_mask shape {finite_mask.shape}")
                mask = None
        
        if mask is not None:
            mask_clean = mask[finite_mask]
            masked_loss = loss * mask_clean
            
            if return_value:
                denom = mask_clean.sum()
                if denom == 0:
                    return torch.tensor(0.0, device=loss.device, requires_grad=True)
                return masked_loss.sum() / (denom + eps)
            else:
                return masked_loss
    
    # No mask case (or mask broadcasting failed)
    if return_value:
        return loss.mean()
    else:
        return loss
        
def mask_mbe(preds, labels, mask, return_value=True):
    loss = (preds-labels)
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
    null_loss = (loss * full_mask)
    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss
        

class MaskedAdmin2Loss():
    def __init__(self, loss_fn, num_zones, device, pop_weights=None):
        self.device = device
        self.loss_fn = loss_fn
        self.num_zones = num_zones
        self.pop_weights = torch.tensor(pop_weights, dtype=torch.float32, device=self.device) if pop_weights is not None else None

    def zone_aggregate(self, pred, zone_map, pop_map=None):
        """
        Aggregate pixel log-rates to zone-level log-expected-cases.

        Without population (pop_map=None):
            log λ_z = logsumexp(ŷ_i for i∈z)           [uniform P=1]

        With population weighting (pop_map = log(P_i)):
            log λ_z = logsumexp(log(P_i) + ŷ_i for i∈z)
                    ≡ log(Σ P_i · exp(ŷ_i))            [epidemiologically correct]

        pop_map must contain log-population (from GPWv4Population.__getitem__),
        NOT raw counts. All arithmetic stays in log-space so there is no
        P_i · exp(ŷ_i) product that can overflow.

        zone_max is computed under no_grad to avoid inplace/backward conflicts.
        """
        B = pred.shape[0]
        pred = pred.squeeze(1).float()          # [B, H, W]

        H, W = pred.shape[-2], pred.shape[-1]
        zone_map = zone_map.view(B, H, W).long()

        if pop_map is not None:
            pop_map = pop_map.view(B, H, W).float().to(pred.device)

        zone_counts = torch.zeros(B, self.num_zones, device=pred.device, dtype=torch.float32)

        # Valid mask: exclude -1 (out-of-zone) and out-of-bounds
        valid = (zone_map >= 0) & (zone_map < self.num_zones)  # [B, H, W]

        # Pass 1: per-zone max of (log_pop + ŷ_i) for logsumexp stability.
        # Computed under no_grad — zone_max is a constant stabilizer, not a
        # learnable quantity. scatter_reduce_ inplace on a grad-tracked tensor
        # causes a ScatterReduceBackward version conflict in pass 2.
        zone_max = torch.full((B, self.num_zones), float('-inf'),
                              device=pred.device, dtype=torch.float32)
        with torch.no_grad():
            for b in range(B):
                zm = zone_map[b][valid[b]].view(-1)
                pv = pred[b][valid[b]].view(-1)
                pv_eff = pv + pop_map[b][valid[b]].view(-1) if pop_map is not None else pv
                zone_max[b].scatter_reduce_(0, zm, pv_eff, reduce='amax', include_self=True)
                zone_counts[b].scatter_add_(0, zm, torch.ones_like(pv))

        # Pass 2: Σ exp(log(P_i) + ŷ_i − m).  When pop_map is None, log(P_i)=0.
        # zone_max is detached; gradients flow through pred only.
        zone_exp_sum = torch.zeros(B, self.num_zones, device=pred.device, dtype=torch.float32)
        for b in range(B):
            zm = zone_map[b][valid[b]].view(-1)
            pv = pred[b][valid[b]].view(-1)
            pv_eff = pv + pop_map[b][valid[b]].view(-1) if pop_map is not None else pv
            shifted = torch.exp(pv_eff - zone_max[b][zm])
            zone_exp_sum[b].scatter_add_(0, zm, shifted)

        # log λ_z = max + log(Σ exp(log(P_i) + ŷ_i − max))
        # Zones with no pixels retain -inf from zone_max → masked out in zone_loss
        log_zone_rates = zone_max + torch.log(zone_exp_sum + 1e-8)
        return log_zone_rates, zone_counts

    def zone_loss(self, pred, target, zone_map, pop_map=None):
        """
        pred:     [B, 1, H, W]  — model output (pixel log per-capita risk)
        target:   [B, num_zones] — ground truth zone-level case counts
        zone_map: [B, H, W]     — integer zone IDs
        pop_map:  [B, H, W]     — GPWv4 population per pixel (optional)
        """
        zone_preds, zone_counts = self.zone_aggregate(pred, zone_map, pop_map)

        # Mask 1: zones not present in this spatial patch
        spatial_mask = zone_counts > 0              # [B, num_zones]

        # Mask 2: zones with null/nan/inf targets
        target_mask = torch.isfinite(target)        # [B, num_zones]

        # Combined mask: zone must be present AND have a valid target
        valid_mask = spatial_mask & target_mask     # [B, num_zones]

        if valid_mask.sum() == 0:
            # No valid zones in this batch — return zero loss with gradient
            return (zone_preds * 0).sum()

        loss = self.loss_fn(zone_preds[valid_mask], target[valid_mask])
        return loss
    
    def weighted_zone_loss(self, zone_preds, target, valid_mask, weights):
        diff = (zone_preds - target) ** 2
        weighted = diff * weights  # [B, num_zones]
        return weighted[valid_mask].mean()

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
class VIIRSData(RasterDataset):
    filename_glob = "VNP*.tif"
    filename_regex = r"^VNP46A3_(?P<date>\d{8})\.tif$"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False

    def __init__(
        self,
        root: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        n_bands: int = 3,
        **kwargs,
    ):
        super().__init__(root, **kwargs)

        self.n_bands = n_bands

        # Only filter if at least one bound is provided
        if min_date is not None or max_date is not None:

            min_date = pd.Timestamp(min_date) if min_date else None
            max_date = pd.Timestamp(max_date) if max_date else None

            intervals = self.index.index  # IntervalIndex
            dates = intervals.left        # start of each interval (daily rasters)

            mask = pd.Series(True, index=self.index.index)

            if min_date is not None:
                mask &= dates >= min_date

            if max_date is not None:
                mask &= dates <= max_date

            self.index = self.index[mask]

    def __getitem__(self, query):
        sample = super().__getitem__(query)

        sample["image"] = sample["image"][:self.n_bands]

        return sample


class StaticLayer(RasterDataset):
    filename_glob = "DEN_riskmap_wmean_masked.tif"
    is_image = True
    separate_files = False
    nodata = -3.3999999521443642e+38

    def __init__(self, paths, crs=None, res=None, transforms=None, nodata=None):
        super().__init__(paths=paths, crs=crs, res=res, transforms=transforms)
        if nodata is not None:
            self.nodata = nodata

    def _merge_files(self, filepaths, query, band_indexes=None):
        tensor = super()._merge_files(filepaths, query, band_indexes)
        
        # Mask nodata sentinel values to NaN
        if self.nodata is not None:
            tensor = tensor.float()
            # Use tolerance to handle float32 precision issues
            nodata_mask = torch.abs(tensor - self.nodata) < torch.abs(torch.tensor(self.nodata)) * 1e-3
            tensor[nodata_mask] = float('nan')
        
        return tensor





class XarraySpatioTemporalDataset(RasterDataset):
    """
    Generic xarray-backed dataset.

    Returns:
        image: Tensor[T, C, H, W]  (or [T, C] if no spatial dims)
    """

    is_image = True
    separate_files = False

    def __init__(
        self,
        ds:xr.Dataset | xr.DataArray,                      # xarray Dataset or DataArray
        variables=None,
        T_max=32,
        transform=None,
        chunks=None,
    ):  
        
        self.ds = self._normalize_coords(ds)
        if variables is not None:
            self.variables = variables
            self.ds = self.ds[variables]

        if "FAO_GAUL_code" in self.ds:
            self.ds, self.zone_remap, self.num_zones = self.remap_zone_ids(self.ds, zone_var="FAO_GAUL_code")

        self.T_max = T_max
        self.transform = transform

        self.times = self.ds.time.values
        self.chunks = "auto" if chunks is None else chunks
    
    def _normalize_coords(self, ds):
        rename = {}
        for k in ("lon", "longitude"):
            if k in ds.coords:
                rename[k] = "x"
        for k in ("lat", "latitude"):
            if k in ds.coords:
                rename[k] = "y"
        if rename:
            ds = ds.rename(rename)

        return ds
    

    def __len__(self):
        return len(self.times)

    def remap_zone_ids(self, ds, zone_var):
        """
        Remap arbitrary zone IDs to contiguous 0..N-1 using xarray.apply_ufunc.
        Returns the remapped DataArray and the lookup table.
        """
        zone_data = ds[zone_var]

        # Get unique IDs from the full array (compute() forces evaluation if dask-backed)
        unique_ids = np.unique(zone_data.values)
        unique_ids = unique_ids[np.isfinite(unique_ids)].astype(np.int32)
        num_zones = len(unique_ids)

        # Build numpy lookup array: old_id -> new_id
        max_id = int(unique_ids.max())
        lookup = np.full(max_id + 1, -1, dtype=np.int32)
        for new_id, old_id in enumerate(sorted(unique_ids.tolist())):
            lookup[old_id] = new_id

        def _remap(arr):
            arr = arr.astype(np.int64)
            arr_clipped = np.clip(arr, 0, max_id)
            return lookup[arr_clipped]

        remapped = xr.apply_ufunc(
            _remap,
            zone_data,
            dask="parallelized",        # works transparently if ds is dask-backed
            output_dtypes=[np.int32],
            keep_attrs=True,
        )

        ds = ds.assign({zone_var: remapped})

        logger.info(f"Zone remapping: {num_zones} zones, "
                    f"ID range {unique_ids.min()}–{unique_ids.max()} → 0–{num_zones - 1}")

        return ds, lookup, num_zones

    def __getitem__(self, query):
        """
        query can be:
            - idx (int)
            - (x_slice, y_slice, t_slice)
        """
        

        # -----------------------------
        # Case B: temporal-only access
        # -----------------------------
        if isinstance(query, int):
            t = pd.Timestamp(self.times[query])
            # Clip time inside dataset bounds
            min_t = pd.Timestamp(self.ds.time.min().values)
            max_t = pd.Timestamp(self.ds.time.max().values)

            if t < min_t or t > max_t:
                return {"image": torch.empty(0)}

            ds = self.ds.sel(time=t)

        # -----------------------------
        # Case A: spatiotemporal access
        # -----------------------------
        else:
            x_slice, y_slice, t_slice = query

            start_time = pd.Timestamp(t_slice.start)
            stop_time  = pd.Timestamp(t_slice.stop)

            # Clip time range to dataset bounds
            min_t = pd.Timestamp(self.ds.time.min().values)
            max_t = pd.Timestamp(self.ds.time.max().values)

            start_time = max(start_time, min_t)
            stop_time  = min(stop_time, max_t)

            if start_time > max_t or stop_time < min_t:
                return {"image": torch.empty(0)}

            ds = self.ds.sel(time=slice(start_time, stop_time))

            if ds.time.size == 0:
                return {"image": torch.empty(0)}

            # ---- Spatial selection (robust to axis order) ----
            sel_dict = {}
            for axis, s in zip(["x", "y"], [x_slice, y_slice]):
                if axis in ds.coords:
                    coord_vals = ds[axis].values
                    if coord_vals[0] < coord_vals[-1]:
                        sel_dict[axis] = slice(s.start, s.stop)
                    else:
                        sel_dict[axis] = slice(s.stop, s.start)

            if sel_dict:
                ds = ds.sel(**sel_dict)

        # -------------------------------------------------
        # Extract variables
        # -------------------------------------------------
        if hasattr(ds, "data_vars"):
            vars_ = self.variables or list(ds.data_vars)
            arrays = [ds[v].values for v in vars_]
        else:
            arrays = [ds.values]

        if len(arrays) == 0:
            return {"image": torch.empty(0)}

        # -------------------------------------------------
        # Build tensor [T, C, H, W]
        # -------------------------------------------------
        tensors = []
        for arr in arrays:
            t = torch.from_numpy(arr).float()

            if t.ndim == 2:       # [H, W]
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.ndim == 3:     # [T, H, W]
                t = t.unsqueeze(1)

            tensors.append(t)

        patch = torch.cat(tensors, dim=1)

        # -------------------------------------------------
        # Temporal padding/truncation
        # -------------------------------------------------
        T = patch.shape[0]
        if T > self.T_max:
            patch = patch[-self.T_max:]
        elif T < self.T_max:
            pad = torch.zeros((self.T_max - T, *patch.shape[1:]), dtype=patch.dtype)
            patch = torch.cat([pad, patch], dim=0)

        if self.transform:
            patch = self.transform(patch)

        return {"image": patch}


class LandCoverData:
    """
    Annual MODIS MCD12Q1 land cover loader.

    Expects one NetCDF per year under `root/`, named with a 4-digit year, e.g.:
        MCD12Q1_latin_america_2016.nc

    Each file must have (lat, lon) coords and a `LC_Type1` data variable
    (IGBP classification, EPSG:4326).

    __getitem__ receives (x_slice, y_slice, year_or_timestamp), crops to the
    query bbox, resamples to `target_size` via nearest-neighbour, and returns
    {"image": Tensor[1, H, W]} of raw IGBP class floats.

    Falls back to the nearest available year when exact year is missing.
    """

    def __init__(self, root, target_size: tuple = (102, 102)):
        import re

        self.root = Path(root)
        self.target_size = target_size

        self._index: dict = {}
        year_re = re.compile(r"(20\d{2})")
        for fpath in sorted(self.root.rglob("*.nc")):
            m = year_re.search(fpath.name)
            if m:
                self._index[int(m.group(1))] = fpath

        if not self._index:
            raise RuntimeError(f"No land-cover NetCDF files found under {self.root}")

        self._years = sorted(self._index.keys())
        logger.info(
            f"LandCoverData: {len(self._years)} annual files indexed "
            f"({self._years[0]}–{self._years[-1]})"
        )

    def _nearest_year(self, year: int) -> int:
        return min(self._years, key=lambda y: abs(y - year))

    def __getitem__(self, query) -> dict:
        """
        query: (x_slice, y_slice, year_or_timestamp)
          x_slice.start/stop: lon degrees
          y_slice.start/stop: lat degrees (start < stop)
          year_or_timestamp: int year OR pandas Timestamp
        """
        import numpy as _np
        from scipy.ndimage import zoom

        x_slice, y_slice, year_ref = query
        year = int(pd.Timestamp(year_ref).year) if not isinstance(year_ref, int) else year_ref
        year = self._nearest_year(year)
        fpath = self._index[year]

        H, W = self.target_size
        dst_arr = _np.full((1, H, W), _np.nan, dtype=_np.float32)

        try:
            ds = xr.open_dataset(fpath, engine="netcdf4")
            da = ds["LC_Type1"]

            # lat may be descending — normalise so sel works in either order
            lat_asc = float(da.lat[0]) < float(da.lat[-1])
            if lat_asc:
                sub = da.sel(lon=slice(x_slice.start, x_slice.stop),
                             lat=slice(y_slice.start, y_slice.stop))
            else:
                sub = da.sel(lon=slice(x_slice.start, x_slice.stop),
                             lat=slice(y_slice.stop,  y_slice.start))

            ds.close()
            arr = sub.values.astype(_np.float32)   # [lat_crop, lon_crop]

            if arr.size == 0:
                return {"image": torch.from_numpy(dst_arr)}

            # Nearest-neighbour resample to target_size via zoom
            zy = H / arr.shape[0]
            zx = W / arr.shape[1]
            resampled = zoom(arr, (zy, zx), order=0, prefilter=False)
            # Flip to standard top-left = N,W orientation if lat was descending
            if not lat_asc:
                resampled = resampled[::-1, :]
            dst_arr[0] = resampled[:H, :W]

        except Exception as e:
            logger.warning(f"LandCoverData: failed to load {fpath}: {e}")

        return {"image": torch.from_numpy(dst_arr.copy())}    # [1, H, W]


class GPWv4Population:
    """
    GPWv4 Rev11 population count loader for exposure-weighted zone aggregation.

    All available epochs are pre-loaded into RAM as float32 numpy arrays
    for fork-safety in DataLoader workers.

    Epoch assignment (no interpolation, per spec):
      year < 2016  → 2010  (covers training years 2012–2015)
      year < 2020  → 2015  (2016–2019)
      year ≥ 2020  → 2020

    __getitem__(x_slice, y_slice, year_or_timestamp)
        Returns Tensor[H, W] of population counts aggregated to target_size.
        NaN pixels in source (no-data) are treated as zero population.
    """

    _EPOCH_BREAKS = [(2016, 2010), (2020, 2015)]   # (threshold, epoch_if_below)
    _FALLBACK_EPOCH = 2020

    def __init__(self, root, target_size: tuple = (86, 86)):
        import re as _re

        self.root = Path(root)
        self.target_size = target_size
        self._cache: dict = {}   # epoch → {"arr": np.ndarray, "lat": ..., "lon": ...}

        year_re = _re.compile(r"GPWv4.*?(\d{4})\.nc$")
        indexed = {}
        for fpath in sorted(self.root.glob("GPWv4*.nc")):
            m = year_re.search(fpath.name)
            if m:
                indexed[int(m.group(1))] = fpath

        if not indexed:
            raise RuntimeError(f"No GPWv4 NetCDF files found under {self.root}")

        for epoch, fpath in sorted(indexed.items()):
            logger.info(f"GPWv4Population: loading epoch {epoch} from {fpath} ...")
            ds = xr.open_dataset(fpath)
            arr = np.asarray(ds["population_count"].values, dtype=np.float32)
            lat = np.asarray(ds["lat"].values, dtype=np.float64)
            lon = np.asarray(ds["lon"].values, dtype=np.float64)
            ds.close()
            arr = np.nan_to_num(arr, nan=0.0)   # uninhabited → 0
            self._cache[epoch] = {"arr": arr, "lat": lat, "lon": lon}
            logger.info(f"GPWv4Population: epoch {epoch} shape={arr.shape}")

        self._available_epochs = sorted(self._cache.keys())

    def _assign_epoch(self, year: int) -> int:
        for threshold, epoch in self._EPOCH_BREAKS:
            if year < threshold:
                break_epoch = epoch
                if break_epoch in self._cache:
                    return break_epoch
        # fallback: nearest available epoch
        return min(self._available_epochs, key=lambda e: abs(e - year))

    def __getitem__(self, query) -> torch.Tensor:
        """
        query: (x_slice, y_slice, year_or_timestamp)
          x_slice.start/stop → lon bounds
          y_slice.start/stop → lat bounds (start=south, stop=north)
        Returns Tensor[H, W] at target_size resolution, north at row 0.
        """
        from scipy.ndimage import zoom as _zoom

        x_slice, y_slice, year_ref = query
        year = (
            int(pd.Timestamp(year_ref).year)
            if not isinstance(year_ref, int)
            else year_ref
        )
        epoch = self._assign_epoch(year)
        cache = self._cache[epoch]
        arr, lat, lon = cache["arr"], cache["lat"], cache["lon"]

        H_tgt, W_tgt = self.target_size

        lon_lo = max(float(x_slice.start), float(lon.min()))
        lon_hi = min(float(x_slice.stop),  float(lon.max()))
        lat_lo = max(float(y_slice.start), float(lat.min()))
        lat_hi = min(float(y_slice.stop),  float(lat.max()))

        if lon_lo >= lon_hi or lat_lo >= lat_hi:
            return torch.zeros(H_tgt, W_tgt, dtype=torch.float32)

        lat_idx = np.where((lat >= lat_lo) & (lat <= lat_hi))[0]
        lon_idx = np.where((lon >= lon_lo) & (lon <= lon_hi))[0]

        if lat_idx.size == 0 or lon_idx.size == 0:
            return torch.zeros(H_tgt, W_tgt, dtype=torch.float32)

        # lat may be descending: lat_idx.min() = northernmost row
        lat_i0, lat_i1 = int(lat_idx.min()), int(lat_idx.max()) + 1
        lon_i0, lon_i1 = int(lon_idx.min()), int(lon_idx.max()) + 1
        sub = arr[lat_i0:lat_i1, lon_i0:lon_i1]   # [H_src, W_src], north at row 0

        if sub.size == 0:
            return torch.zeros(H_tgt, W_tgt, dtype=torch.float32)

        zy = H_tgt / sub.shape[0]
        zx = W_tgt / sub.shape[1]
        resampled = _zoom(sub, (zy, zx), order=1, prefilter=False)
        resampled = np.clip(resampled, 0.0, None).astype(np.float32)
        # Return log-population so zone_aggregate can compute
        # logsumexp(log(P_i) + ŷ_i) ≡ log(Σ P_i · exp(ŷ_i)) without overflow.
        # epsilon avoids log(0) for uninhabited pixels (they get log(1e-6) ≈ −14,
        # contributing near-zero weight in the aggregation).
        log_pop = np.log(resampled + 1e-6).astype(np.float32)
        return torch.from_numpy(log_pop.copy())   # [H, W], log-population


class SoilMoistureData:
    """
    Loader for daily Sentinel-1 soil moisture GeoTIFFs.

    File layout: <root>/<year>/SM_[A|D]_YYYYMMDD.tif
    Two sensor passes per day (A = ascending, D = descending).

    Fast path (recommended): pass `cache_dir` pointing to a directory that
    contains `sm_6day_1km.zarr` (built by build_sm_cache.py).  The cache stores
    all 6-day composites on a ~1 km (0.01°) grid covering Latin America;
    __getitem__ becomes a lazy zarr crop — no rasterio per sample.
    Values are stored as float32 m³/m³ (already scaled).

    Slow path (fallback): when no cache is available, __getitem__ opens the
    raw TIF files via rasterio on every call.  This is ~5 s/sample on lustre
    and is only suitable for debugging.

    Returns {"image": Tensor[num_steps, 2, H, W]}
      channel 0: SM value in m³/m³ (0 where gap)
      channel 1: validity mask (1 where data, 0 where gap)
    """

    def __init__(
        self,
        root,
        min_date=None,
        max_date=None,
        window_days: int = 6,
        num_steps: int = 5,
        target_size: tuple = None,   # None = return native zarr resolution
        cache_dir=None,
    ):
        import re
        from datetime import date as _date, timedelta as _td

        self.root = Path(root)
        self.window_days = window_days
        self.num_steps = num_steps
        self.target_size = target_size  # None → native zarr resolution, else (H, W)

        # ── Fast path: open zarr lazily (no RAM preload) ──────────────
        # Prefer v2 zarr (chunks=(5,2,100,100), ~10x fewer I/Os per sample).
        # Fall back to v1 if v2 not yet built.
        self._zarr_cache = None
        if cache_dir is not None:
            _cache_dir = Path(cache_dir)
            sm_zarr = next(
                (p for p in [_cache_dir / 'sm_6day_1km_v2.zarr',
                              _cache_dir / 'sm_6day_1km.zarr'] if p.exists()),
                None,
            )
            if sm_zarr is not None:
                import zarr as _zarr
                logger.info(f"SoilMoistureData: opening zarr cache at {sm_zarr}")
                _store = _zarr.open(str(sm_zarr), mode='r')
                attrs = dict(_store.attrs)
                self._cache_start   = pd.Timestamp(attrs['start_date']).date()
                self._cache_lon_min = float(attrs['lon_min'])
                self._cache_lat_min = float(attrs['lat_min'])
                self._cache_res     = float(attrs['resolution'])
                self._cache_nwin    = int(attrs['n_windows'])
                self._cache_h       = int(attrs.get('h_global', _store.shape[2]))
                self._cache_w       = int(attrs.get('w_global', _store.shape[3]))
                self._zarr_cache    = _store   # lazy — reads on demand
                logger.info(
                    f"SoilMoistureData: zarr ready "
                    f"({self._cache_nwin} windows, {self._cache_res}°/px, "
                    f"{_store.nbytes / 1e9:.1f} GB uncompressed)"
                )

        # ── Slow path: build index of raw TIF files ──────────────────
        # Always index files so we can report the date range.
        self._index: dict = {}
        pattern = re.compile(r"^SM_([AD])_(\d{8})\.tif$")

        min_dt = pd.Timestamp(min_date).date() if min_date else None
        max_dt = pd.Timestamp(max_date).date() if max_date else None

        for year_dir in sorted(self.root.iterdir()):
            if not year_dir.is_dir():
                continue
            for fpath in sorted(year_dir.glob("SM_*.tif")):
                m = pattern.match(fpath.name)
                if not m:
                    continue
                sensor = m.group(1)          # 'A' or 'D'
                dt = pd.Timestamp(m.group(2), tz=None).date()
                if min_dt and dt < min_dt:
                    continue
                if max_dt and dt > max_dt:
                    continue
                if dt not in self._index:
                    self._index[dt] = {"A": None, "D": None}
                self._index[dt][sensor] = fpath

        self._dates = sorted(self._index.keys())
        if not self._dates:
            raise RuntimeError(f"No SM files found under {self.root}")
        logger.info(
            f"SoilMoistureData: {len(self._dates)} days indexed "
            f"({self._dates[0]} – {self._dates[-1]})"
        )

    # ------------------------------------------------------------------
    def _load_window_max(self, window_dates, x_slice, y_slice) -> "np.ndarray":
        """
        Load all A+D files for the given list of dates, crop to bbox,
        resample to target_size, return pixel-wise max + validity mask → [2, H, W].

        Channel 0: SM value  (0 where no observation exists in the window)
        Channel 1: valid mask (1 where ≥1 observation exists, 0 where gap)

        Swath gaps are common — a 6-day window may cover only part of the bbox
        or have no overpasses at all.  The mask lets the model distinguish
        "no data" from "SM = 0 (measured wet/water)" and prevents the temporal
        attention from collapsing on all-zero gap windows.
        """
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds
        import numpy as _np

        H, W = self.target_size
        stacked = []

        for dt in window_dates:
            entry = self._index.get(dt, {})
            for sensor in ("A", "D"):
                fpath = entry.get(sensor)
                if fpath is None:
                    continue
                try:
                    with rasterio.open(fpath) as src:
                        dst_transform = from_bounds(
                            x_slice.start, y_slice.start,
                            x_slice.stop,  y_slice.stop,
                            W, H
                        )
                        dst_crs = src.crs
                        dst_arr = _np.full((1, H, W), _np.nan, dtype=_np.float32)
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=dst_arr,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest,
                            dst_nodata=_np.nan,
                        )
                        # nodata is 32767 (int16 max) but not set in GeoTIFF metadata
                        nd = src.nodata if src.nodata is not None else 32767.0
                        dst_arr[dst_arr >= nd] = _np.nan
                        stacked.append(dst_arr[0])   # [H, W]
                except Exception as e:
                    logger.debug(f"SM: failed to load {fpath}: {e}")

        if not stacked:
            # Entire window is a gap — return zeros + zero mask
            return _np.zeros((2, H, W), dtype=_np.float32)

        arr = _np.stack(stacked, axis=0)              # [N, H, W]
        # valid mask: True where at least one swath covers the pixel
        valid = _np.any(_np.isfinite(arr), axis=0)    # [H, W] bool
        sm_max = _np.where(valid, _np.nanmax(arr, axis=0), 0.0).astype(_np.float32)
        mask   = valid.astype(_np.float32)            # [H, W] 0/1
        return _np.stack([sm_max, mask], axis=0)      # [2, H, W]

    # ------------------------------------------------------------------
    def __getitem__(self, query) -> dict:
        """
        query: (x_slice, y_slice, t_end)
          x_slice, y_slice: slice objects with .start/.stop in lon/lat degrees
          t_end: pandas Timestamp (or anything castable to pd.Timestamp)

        Returns {"image": Tensor[num_steps, 2, H, W]}
          Channel 0: SM composite value (0 where gap)
          Channel 1: validity mask (1 where data, 0 where gap)
        """
        import numpy as _np
        import datetime as _dt

        x_slice, y_slice, t_end = query
        t_end_date = pd.Timestamp(t_end).date()

        if self._zarr_cache is not None:
            # Derive pixel dimensions from bbox and cache resolution (native res)
            res = self._cache_res
            H_tgt = max(1, round((y_slice.stop - y_slice.start) / res))
            W_tgt = max(1, round((x_slice.stop - x_slice.start) / res))
            return self._getitem_cached(x_slice, y_slice, t_end_date, H_tgt, W_tgt)

        if self.target_size is None:
            raise RuntimeError("target_size must be set when no zarr cache is available")
        H_tgt, W_tgt = self.target_size

        # ── Slow rasterio fallback ───────────────────────────────────
        composites = []
        for step in range(self.num_steps - 1, -1, -1):   # oldest first → chronological
            win_end   = t_end_date - _dt.timedelta(days=step * self.window_days)
            win_start = win_end    - _dt.timedelta(days=self.window_days - 1)
            window_dates = [win_start + _dt.timedelta(days=d) for d in range(self.window_days)]
            composites.append(self._load_window_max(window_dates, x_slice, y_slice))

        arr = _np.stack(composites, axis=0)
        return {"image": torch.from_numpy(arr)}

    def _getitem_cached(self, x_slice, y_slice, t_end_date, H_tgt: int, W_tgt: int) -> dict:
        """Fast path: lazy zarr crop at native ~1 km resolution — no rasterio, no zoom.

        Values are already float32 m³/m³ as stored by build_sm_cache.py.
        H_tgt/W_tgt are derived from the bbox size and cache resolution, so the
        zarr crop always matches exactly (no resize needed).
        """
        import numpy as _np

        res = self._cache_res
        current_win = (t_end_date - self._cache_start).days // self.window_days

        lon_i0 = max(0, round((x_slice.start - self._cache_lon_min) / res))
        lon_i1 = min(self._cache_w, round((x_slice.stop  - self._cache_lon_min) / res))
        lat_i0 = max(0, round((y_slice.start - self._cache_lat_min) / res))
        lat_i1 = min(self._cache_h, round((y_slice.stop  - self._cache_lat_min) / res))

        empty = _np.zeros((2, H_tgt, W_tgt), dtype=_np.float32)

        composites = []
        for step in range(self.num_steps - 1, -1, -1):   # oldest first → chronological
            win_idx = current_win - step
            if win_idx < 0 or win_idx >= self._cache_nwin:
                composites.append(empty)
                continue

            patch = self._zarr_cache[win_idx, :, lat_i0:lat_i1, lon_i0:lon_i1]  # [2, dH, dW]
            dH, dW = patch.shape[1], patch.shape[2]

            if dH == 0 or dW == 0:
                composites.append(empty)
                continue

            # Pad to H_tgt×W_tgt if the crop lands near the bbox edge
            if dH < H_tgt or dW < W_tgt:
                padded = _np.zeros((2, H_tgt, W_tgt), dtype=_np.float32)
                padded[:, :dH, :dW] = patch
                composites.append(padded)
            else:
                composites.append(patch[:, :H_tgt, :W_tgt].astype(_np.float32))

        arr = _np.stack(composites, axis=0)   # [num_steps, 2, H, W]
        return {"image": torch.from_numpy(arr)}


class ERA5Daily:
    """
    ERA5 lazy loader with TorchGeo-style [query] interface.
    Loads only requested time slices on the fly.
    Returns: {"image": Tensor[T, C, H, W]}
    """

    def __init__(
        self,
        root,
        variables=None,
        T_max=32,
        T_offset=1,
        transform=None,
        min_date=None,
        max_date=None,
        weekly_cache_dir=None,
    ):
        import xarray as xr
        from pathlib import Path

        self.root = Path(root)
        self.variables = variables
        self.T_max = int(math.ceil(T_max / 7))
        logger.info(f"ERA5 dataset initialized with T_max={self.T_max} weeks (T_offset={T_offset} days)")
        self.T_offset = T_offset
        self.transform = transform

        # Check for pre-computed weekly zarr cache (avoids on-the-fly resampling in __getitem__)
        self._weekly_cache = None
        if weekly_cache_dir is not None:
            weekly_zarr = Path(weekly_cache_dir) / "era5_weekly.zarr"
            if weekly_zarr.exists():
                logger.info(f"ERA5: loading pre-computed weekly cache from {weekly_zarr}")
                _ds = xr.open_zarr(str(weekly_zarr))
                if min_date is not None and max_date is not None:
                    _ds = _ds.sel(time=slice(min_date, max_date))
                # Load entire cache into RAM and rebuild as a pure-numpy Dataset.
                # xr.open_zarr() leaves dask graph references to the zarr store even after
                # .load(). Forked DataLoader workers inherit those references and deadlock
                # when they try to re-open the store. Rebuilding from numpy arrays severs
                # all zarr/dask ties so fork() is safe with num_workers > 0.
                logger.info("ERA5: loading zarr cache into RAM (5 GB, ~30s)...")
                _ds.load()  # materialise all variables in-place
                import numpy as np
                data_vars = {v: (_ds[v].dims, np.asarray(_ds[v].values)) for v in _ds.data_vars}
                coords_dict = {c: np.asarray(_ds.coords[c].values) for c in _ds.coords}
                self._weekly_cache = xr.Dataset(data_vars, coords=coords_dict)
                self.time_index = self._weekly_cache.time.values
                self.x_coords = self._weekly_cache.x.values
                self.y_coords = self._weekly_cache.y.values
                logger.info(f"ERA5 weekly cache in RAM (fork-safe): {len(self.time_index)} weeks.")
                return

        files = sorted(self.root.glob("era5land_latin_america*.nc"))
        if not files:
            raise RuntimeError("No ERA5 files found")

        # open lazily with chunking
        self.ds = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            chunks={"time": 63,
            "x": 43,
            "y": 43},
        )

        self.ds = self._normalize_coords(self.ds)

        if min_date is not None and max_date is not None:
            self.ds = self.ds.sel(time=slice(min_date, max_date))

        self.vars_ = variables or list(self.ds.data_vars)

        # store coords only (cheap)
        self.time_index = self.ds.time.values
        self.x_coords = self.ds.x.values
        self.y_coords = self.ds.y.values

        logger.info("ERA5 opened lazily.")

    def precompute_weekly_cache(self, cache_dir):
        """
        Compute weekly ERA5 aggregations for the full dataset and save as zarr.
        Run this once before training to avoid on-the-fly resampling overhead.
        """
        import xarray as xr
        from pathlib import Path
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        weekly_zarr = cache_dir / "era5_weekly.zarr"
        if weekly_zarr.exists():
            logger.info(f"ERA5 weekly cache already exists at {weekly_zarr}")
            return

        logger.info("Pre-computing weekly ERA5 aggregations (this runs once)...")
        ds_weekly = self._era5_to_weekly(self.ds[self.vars_])
        # rechunk for efficient spatial queries: (week, y, x)
        ds_weekly = ds_weekly.chunk({"time": 9, "x": 43, "y": 43})
        logger.info(f"Writing weekly ERA5 zarr to {weekly_zarr} ...")
        ds_weekly.to_zarr(str(weekly_zarr), mode="w")
        logger.info("ERA5 weekly cache written.")


    def _era5_to_weekly(self, ds, week_freq="1W-MON", week_label="left"):
        """
        Convert ERA5 dataset to weekly resolution using
        physically meaningful aggregations.
        
        Fluxes -> weekly sum
        Temperatures -> weekly mean, min, max
        
        Parameters
        ----------
        ds : xr.Dataset
        week_freq : str
            Resampling frequency (default Monday-based weeks)
        week_label : str
            'left' or 'right' for week labeling
            
        Returns
        -------
        xr.Dataset
            Weekly aggregated dataset
        """
        
        # Define groups
        sum_vars = ["tp", "pev", "e"]
        temp_vars = ["skt", "stl1", "stl2", "stl3", "stl4"]
        
        # Keep only variables present
        sum_vars = [v for v in sum_vars if v in ds]
        temp_vars = [v for v in temp_vars if v in ds]
        
        # --- Fluxes (sum)
        weekly_sum = (
            ds[sum_vars]
            .resample(time=week_freq, label=week_label)
            .sum()
        )
        
        # --- Temperature statistics
        weekly_mean = (
            ds[temp_vars]
            .resample(time=week_freq, label=week_label)
            .mean()
        )
        
        weekly_min = (
            ds[temp_vars]
            .resample(time=week_freq, label=week_label)
            .min()
        )
        
        weekly_max = (
            ds[temp_vars]
            .resample(time=week_freq, label=week_label)
            .max()
        )
        
        # Rename to avoid overwriting
        weekly_mean = weekly_mean.rename({v: f"{v}_mean" for v in temp_vars})
        weekly_min  = weekly_min.rename({v: f"{v}_min"  for v in temp_vars})
        weekly_max  = weekly_max.rename({v: f"{v}_max"  for v in temp_vars})
        
        # Merge all
        ds_weekly = xr.merge([weekly_sum, weekly_mean, weekly_min, weekly_max])
        
        return ds_weekly

    def _normalize_coords(self, ds):
        rename = {}
        for k in ("lon", "longitude"):
            if k in ds.coords:
                rename[k] = "x"
        for k in ("lat", "latitude"):
            if k in ds.coords:
                rename[k] = "y"
        for k in ("valid_time", "date"):
            if k in ds.coords:
                rename[k] = "time"
        if rename:
            ds = ds.rename(rename)

        if "x" in ds.coords:
            x = ds.x.values
            if x.max() > 180:
                ds = ds.assign_coords(x=((ds.x + 180) % 360) - 180).sortby("x")

        if "y" in ds.coords:
            if ds.y[0] > ds.y[-1]:
                ds = ds.sortby("y")

        return ds

    def __getitem__(self, query):
        import torch
        import numpy as np

        x_slice, y_slice, t_slice = query

        if self._weekly_cache is not None:
            # Fast path: read pre-computed weekly data directly from zarr cache
            ds_sel = self._weekly_cache.sel(
                time=slice(t_slice.start, t_slice.stop),
                x=slice(x_slice.start, x_slice.stop),
                y=slice(y_slice.start, y_slice.stop),
            )
        else:
            # Slow path: select from daily data and resample to weekly on the fly
            ds_sel = (
                self.ds[self.vars_]
                .sel(
                    time=slice(t_slice.start, t_slice.stop),
                    x=slice(x_slice.start, x_slice.stop),
                    y=slice(y_slice.start, y_slice.stop),
                )
            )
            ds_sel = self._era5_to_weekly(ds_sel)

        # --- Convert to array lazily ---
        da = ds_sel.to_array()  # [C,T,H,W] lazy

        # --- Move time first ---
        da = da.transpose("time", "variable", "y", "x")

        # --- Compute ONLY this subset ---
        arr = da.compute().values  # loads only selected window

        if arr.size == 0:
            return {"image": torch.empty(0)}

        patch = torch.from_numpy(arr).float()  # [T,C,H,W]

        # --- Temporal padding ---
        T = patch.shape[0]

        if T > self.T_max:
            patch = patch[-self.T_max :]
        elif T < self.T_max:
            pad = torch.zeros((self.T_max - T, *patch.shape[1:]), dtype=patch.dtype)
            patch = torch.cat([pad, patch], dim=0)

        if self.transform:
            patch = self.transform(patch)

        return {"image": patch}

    def __len__(self):
        return len(self.time_index)


# -----------------------------------------------------------------------------
# Multi-resolution dataset
# -----------------------------------------------------------------------------
class DengueDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        viirs,
        era5,
        static,
        spatial_conditioning,
        y,
        patch_sizes={
            "viirs": (1024, 1024),
            "era5": (43, 43),
            "static": (102, 102),
            "y": (86, 86),
            "spatial_conditioning": (86, 86),
            "soil_moisture": None,   # None = native zarr resolution (~1 km)
        },
        y_valid_threshold = 0.3,
        loss_fn:Literal["mse", "poisson"] = "mse",
        num_zones=None,
        bbox=None,
        skip_era5_bounds=False,
        cache_dir=None,
        soil_moisture=None,
        land_cover=None,
        population=None):



        self.viirs = viirs
        self.era5 = era5
        self.static = static
        self.y = y
        self.spatial_conditioning = spatial_conditioning
        self.soil_moisture = soil_moisture
        self.land_cover = land_cover
        self.population = population
        self.cache_dir = cache_dir
        self.skip_era5_bounds = skip_era5_bounds
        self.loss_fn = loss_fn
        self.num_zones = num_zones

        if self.loss_fn == "poisson" and num_zones is None:
            raise ValueError("num_zones must be provided for poisson loss")

        from modis_majortom.utils import latin_box

        self.patch_size = patch_sizes.get("viirs", (1024, 1024))
        self.patch_size_era5 = patch_sizes.get("era5", (43, 43))
        self.patch_size_static = patch_sizes.get("static", (512, 512))
        self.patch_size_y = patch_sizes.get("y", (86, 86))
        self.patch_size_spatial_cond = patch_sizes.get("spatial_conditioning", (86, 86))
        self.patch_size_sm = patch_sizes.get("soil_moisture", (512, 512))

        self.y_valid_threshold = y_valid_threshold
        
        user_bbox = bbox if bbox is not None else latin_box()

        if not isinstance(user_bbox, (list, tuple, np.ndarray)) or len(user_bbox) != 4:
            raise ValueError("bbox must be a list/tuple of 4 values")

        a, b, c, d = map(float, user_bbox)

        # Detect if bbox is [miny, minx, maxy, maxx] (lat-first)
        if abs(a) <= 90 and abs(b) <= 180:
            # Likely [lat_min, lon_min, lat_max, lon_max]
            ub_miny, ub_minx, ub_maxy, ub_maxx = a, b, c, d
        else:
            # Assume [lon_min, lat_min, lon_max, lat_max]
            ub_minx, ub_miny, ub_maxx, ub_maxy = a, b, c, d

        logger.info(
            f"[DengueDataset] Normalized bbox -> "
            f"(minx={ub_minx}, miny={ub_miny}, maxx={ub_maxx}, maxy={ub_maxy})"
        )


        # Compute ERA5 valid data bounds (optionally using fast metadata-only approach)
        era5_bounds = None
        try:
            if self.skip_era5_bounds:
                # FAST: Use coordinate metadata without loading data
                logger.info("[DengueDataset] Using fast metadata-based ERA5 bounds (skipping data load)")
                if hasattr(self.era5.ds, 'x') and hasattr(self.era5.ds, 'y'):
                    x_coords = self.era5.ds.x.values
                    y_coords = self.era5.ds.y.values
                    
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        era_minx = float(x_coords.min())
                        era_maxx = float(x_coords.max())
                        era_miny = float(y_coords.min())
                        era_maxy = float(y_coords.max())
                        
                        era5_bounds = (era_minx, era_miny, era_maxx, era_maxy)
                        logger.info(f"[DengueDataset] ERA5 bounds from metadata: {era5_bounds}")
            else:
                # SLOW: Load data to compute bounds (original behavior)
                logger.info("[DengueDataset] Computing ERA5 bounds by loading data...")
                var0 = list(self.era5.ds.data_vars)[0]
                da = self.era5.ds[var0]

                # Remove any time-like dimension
                if "time" in da.dims:
                    da = da.isel(time=0)

                # Drop non-spatial dims safely
                spatial_dims = [d for d in da.dims if d in ("y", "x")]
                da = da.transpose(*spatial_dims)

                arr = da.values

                # If still >2D (e.g., pressure level), reduce
                while arr.ndim > 2:
                    arr = arr[0]

                valid_mask = ~np.isnan(arr)

                if valid_mask.any():
                    ys, xs = np.where(valid_mask)

                    x_coords = self.era5.ds.x.values[xs]
                    y_coords = self.era5.ds.y.values[ys]

                    era_minx, era_maxx = float(x_coords.min()), float(x_coords.max())
                    era_miny, era_maxy = float(y_coords.min()), float(y_coords.max())

                    era5_bounds = (era_minx, era_miny, era_maxx, era_maxy)

                    logger.info(f"[DengueDataset] ERA5 valid bounds computed: {era5_bounds}")

        except Exception as e:
            logger.warning(f"[DengueDataset] Could not compute ERA5 bounds: {e}")
            era5_bounds = None

        self.era5_bounds = era5_bounds

        # ---- Intersect user bbox with ERA5 valid bounds ----
        if era5_bounds is not None:
            minx = max(ub_minx, era5_bounds[0])
            miny = max(ub_miny, era5_bounds[1])
            maxx = min(ub_maxx, era5_bounds[2])
            maxy = min(ub_maxy, era5_bounds[3])

            if minx >= maxx or miny >= maxy:
                logger.warning("[DengueDataset] No overlap with ERA5 bounds. Using user bbox.")
                self.bbox = [ub_minx, ub_miny, ub_maxx, ub_maxy]
            else:
                self.bbox = [minx, miny, maxx, maxy]
                logger.info(
                    f"[DengueDataset] Intersected bbox: "
                    f"(minx: {minx}, miny: {miny}, maxx: {maxx}, maxy: {maxy})"
                )
        else:
            logger.info("[DengueDataset] ERA5 bounds not available, using user bbox.")
            self.bbox = [ub_minx, ub_miny, ub_maxx, ub_maxy]


        # ---- Weekly → Monthly mapping ----
        self.time_pairs = []
        for t in self.y.times:
            t_week = pd.Timestamp(t)
            t_viirs = t_week.to_period("M").to_timestamp()
            self.time_pairs.append((t_week, t_viirs))

        # ---- Build spatial grid ONCE using random time ----
        sample_idx = 5

        sampler = GridGeoSampler(
            self.viirs,
            size=self.patch_size,
            roi=self.bbox,
            toi=pd.Interval(self.time_pairs[sample_idx][1], 
                            self.time_pairs[sample_idx][0]),
        )

        logger.info("VIIRS x,y range: (%f, %f, %f, %f)", viirs.bounds[0].start, viirs.bounds[0].stop, viirs.bounds[1].start, viirs.bounds[1].stop)
        logger.info("ERA5 x,y range: (%f, %f, %f, %f)", float(self.era5.x_coords.min()), float(self.era5.x_coords.max()), float(self.era5.y_coords.min()), float(self.era5.y_coords.max()))
        logger.info("Output x,y range: (%f, %f, %f, %f)", float(self.y.ds.x.min()), float(self.y.ds.x.max()), float(self.y.ds.y.min()), float(self.y.ds.y.max()))
        logger.info("Sample time: %s", self.time_pairs[sample_idx][0])

        self.spatial_queries = list(sampler)

        if len(self.spatial_queries) == 0:
            raise RuntimeError("No spatial patches found!")
        
        self._compute_valid_patches()


    def __len__(self):
        return len(self.valid_indices)
    
    def is_mostly_nan(self, tensor, threshold=0.9):
        nan_ratio = torch.isnan(tensor).float().mean()
        return nan_ratio > threshold
    
    def _is_patch_valid(self, tensor, threshold=None):
        """Check if a patch has enough valid (non-NaN) pixels"""
        if tensor.numel() == 0:
            return False
        if threshold is None:
            threshold = self.y_valid_threshold
        valid_ratio = (~torch.isnan(tensor)).float().mean()
        return valid_ratio >= threshold


    def _pad_to_size(self, tensor, target_h, target_w):
        """
        Pads tensor [T, C, H, W] to target spatial size.
        Pads on the right and bottom only.
        """
        if tensor.ndim == 5:
            _, _, _, h, w = tensor.shape
        elif tensor.ndim == 4:
            _, _, h, w = tensor.shape
        elif tensor.ndim == 3:
            _, h, w = tensor.shape
        else:
            raise ValueError(f"Unexpected tensor shape {tensor.shape}")

        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h < 0 or pad_w < 0:
            # If larger (should not happen), crop
            if tensor.ndim == 5:
                tensor = tensor[:, :, :, :target_h, :target_w]
            elif tensor.ndim == 4:
                tensor = tensor[:, :, :target_h, :target_w]
            elif tensor.ndim == 3:
                tensor = tensor[:, :target_h, :target_w]
            return tensor

        # pad format: (left, right, top, bottom)
        padding = (0, pad_w, 0, pad_h)
        tensor = F.pad(tensor, padding, mode="constant", value=0)

        return tensor
        

    def _compute_valid_patches(self):
        """Compute and cache valid patch indices for faster subsequent runs"""

        # -----------------------------
        # Build robust cache filename
        # -----------------------------
        cache_file = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

            # hash time_pairs so cache invalidates if timestamps change
            time_hash = hashlib.md5(str(self.time_pairs).encode()).hexdigest()[:10]

            cache_name = (
                f"valid_indices_"
                f"{len(self.time_pairs)}t_"
                f"{len(self.spatial_queries)}s_"
                f"{time_hash}_v2.pkl"
            )

            cache_file = Path(self.cache_dir) / cache_name

        # -----------------------------
        # Try loading cache
        # -----------------------------
        if cache_file and cache_file.exists():
            try:
                logger.info(f"[DengueDataset] Loading cached valid patches from {cache_file}...")

                with open(cache_file, "rb") as f:
                    self.valid_indices = pickle.load(f)

                # Safety check: ensure cached indices are still valid
                max_time_idx = max(t for _, t in self.valid_indices)

                if max_time_idx >= len(self.time_pairs):
                    raise RuntimeError(
                        "Cached valid_indices incompatible with current time_pairs."
                    )

                logger.info(f"[DengueDataset] Loaded {len(self.valid_indices)} cached patches")

                return self.valid_indices

            except Exception as e:
                logger.warning(
                    f"[DengueDataset] Cache invalid or incompatible ({e}). Recomputing patches..."
                )

        # -----------------------------
        # Compute valid patches (vectorized: one xarray load per spatial patch)
        # -----------------------------
        logger.info("[DengueDataset] Computing valid spatiotemporal patches (vectorized)...")
        self.valid_indices = []

        var_name = self.y.variables[0] if hasattr(self.y, "variables") else list(self.y.ds.data_vars)[0]

        # y coordinate may be descending (lat) — sel direction must be flipped
        y_coords = self.y.ds.y.values
        y_descending = bool(y_coords[0] > y_coords[-1])

        for spatial_idx, bbox in tqdm(
            enumerate(self.spatial_queries),
            total=len(self.spatial_queries),
            desc="Checking valid patches over space and time",
        ):
            x_slice, y_slice = bbox[0], bbox[1]

            # Flip y slice for descending coordinate axis
            y_sel = (slice(y_slice.stop, y_slice.start) if y_descending
                     else slice(y_slice.start, y_slice.stop))

            ds_patch = self.y.ds.sel(
                x=slice(x_slice.start, x_slice.stop),
                y=y_sel,
            )

            if ds_patch.time.size == 0 or ds_patch.y.size == 0 or ds_patch.x.size == 0:
                continue

            # Load full (T, H, W) block for this spatial patch in one dask compute
            arr = ds_patch[var_name].values  # (T, H, W)

            if arr.size == 0:
                continue

            # Valid ratio per time step — fully vectorized, no Python loop
            valid_ratios = np.isfinite(arr).reshape(arr.shape[0], -1).mean(axis=1)  # (T,)

            for time_idx in np.where(valid_ratios >= self.y_valid_threshold)[0].tolist():
                # Fast rtree check: skip if VIIRS has no tile covering this bbox+month.
                # This is a metadata-only query (no pixel loading) — O(log n) per call.
                t_week, t_viirs = self.time_pairs[time_idx]
                try:
                    t_ts = pd.Timestamp(t_viirs).to_period("M").to_timestamp()
                    mint = t_ts.timestamp()
                    maxt = (t_ts + pd.offsets.MonthEnd(1)).timestamp()
                    viirs_bb = TorchGeoBoundingBox(
                        x_slice.start, x_slice.stop,
                        y_slice.start, y_slice.stop,
                        mint, maxt,
                    )
                    if not list(self.viirs.index.intersection(viirs_bb.to_tuple(), objects=False)):
                        continue
                except Exception:
                    pass  # if the check fails for any reason, keep the patch
                self.valid_indices.append((spatial_idx, time_idx))

        if len(self.valid_indices) == 0:
            raise RuntimeError("No valid spatiotemporal patches found!")

        # -----------------------------
        # Save cache (atomic write via tmp + rename, safe under DDP)
        # -----------------------------
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_file = cache_file.with_suffix(".tmp")

                with open(tmp_file, "wb") as f:
                    pickle.dump(self.valid_indices, f)
                tmp_file.replace(cache_file)  # atomic on POSIX / lustre

                logger.info(
                    f"[DengueDataset] Cached {len(self.valid_indices)} patches to {cache_file}"
                )

            except Exception as e:
                logger.warning(f"[DengueDataset] Failed to save cache: {e}")

        return self.valid_indices

    def __getitem__(self, idx):
        
        spatial_idx, time_idx = self.valid_indices[idx]
        bbox = self.spatial_queries[spatial_idx]
        x_slice, y_slice = bbox[0], bbox[1]
        t_week, t_viirs = self.time_pairs[time_idx]

        # DataLoader workers inherit the NCCL process group after fork(); calling
        # dist.get_rank() from a worker is safe only for reading the stored int,
        # but using dist.is_initialized() as a guard is sufficient.
        _winfo = torch.utils.data.get_worker_info()
        rank = 0 if _winfo is not None or not dist.is_initialized() else dist.get_rank()
        timings = {}

        static_query = (x_slice, y_slice)
        query_viirs = (x_slice, y_slice, slice(
            pd.Timestamp(t_viirs).to_period("M").to_timestamp(),
            pd.Timestamp(t_viirs).to_period("M").to_timestamp()
        ))
        query_current = (x_slice, y_slice, slice(t_week, t_week))
        query_previous_n_days = (x_slice, y_slice, slice(
            pd.Timestamp(t_week) - pd.Timedelta(days=self.era5.T_max * 7 + self.era5.T_offset),
            pd.Timestamp(t_week) - pd.Timedelta(days=self.era5.T_offset)
        ))
        try:
            t0 = time.perf_counter()
            x_high    = self._pad_to_size(self.viirs[query_viirs]["image"].float(), *self.patch_size)
            timings["viirs"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            x_med     = self._pad_to_size(self.era5[query_previous_n_days]["image"].float(), *self.patch_size_era5)
            timings["era5"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            x_static  = self._pad_to_size(self.static[static_query]["image"].float(), *self.patch_size_static)
            x_lc = None
            if self.land_cover is not None:
                query_lc = (x_slice, y_slice, t_week)
                x_lc = self._pad_to_size(
                    self.land_cover[query_lc]["image"].float(), *self.patch_size_static
                )   # [1, H, W] float class IDs
                # Keep as integer class IDs for embedding lookup; replace nodata (nan→0)
                x_lc = torch.nan_to_num(x_lc, nan=0.0).long()  # [1, H, W]
            timings["static"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            x_cond    = self._pad_to_size(self.spatial_conditioning[query_current]["image"].float(), *self.patch_size_spatial_cond)
            timings["cond"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            y_spatial = self._pad_to_size(self.y[query_current]["image"].float(), *self.patch_size_y)
            timings["y"] = time.perf_counter() - t0

            # Optional soil moisture branch
            x_sm = None
            if self.soil_moisture is not None:
                t0 = time.perf_counter()
                query_sm = (x_slice, y_slice, t_week)
                x_sm = self.soil_moisture[query_sm]["image"].float()
                # patch_size_sm=None means use native zarr resolution as-is
                if self.patch_size_sm is not None:
                    x_sm = self._pad_to_size(x_sm, *self.patch_size_sm)
                timings["sm"] = time.perf_counter() - t0

            total = sum(timings.values())
            if total > 5.0:
                logger.warning(
                    f"[Rank {rank}] SLOW sample idx={idx} spatial={spatial_idx} time={time_idx} "
                    f"total={total:.2f}s | " +
                    " | ".join(f"{k}={v:.2f}s" for k, v in timings.items())
                )

            if self.loss_fn == "poisson":
                # Aggregate y from pixel-level [1, H, W] to zone-level [num_zones]
                y_2d = y_spatial.view(y_spatial.shape[-2], y_spatial.shape[-1]).squeeze(0) # [H, W]
                zone_map = x_cond.view(x_cond.shape[-2], x_cond.shape[-1]).squeeze(0).long()   # [H, W]

                zone_sums   = torch.zeros(self.num_zones, dtype=torch.float32)
                zone_counts = torch.zeros(self.num_zones, dtype=torch.float32)

                valid = torch.isfinite(y_2d)
                if valid.any():
                    zm = zone_map[valid].view(-1)
                    yv = y_2d[valid].float().view(-1)
                    # Drop any remaining -1s that slipped through
                    pos = zm >= 0
                    zone_sums.scatter_add_(0, zm[pos], yv[pos])
                    zone_counts.scatter_add_(0, zm[pos], torch.ones_like(yv[pos]))
                # NaN for zones absent from this patch → masked out in loss
                y_val = torch.where(
                    zone_counts > 0,
                    zone_sums / zone_counts,
                    torch.full_like(zone_sums, float('nan'))
                )  # [num_zones]

                # Population exposure map for this patch/year — [H, W]
                pop_map = None
                if self.population is not None:
                    H_y, W_y = self.patch_size_y
                    pop_map = self.population[(x_slice, y_slice, t_week)]  # [H, W]
                    if pop_map.shape != (H_y, W_y):
                        pop_map = self._pad_to_size(pop_map.unsqueeze(0), H_y, W_y).squeeze(0)

            else:
                y_val = y_spatial  # [1, H, W] — original pixel-level behaviour
                pop_map = None     # not used outside poisson mode

            if x_sm is not None and x_lc is not None:
                if pop_map is not None:
                    return x_high, x_med, x_static, x_cond, x_sm, x_lc, pop_map, y_val
                return x_high, x_med, x_static, x_cond, x_sm, x_lc, y_val
            if x_sm is not None:
                if pop_map is not None:
                    return x_high, x_med, x_static, x_cond, x_sm, pop_map, y_val
                return x_high, x_med, x_static, x_cond, x_sm, y_val
            if x_lc is not None:
                if pop_map is not None:
                    return x_high, x_med, x_static, x_cond, x_lc, pop_map, y_val
                return x_high, x_med, x_static, x_cond, x_lc, y_val
            if pop_map is not None:
                return x_high, x_med, x_static, x_cond, pop_map, y_val
            return x_high, x_med, x_static, x_cond, y_val

        except Exception as e:
            logger.error(f"Error loading patch at spatial_idx={spatial_idx}, time_idx={time_idx}: {e}")
            return None


# -----------------------------------------------------------------------------
# Collate function (skip None)
# -----------------------------------------------------------------------------
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

def worker(args):
    world_size = args.num_gpus * args.num_nodes
    global_rank = args.node_id * args.num_gpus + args.local_rank

    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(device)

    torch.dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank= global_rank,
        device_id=device)
    
    return device, world_size