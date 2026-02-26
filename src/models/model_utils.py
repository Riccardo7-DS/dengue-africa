"""
Functions for DL models
"""

import os 
import logging 
import numpy as np 
import torch
from torch.utils.data import Subset
import pandas as pd
from torchgeo.samplers import GridGeoSampler
import xarray as xr
import torch
import torch.nn.functional as F

from torchgeo.datasets import RasterDataset
from torchgeo.samplers import GridGeoSampler

from tqdm import tqdm

from typing import Optional, Literal

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
        data = data.clone().float()  # no clone — already cloned before calling this

        # Single-pass sentinel + nan + inf replacement
        # Handles -3.4e38 ESRI nodata, NaN, and Inf in one operation
        sentinel_mask = data.abs() > 3.3e38
        if sentinel_mask.any():  # .any() stays on GPU, no .item() sync
            data = data.masked_fill(sentinel_mask, replace_nan)

        # One call handles all remaining NaN/Inf
        data = torch.nan_to_num(data, nan=replace_nan, 
                                posinf=replace_nan, neginf=replace_nan)

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

class CustomMetrics():
    def __init__(self,
                 preds:Union[torch.tensor, np.ndarray],
                 labels:Union[torch.tensor, np.ndarray], 
                 metric_lists:Union[list, str],
                 mask = Union[None, torch.tensor, np.ndarray],
                 masked:bool=False):
        
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if masked:
            self._count_masked_pixels(mask)

        self.mask = mask
        if masked and mask is None:
            logger.error(RuntimeError("No provided mask but chosen masked loss option"))

        self.losses, self.metrics = self._get_metrics(metric_lists, preds, labels, masked)

    def _get_metrics(self, metric_list, preds, labels, masked):
        if isinstance(metric_list, list):
            results = []
            metrics = []
            for metric in metric_list:
                res, m = self._apply_metric(metric, preds, labels)
                if masked:
                    res = self._metric_masking(res, self.mask)
                results.append(res)
                metrics.append(m)
            return results, metrics
        
        elif isinstance(metric_list, str):
            results, m = self._apply_metric(metric_list, preds, labels)
            if masked:
                results = self._metric_masking(results, self.mask)
            return [results], [m]
    
    def _apply_metric(self, metric, preds, labels):
        if metric == "rmse":
            loss = torch.sqrt((preds-labels)**2)
        elif metric == "bias":
            loss = preds - labels
        
        elif metric == "mse":
            loss = (preds-labels)**2

        elif metric == "mae":
            loss = torch.abs(preds-labels)

        elif metric == "mape":
            loss = torch.abs((preds-labels)/labels)
        else:
            logger.warning(f"Metric {metric} not recognized")
            loss = None

        return loss, metric
    
    def _count_masked_pixels(self, mask):
        w, h = mask.shape
        good_pixels = mask.sum()
        tot_pixels = w*h
        logger.info(f"{(1-good_pixels/tot_pixels):.2%} of the pixels are masked in the loss computation")
        

    def _metric_masking(self, loss, mask, return_value=True):
        if isinstance(loss, np.ndarray):
            loss = torch.from_numpy(loss)

        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(null_loss), torch.tensor(0.0), null_loss)

        if return_value:
            non_zero_elements = full_mask.sum()
            return (null_loss.sum() / non_zero_elements).item()
        else:
            if len(loss.shape)>2:
                return loss.mean(0)
            else:
                return loss

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

    def zone_aggregate(self, pred, zone_map):
        B = pred.shape[0]
        pred = pred.squeeze(1).float()          # [B, H, W]

        # Handle any shape [..., H, W] → [B, H, W]
        H, W = pred.shape[-2], pred.shape[-1]
        zone_map = zone_map.view(B, H, W).long()

        zone_preds  = torch.zeros(B, self.num_zones, device=pred.device, dtype=torch.float32)
        zone_counts = torch.zeros(B, self.num_zones, device=pred.device, dtype=torch.float32)

        # Valid mask: exclude -1 (out-of-zone) and out-of-bounds
        valid = (zone_map >= 0) & (zone_map < self.num_zones)  # [B, H, W]

        for b in range(B):
            zm = zone_map[b][valid[b]].view(-1)
            pv = pred[b][valid[b]].view(-1)
            zone_preds[b].scatter_add_(0, zm, pv)
            zone_counts[b].scatter_add_(0, zm, torch.ones_like(pv))

        zone_means = zone_preds / (zone_counts + 1e-8)
        return zone_means, zone_counts

    def zone_loss(self, pred, target, zone_map):
        """
        pred:     [B, 1, H, W]  — model output (log-counts)
        target:   [B, num_zones] — ground truth zone-level case counts
        zone_map: [B, H, W]     — integer zone IDs
        """
        zone_preds, zone_counts = self.zone_aggregate(pred, zone_map)

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
        **kwargs,
    ):
        super().__init__(root, **kwargs)

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
        transform=None,
        min_date=None,
        max_date=None,
    ):
        import xarray as xr
        from pathlib import Path

        self.root = Path(root)
        self.variables = variables
        self.T_max = T_max
        self.transform = transform

        files = sorted(self.root.glob("era5land_latin_america*.nc"))
        if not files:
            raise RuntimeError("No ERA5 files found")

        # 🔹 open lazily with chunking
        self.ds = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            chunks={"time": 1},  # <- key for daily loading
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

        # --- Select subset lazily ---
        ds_sel = (
            self.ds[self.vars_]
            .sel(
                time=slice(t_slice.start, t_slice.stop),
                x=slice(x_slice.start, x_slice.stop),
                y=slice(y_slice.start, y_slice.stop),
            )
        )

        ds_sel = self._era5_to_weekly(ds_sel)  # Resample to weekly on the fly

        # --- Convert to array lazily ---
        da = ds_sel.to_array()  # [C,T,H,W] lazy

        # --- Move time first ---
        da = da.transpose("time", "variable", "y", "x")

        # --- Compute ONLY this subset ---
        arr = da.compute().values  # 🔥 loads only selected window

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
            "spatial_conditioning":(86, 86)
        },
        y_valid_threshold = 0.3,
        loss_fn:Literal["mse", "poisson"] = "mse",
        num_zones=None,
        bbox=None,
        skip_era5_bounds=False,
        cache_dir=None):



        self.viirs = viirs
        self.era5 = era5
        self.static = static
        self.y = y
        self.spatial_conditioning = spatial_conditioning
        self.cache_dir = cache_dir
        self.skip_era5_bounds = skip_era5_bounds
        self.loss_fn = loss_fn
        self.num_zones = num_zones

        if self.loss_fn == "poisson" and num_zones is None:
            raise ValueError("num_zones must be provided for poisson loss")

        from utils import latin_box

        self.patch_size = patch_sizes.get("viirs", (1024, 1024))
        self.patch_size_era5 = patch_sizes.get("era5", (43, 43))
        self.patch_size_static = patch_sizes.get("static", (512, 512))
        self.patch_size_y = patch_sizes.get("y", (86, 86))
        self.patch_size_spatial_cond = patch_sizes.get("spatial_conditioning", (86, 86))

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
        import pickle
        from pathlib import Path
        
        # Determine cache file path
        cache_file = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = Path(self.cache_dir) / "valid_patches_cache.pkl"
        
        # Try to load from cache first
        if cache_file and cache_file.exists():
            try:
                logger.info(f"[DengueDataset] Loading cached valid patches from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    self.valid_indices = pickle.load(f)
                logger.info(f"[DengueDataset] Loaded {len(self.valid_indices)} cached patches")
                return self.valid_indices
            except Exception as e:
                logger.warning(f"[DengueDataset] Failed to load cache: {e}. Computing patches...")
        
        # Compute valid patches (same logic as before)
        logger.info("[DengueDataset] Computing valid spatiotemporal patches...")
        self.valid_indices = []

        for spatial_idx, bbox in tqdm(enumerate(self.spatial_queries), total=len(self.spatial_queries), desc="Checking valid patches over space and time"):

            x_slice, y_slice = bbox[0], bbox[1]

            for time_idx, (t_week, _) in enumerate(self.time_pairs):

                query_xarray = (x_slice, y_slice, slice(t_week, t_week))
                y_patch = self.y[query_xarray]["image"].float()

                if y_patch.numel() == 0:
                    continue

                valid_ratio = (~torch.isnan(y_patch)).float().mean()

                if valid_ratio >= self.y_valid_threshold:
                    self.valid_indices.append((spatial_idx, time_idx))

        if len(self.valid_indices) == 0:
            raise RuntimeError("No valid spatiotemporal patches found!")
        
        # Save to cache for future runs
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.valid_indices, f)
                logger.info(f"[DengueDataset] Cached {len(self.valid_indices)} patches to {cache_file}")
            except Exception as e:
                logger.warning(f"[DengueDataset] Failed to save cache: {e}")
        
        return self.valid_indices

    def __getitem__(self, idx):
        
        spatial_idx, time_idx = self.valid_indices[idx]
        bbox = self.spatial_queries[spatial_idx]
        x_slice, y_slice = bbox[0], bbox[1]
        t_week, t_viirs = self.time_pairs[time_idx]

        static_query = (x_slice, y_slice)
        query_viirs = (x_slice, y_slice, slice(
            pd.Timestamp(t_viirs).to_period("M").to_timestamp(),
            pd.Timestamp(t_viirs).to_period("M").to_timestamp()
        ))
        query_xarray = (x_slice, y_slice, slice(t_week, t_week))

        try:
            x_high    = self._pad_to_size(self.viirs[query_viirs]["image"].float(), *self.patch_size)
            x_med     = self._pad_to_size(self.era5[query_xarray]["image"].float(), *self.patch_size_era5)
            x_static  = self._pad_to_size(self.static[static_query]["image"].float(), *self.patch_size_static)
            x_cond    = self._pad_to_size(self.spatial_conditioning[query_xarray]["image"].float(), *self.patch_size_spatial_cond)
            y_spatial = self._pad_to_size(self.y[query_xarray]["image"].float(), *self.patch_size_y)

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

            else:
                y_val = y_spatial  # [1, H, W] — original pixel-level behaviour

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


