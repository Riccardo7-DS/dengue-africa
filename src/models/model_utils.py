"""
Functions for DL models
"""

import os 
import logging 
import numpy as np 
import torch
from torch.utils.data import Subset
import pandas as pd

logger = logging.getLogger(__name__)


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
            verbose (bool): If True, prints a message for each validation loss improvement. 
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
        # self._cleanup_checkpoints(save_path, n_save)


def collate_skip_none(batch):
    # filter out None items
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


import torch
import torch.nn.functional as F

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
            

def masked_custom_loss(criterion, preds, labels, mask=None, return_value=True):
    loss = criterion(preds, labels)
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