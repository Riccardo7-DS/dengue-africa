import torch
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader, random_split
from utils import init_logging


def plot_learning_curves(train_losses, val_losses, plot_dir, logger):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = plot_dir / "learning_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Learning curves saved to {plot_path}")
    plt.close()


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    return start_epoch, ckpt


def setup_run_dirs_and_logger(config, default_save_dir, run_prefix="run"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(config.get("save_dir", default_save_dir))
    run_dir = base_dir / f"{run_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    plot_dir = run_dir / "plots"

    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    logger = init_logging(log_file=log_dir / "training.log", verbose=False)
    return run_dir, checkpoint_dir, log_dir, plot_dir, logger


def setup_device_and_distributed(args, cfg, logger):
    import torch.distributed as dist
    from models import worker

    is_ddp = args.ddp and dist.is_available()

    if is_ddp:
        if dist.is_available() and dist.is_initialized():
            device = torch.device(f"cuda:{args.local_rank}")
            world_size = dist.get_world_size()
        else:
            device, world_size = worker(args)
        logger.info(
            f"Using DDP: device={device}, world_size={world_size}, local_rank={args.local_rank}"
        )
    else:
        device = torch.device(cfg["device"])
        world_size = None
        logger.info(f"Using single GPU: device={device}")

    return is_ddp, device, world_size


def wrap_model_for_parallel(model, args, device):
    if args.ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model.to(device), device_ids=[args.local_rank])
    else:
        from torch.nn import DataParallel
        return DataParallel(model.to(device))


def build_train_val_loaders(
    full_dataset,
    train_split,
    batch_size,
    num_workers,
    collate_fn,
    is_ddp=False,
    world_size=None,
    local_rank=0,
):
    from torch.utils.data.distributed import DistributedSampler

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    def _worker_init_fn(_):
        torch.set_num_threads(1)

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(not is_ddp),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = False
        val_loader_kwargs["persistent_workers"] = False

    train_sampler = None
    val_sampler = None

    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
        )
        train_loader_kwargs["sampler"] = train_sampler
        val_loader_kwargs["sampler"] = val_sampler
        del train_loader_kwargs["shuffle"]

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    return train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler


def standardize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    return (x - mean) / (std + eps)


def destandardize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    return x * std + mean


def compute_channelwise_mean_std_btchw(loader, device, tensor_index=1, ensure_fn=None, logger=None):
   
    channel_sum = None
    channel_sq_sum = None
    n_total = 0

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue

        x = batch[tensor_index].to(device, non_blocking=True)

        if ensure_fn is not None:
            x = ensure_fn(x)

        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if channel_sum is None:
            C = x.shape[2]
            channel_sum = torch.zeros(C, device=device, dtype=torch.float64)
            channel_sq_sum = torch.zeros(C, device=device, dtype=torch.float64)

        channel_sum += x.sum(dim=(0, 1, 3, 4), dtype=torch.float64)
        channel_sq_sum += (x ** 2).sum(dim=(0, 1, 3, 4), dtype=torch.float64)
        n_total += x.shape[0] * x.shape[1] * x.shape[3] * x.shape[4]

    if n_total == 0:
        raise RuntimeError("Impossibile calcolare mean/std: nessun sample valido trovato")

    mean = channel_sum / n_total
    var = channel_sq_sum / n_total - mean ** 2
    var = torch.clamp(var, min=1e-8)
    std = torch.sqrt(var)

    mean = mean.view(1, 1, -1, 1, 1).float()
    std = std.view(1, 1, -1, 1, 1).float()

    if logger is not None:
        logger.info(f"Computed mean with shape {tuple(mean.shape)}")
        logger.info(f"Computed std with shape {tuple(std.shape)}")

    return mean, std

def create_logger(log_dir, is_main_process=True):
    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger("sample_dit_latent")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        fh = logging.FileHandler(os.path.join(log_dir, "sample.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
    else:
        logger = logging.getLogger("sample_dit_latent")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        return logger

def normalize_to_neg_one_one_with_minmax_ignore_nan(x, data_min, data_max, fill_value=-2.0):
    
    data_min = torch.as_tensor(data_min, device=x.device, dtype=x.dtype)
    data_max = torch.as_tensor(data_max, device=x.device, dtype=x.dtype)
    denom = torch.clamp(data_max - data_min, min=1e-6)

    valid_mask = ~torch.isnan(x)

    x_norm = x.clone()
    x_norm[valid_mask] = 2.0 * (x[valid_mask] - data_min) / denom - 1.0
    x_norm[~valid_mask] = fill_value

    return x_norm


def denormalize_from_neg_one_one(x, data_min, data_max):
    data_min = torch.as_tensor(data_min, device=x.device, dtype=x.dtype)
    data_max = torch.as_tensor(data_max, device=x.device, dtype=x.dtype)
    return 0.5 * (x + 1.0) * (data_max - data_min) + data_min


def ensure_bchw_CxHxW(x: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input is not a torch.Tensor, got: {type(x)}")

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        if x.shape[0] == 1:
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)
    elif x.ndim == 4:
        pass
    elif x.ndim == 5:
        if x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.shape[2] == 1:
            x = x.squeeze(2)
        else:
            raise ValueError(f"Incompatible 5D input shape: {tuple(x.shape)}")
    else:
        raise ValueError(f"Unsupported number of dimensions: {tuple(x.shape)}")

    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor after reshape, got: {tuple(x.shape)}")

    if x.shape[1] != C:
        raise ValueError(
            f"Expected a single channel, but got shape {tuple(x.shape)}. "
            f"C={x.shape[1]} instead of 1."
        )

    if x.shape[2] != H or x.shape[3] != W:
        raise ValueError(f"Expected H={H}, W={W}, but got shape {tuple(x.shape)}")

    return x