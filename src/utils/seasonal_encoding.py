"""
seasonal_encoding.py

Cyclical sine/cosine encoding of calendar position for use as additional
channels in the ERA5 medium-resolution temporal branch.

For each weekly ERA5 time step the encoding broadcasts two scalar values
to the full spatial grid [H, W]:

    sin(2π · doy / days_in_year)
    cos(2π · doy / days_in_year)

where doy is the day-of-year of that week's timestamp.  These two channels
appended to the 18 ERA5 climate channels give the temporal self-attention
explicit position information so it can learn which weeks in the seasonal
cycle are most predictive for dengue transmission.

Public API
----------
append_seasonal_encoding(x_med, timestamps)
    Appends sin/cos channels to a numpy ERA5 array [T, C, H, W].
    Returns [T, C+2, H, W].  Used by both the dataset and the
    Africa inference script so the logic is guaranteed identical.

append_seasonal_encoding_tensor(x_med, timestamps)
    Same operation on a float32 torch.Tensor.  Used inside DengueDataset.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch


def _sin_cos(timestamp) -> tuple[float, float]:
    """Return (sin, cos) of day-of-year fraction for a single timestamp."""
    ts = pd.Timestamp(timestamp)
    doy = ts.day_of_year
    days_in_year = 366 if ts.is_leap_year else 365
    angle = 2.0 * math.pi * doy / days_in_year
    return math.sin(angle), math.cos(angle)


def append_seasonal_encoding(
    x_med: np.ndarray,
    timestamps: Sequence,
) -> np.ndarray:
    """Append sin/cos seasonal channels to a numpy ERA5 array.

    Parameters
    ----------
    x_med      : float32 ndarray, shape [T, C, H, W]
    timestamps : sequence of T timestamps (any format pd.Timestamp accepts),
                 one per weekly time step in x_med

    Returns
    -------
    float32 ndarray, shape [T, C+2, H, W]
    """
    T, C, H, W = x_med.shape
    if len(timestamps) != T:
        raise ValueError(f"len(timestamps)={len(timestamps)} != T={T}")

    encoding = np.empty((T, 2, H, W), dtype=np.float32)
    for i, ts in enumerate(timestamps):
        s, c = _sin_cos(ts)
        encoding[i, 0] = s
        encoding[i, 1] = c

    return np.concatenate([x_med, encoding], axis=1)   # [T, C+2, H, W]


def append_seasonal_encoding_tensor(
    x_med: torch.Tensor,
    timestamps: Sequence,
) -> torch.Tensor:
    """Append sin/cos seasonal channels to a torch ERA5 tensor.

    Parameters
    ----------
    x_med      : float32 tensor, shape [T, C, H, W]
    timestamps : sequence of T timestamps

    Returns
    -------
    float32 tensor, shape [T, C+2, H, W]
    """
    T, C, H, W = x_med.shape
    if len(timestamps) != T:
        raise ValueError(f"len(timestamps)={len(timestamps)} != T={T}")

    encoding = torch.empty(T, 2, H, W, dtype=torch.float32)
    for i, ts in enumerate(timestamps):
        s, c = _sin_cos(ts)
        encoding[i, 0] = s
        encoding[i, 1] = c

    return torch.cat([x_med, encoding], dim=1)          # [T, C+2, H, W]


def weekly_timestamps_for_window(t_end, n_weeks: int) -> list[pd.Timestamp]:
    """Return n_weeks approximate timestamps ending at t_end, stepping back 7 days each.

    Used when the individual week timestamps are not stored alongside the tensor
    (e.g. after loading from zarr).  The approximation is consistent with how
    ERA5Dataset selects weeks via query_previous_n_days.
    """
    t_end = pd.Timestamp(t_end)
    return [t_end - pd.Timedelta(days=(n_weeks - 1 - i) * 7) for i in range(n_weeks)]
