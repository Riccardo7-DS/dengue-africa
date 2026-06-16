"""
aedes_suitability.py

Monthly climatic suitability index for Aedes mosquito activity.

Two formulations, depending on available ERA5 variables:

A) Full atmospheric formulation (requires t2m, d2m, tp):

        S = ST × SH × SR

   ST  temperature suitability — Gaussian centred at 29 °C
   SH  atmospheric humidity suitability — logistic on RH [%] derived from
       t2m / d2m via the Magnus formula
   SR  rainfall suitability — 1 − exp(−P/80), P in mm/month

B) Land-surface formulation (ERA5-Land reduced set: skt, pev, e, tp):

        S = ST × SM × SR

   ST  same temperature term using skin temperature (skt)
   SM  land-surface moisture suitability — logistic on evaporative fraction
       EF = |e / pev|, capturing desiccation stress and larval habitat
       persistence rather than atmospheric humidity directly
   SR  same rainfall term

The EF-based SM term is NOT a substitute for atmospheric RH.  It encodes
a distinct ecological constraint: the degree to which the land surface is
moisture-limited vs. energy-limited.  High EF → wet surface → low
desiccation stress → higher mosquito persistence and larval site survival.
When t2m + d2m are available, use formulation A (SH); otherwise use B (SM).

Entry points
------------
compute_suitability(T, RH, P, lag_P)
    Formulation A: atmospheric RH. Units: T °C, RH %, P mm/month.

compute_suitability_surface(T, EF, P, lag_P)
    Formulation B: evaporative fraction. Units: T °C, EF ∈ [0,1], P mm/month.

from_era5_dataset(ds, lag_months)
    Auto-selects A or B based on available variables.

load_monthly(era5_path, lag_months)
    Load, aggregate, compute S, return xarray DataArray.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path


# ── Core computation ──────────────────────────────────────────────────────────

def _magnus_sat(T_degC: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure [hPa] via Magnus formula."""
    return 6.112 * np.exp(17.67 * T_degC / (T_degC + 243.5))


def temperature_suitability(T: np.ndarray) -> np.ndarray:
    """ST: Gaussian centred at 29 °C, σ = 6 °C.  Input T in °C."""
    return np.clip(np.exp(-((T - 29.0) ** 2) / (2 * 36.0)), 0.0, 1.0)


def humidity_suitability(RH: np.ndarray) -> np.ndarray:
    """SH: atmospheric humidity suitability.  Logistic at 60 % RH, scale 10.
    Input RH in [0, 100]."""
    RH = np.clip(RH, 0.0, 100.0)
    return np.clip(1.0 / (1.0 + np.exp(-(RH - 60.0) / 10.0)), 0.0, 1.0)


def surface_moisture_suitability(EF: np.ndarray) -> np.ndarray:
    """SM: land-surface moisture suitability from evaporative fraction.

    EF = |actual evaporation / potential evapotranspiration| ∈ [0, 1].

    Captures the ecological constraint of desiccation stress: when EF is
    low (moisture-limited, dry surface) larval breeding sites evaporate
    quickly and adult survival is impaired; when EF is high (energy-limited,
    wet surface) microhabitat persistence is favoured.

    This is NOT a proxy for atmospheric RH — it is an independent
    land-surface hydrological constraint on mosquito viability.

    Logistic centred at EF = 0.4 with scale 0.1:
        EF = 0.1 (arid)   → SM ≈ 0.05
        EF = 0.4           → SM = 0.50
        EF = 0.7 (humid)  → SM ≈ 0.95
    """
    EF = np.clip(EF, 0.0, 1.0)
    return np.clip(1.0 / (1.0 + np.exp(-(EF - 0.4) / 0.1)), 0.0, 1.0)


def rainfall_suitability(P: np.ndarray) -> np.ndarray:
    """SR: 1 − exp(−P/80).  Input P in mm/month."""
    P = np.clip(P, 0.0, None)
    return np.clip(1.0 - np.exp(-P / 80.0), 0.0, 1.0)


def compute_suitability(
    T: np.ndarray,
    RH: np.ndarray,
    P: np.ndarray,
    lag_P: np.ndarray | None = None,
) -> np.ndarray:
    """Formulation A: S = ST × SH × SR (atmospheric RH path).

    Parameters
    ----------
    T     : temperature [°C], shape (..., H, W)
    RH    : relative humidity [%], shape (..., H, W)
    P     : total precipitation [mm/month], shape (..., H, W)
    lag_P : if given, SR uses this (previous-month precipitation) instead of P
    """
    ST = temperature_suitability(T)
    SH = humidity_suitability(RH)
    SR = rainfall_suitability(lag_P if lag_P is not None else P)
    return np.clip(ST * SH * SR, 0.0, 1.0).astype(np.float32)


def compute_suitability_surface(
    T: np.ndarray,
    EF: np.ndarray,
    P: np.ndarray,
    lag_P: np.ndarray | None = None,
) -> np.ndarray:
    """Formulation B: S = ST × SM × SR (land-surface moisture path).

    Parameters
    ----------
    T     : temperature [°C], shape (..., H, W)
    EF    : evaporative fraction |e/pev| ∈ [0, 1], shape (..., H, W)
    P     : total precipitation [mm/month], shape (..., H, W)
    lag_P : if given, SR uses this (previous-month precipitation) instead of P
    """
    ST = temperature_suitability(T)
    SM = surface_moisture_suitability(EF)
    SR = rainfall_suitability(lag_P if lag_P is not None else P)
    return np.clip(ST * SM * SR, 0.0, 1.0).astype(np.float32)


# ── ERA5-Land dataset helpers ─────────────────────────────────────────────────

def _rh_from_dewpoint(T_K: xr.DataArray, Td_K: xr.DataArray) -> xr.DataArray:
    """RH [%] from 2 m temperature and dewpoint (both in K)."""
    T  = T_K  - 273.15
    Td = Td_K - 273.15
    es_T  = 6.112 * np.exp(17.67 * T  / (T  + 243.5))
    es_Td = 6.112 * np.exp(17.67 * Td / (Td + 243.5))
    return (100.0 * es_Td / es_T).clip(0.0, 100.0)


def _evap_fraction(e: xr.DataArray, pev: xr.DataArray) -> xr.DataArray:
    """Compute evaporative fraction EF = |e / pev| ∈ [0, 1].

    ERA5-Land e (actual evaporation) and pev (potential evapotranspiration)
    are both accumulated fluxes in m, conventionally ≤ 0 (upward).
    EF → 1: surface is energy-limited (wet); EF → 0: moisture-limited (dry).
    """
    pev_abs = np.abs(pev)
    e_abs   = np.abs(e)
    return xr.where(pev_abs > 1e-8, e_abs / pev_abs, 0.0).clip(0.0, 1.0)


def from_era5_dataset(
    ds: xr.Dataset,
    lag_months: int = 0,
) -> xr.DataArray:
    """
    Compute monthly Aedes suitability from an ERA5-Land xarray Dataset.

    Auto-selects formulation based on available variables:

    A) Atmospheric formulation — S = ST × SH × SR
       Requires: t2m [K], d2m [K], tp [m]
       SH = humidity_suitability(RH) where RH from Magnus formula.

    B) Land-surface formulation — S = ST × SM × SR
       Requires: skt [K], pev [m], e [m], tp [m]
       SM = surface_moisture_suitability(EF) where EF = |e/pev|.
       SM encodes desiccation stress and larval habitat persistence —
       it is NOT a surrogate for atmospheric RH.

    Parameters
    ----------
    ds          : xarray Dataset with a time dimension (daily or sub-daily).
    lag_months  : if > 0, SR uses precipitation lagged by this many months
                  (antecedent rainfall for larval site formation).

    Returns
    -------
    S : xarray DataArray (time=monthly, lat, lon), float32, in [0, 1].
        'time' is the first day of each month.
        Attribute 'formulation' is 'A_atmospheric' or 'B_surface'.
    """
    # ── normalise coordinates ─────────────────────────────────────────────
    time_dim = next((c for c in ("time", "valid_time", "date") if c in ds.coords), None)
    if time_dim is None:
        raise ValueError("Dataset has no recognised time coordinate.")
    if time_dim != "time":
        ds = ds.rename({time_dim: "time"})

    rename = {}
    for src, dst in [("latitude", "lat"), ("longitude", "lon")]:
        if src in ds.coords and dst not in ds.coords:
            rename[src] = dst
    if rename:
        ds = ds.rename(rename)

    # ── monthly means (temperature, EF numerator/denominator) ────────────
    ds_m = ds.resample(time="MS").mean()

    # ── precipitation: monthly sum [mm] ──────────────────────────────────
    P_m = ds["tp"].resample(time="MS").sum() * 1000.0   # m → mm
    if lag_months > 0:
        P_m = P_m.shift(time=lag_months, fill_value=0.0)

    # ── choose formulation ────────────────────────────────────────────────
    has_atm = ("t2m" in ds_m) and ("d2m" in ds_m)
    has_sfc = ("skt" in ds_m) and ("e" in ds_m) and ("pev" in ds_m)

    if has_atm:
        formulation = "A_atmospheric"
        T  = (ds_m["t2m"] - 273.15).values
        RH = _rh_from_dewpoint(ds_m["t2m"], ds_m["d2m"]).values
        S  = compute_suitability(T, RH, P_m.values)
        ref_da = ds_m["t2m"]
        formula_str = "ST×SH×SR; ST=Gauss(T,29,6), SH=sigmoid((RH-60)/10), SR=1-exp(-P/80)"

    elif has_sfc:
        formulation = "B_surface"
        T  = (ds_m["skt"] - 273.15).values
        EF = _evap_fraction(ds_m["e"], ds_m["pev"]).values
        S  = compute_suitability_surface(T, EF, P_m.values)
        ref_da = ds_m["skt"]
        formula_str = (
            "ST×SM×SR; ST=Gauss(T_skt,29,6), "
            "SM=sigmoid((EF-0.4)/0.1) [land-surface moisture], "
            "SR=1-exp(-P/80)"
        )
    else:
        raise KeyError(
            "Need (t2m+d2m) for formulation A or (skt+e+pev) for formulation B. "
            f"Available vars: {list(ds.data_vars)}"
        )

    # ── wrap result ───────────────────────────────────────────────────────
    coords = {"time": ds_m.time}
    for c in ("lat", "lon", "latitude", "longitude", "y", "x"):
        if c in ref_da.coords:
            coords[c] = ref_da.coords[c]

    return xr.DataArray(
        S.astype(np.float32),
        dims=ref_da.dims,
        coords=coords,
        name="aedes_suitability",
        attrs={
            "long_name": "Monthly Aedes climatic suitability index",
            "units": "1",
            "formulation": formulation,
            "formula": formula_str,
        },
    )


def load_monthly(
    era5_path: str | Path,
    lag_months: int = 0,
) -> xr.DataArray:
    """
    Load an ERA5-Land NetCDF file, compute monthly Aedes suitability.

    Returns xarray DataArray S (time, lat, lon) in [0, 1].
    """
    ds = xr.open_dataset(era5_path)
    return from_era5_dataset(ds, lag_months=lag_months)
