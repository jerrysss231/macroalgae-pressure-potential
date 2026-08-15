"""Shared gridded-environment helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


TARGET_RESOLUTION_DEG = 0.25


def _latlon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat = next((x for x in ("lat", "latitude", "Latitude", "y") if x in ds.coords or x in ds.dims), None)
    lon = next((x for x in ("lon", "longitude", "Longitude", "x") if x in ds.coords or x in ds.dims), None)
    if lat is None or lon is None:
        raise ValueError(f"No latitude/longitude coordinates found: {list(ds.coords)}")
    return lat, lon


def _pick_variable(ds: xr.Dataset, hints: tuple[str, ...]) -> str:
    variables = [v for v in ds.data_vars if "bounds" not in v.lower() and "bnds" not in v.lower()]
    for hint in hints:
        for variable in variables:
            if variable.lower() == hint.lower() or hint.lower() in variable.lower():
                return variable
    if len(variables) == 1:
        return variables[0]
    raise ValueError(f"Cannot identify variable from hints={hints}; variables={variables}")


def normalize_longitude(da: xr.DataArray) -> xr.DataArray:
    if np.nanmax(da["lon"].values) > 180:
        da = da.assign_coords(lon=((da["lon"].values + 180) % 360) - 180).sortby("lon")
    return da


def _resolution(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype="float64"))
    diff = np.diff(values[np.isfinite(values)])
    diff = diff[(diff > 0) & np.isfinite(diff)]
    return float(np.nanmedian(diff)) if diff.size else np.nan


def coarsen_mean(da: xr.DataArray, target: float = TARGET_RESOLUTION_DEG) -> xr.DataArray:
    lat_factor = max(int(round(target / _resolution(da["lat"].values))), 1)
    lon_factor = max(int(round(target / _resolution(da["lon"].values))), 1)
    if lat_factor == lon_factor == 1:
        return da.astype("float32")
    out = da.coarsen(
        lat=lat_factor,
        lon=lon_factor,
        boundary="trim",
        coord_func="mean",
    ).mean(skipna=True)
    return normalize_longitude(out.sortby("lat").astype("float32"))


def open_2d(
    path: Path,
    hints: tuple[str, ...],
    *,
    template: xr.DataArray | None = None,
    target_resolution: float | None = TARGET_RESOLUTION_DEG,
) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path, decode_times=False)
    da = ds[_pick_variable(ds, hints)]
    lat, lon = _latlon_names(ds)
    rename = {}
    if lat != "lat":
        rename[lat] = "lat"
    if lon != "lon":
        rename[lon] = "lon"
    if rename:
        da = da.rename(rename)
    for dim in [d for d in da.dims if d not in ("lat", "lon")]:
        da = da.mean(dim=dim, skipna=True)
    da = normalize_longitude(da.squeeze(drop=True).sortby("lat").astype("float32"))
    if target_resolution is not None:
        da = coarsen_mean(da, target_resolution)
    if template is not None:
        da = da.interp(lat=template["lat"], lon=template["lon"], method="nearest")
    return da


def clean_values(
    values: np.ndarray,
    minimum: float | None = None,
    maximum: float | None = None,
    fill_threshold: float = -9000,
) -> np.ndarray:
    out = np.asarray(values, dtype="float32").copy()
    out[out <= fill_threshold] = np.nan
    if minimum is not None:
        out[out < minimum] = np.nan
    if maximum is not None:
        out[out > maximum] = np.nan
    return out


def area_weight(latitude: np.ndarray) -> np.ndarray:
    weight = np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))
    weight[~np.isfinite(weight) | (weight <= 0)] = np.nan
    return weight
