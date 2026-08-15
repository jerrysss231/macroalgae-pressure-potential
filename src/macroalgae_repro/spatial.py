"""Shared helpers for gridded environmental data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


TARGET_RESOLUTION_DEG = 0.25


def _latlon_names(dataset: xr.Dataset) -> tuple[str, str]:
    latitude = next(
        (
            name
            for name in ("lat", "latitude", "Latitude", "y")
            if name in dataset.coords or name in dataset.dims
        ),
        None,
    )
    longitude = next(
        (
            name
            for name in ("lon", "longitude", "Longitude", "x")
            if name in dataset.coords or name in dataset.dims
        ),
        None,
    )
    if latitude is None or longitude is None:
        raise ValueError(
            "No latitude/longitude coordinates found: "
            f"{list(dataset.coords)}"
        )
    return latitude, longitude


def _pick_variable(dataset: xr.Dataset, hints: tuple[str, ...]) -> str:
    variables = [
        variable
        for variable in dataset.data_vars
        if "bounds" not in variable.lower() and "bnds" not in variable.lower()
    ]
    for hint in hints:
        for variable in variables:
            if variable.lower() == hint.lower() or hint.lower() in variable.lower():
                return variable
    if len(variables) == 1:
        return variables[0]
    raise ValueError(
        f"Cannot identify variable from hints={hints}; variables={variables}"
    )


def normalize_longitude(data: xr.DataArray) -> xr.DataArray:
    """Represent longitudes on the [-180, 180) interval."""
    if np.nanmax(data["lon"].values) > 180:
        longitude = ((data["lon"].values + 180) % 360) - 180
        data = data.assign_coords(lon=longitude).sortby("lon")
    return data


def _resolution(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype="float64"))
    difference = np.diff(values[np.isfinite(values)])
    difference = difference[(difference > 0) & np.isfinite(difference)]
    return float(np.nanmedian(difference)) if difference.size else np.nan


def coarsen_mean(
    data: xr.DataArray,
    target: float = TARGET_RESOLUTION_DEG,
) -> xr.DataArray:
    """Mean-coarsen a regular latitude-longitude grid to the target resolution."""
    latitude_factor = max(int(round(target / _resolution(data["lat"].values))), 1)
    longitude_factor = max(int(round(target / _resolution(data["lon"].values))), 1)
    if latitude_factor == longitude_factor == 1:
        return data.astype("float32")

    output = data.coarsen(
        lat=latitude_factor,
        lon=longitude_factor,
        boundary="trim",
        coord_func="mean",
    ).mean(skipna=True)
    return normalize_longitude(output.sortby("lat").astype("float32"))


def open_2d(
    path: Path,
    hints: tuple[str, ...],
    *,
    template: xr.DataArray | None = None,
    target_resolution: float | None = TARGET_RESOLUTION_DEG,
) -> xr.DataArray:
    """Open, normalize and optionally align one environmental raster."""
    if not path.exists():
        raise FileNotFoundError(path)

    dataset = xr.open_dataset(path, decode_times=False)
    data = dataset[_pick_variable(dataset, hints)]
    latitude, longitude = _latlon_names(dataset)
    rename = {}
    if latitude != "lat":
        rename[latitude] = "lat"
    if longitude != "lon":
        rename[longitude] = "lon"
    if rename:
        data = data.rename(rename)

    for dimension in [dim for dim in data.dims if dim not in ("lat", "lon")]:
        data = data.mean(dim=dimension, skipna=True)
    data = normalize_longitude(
        data.squeeze(drop=True).sortby("lat").astype("float32")
    )
    if target_resolution is not None:
        data = coarsen_mean(data, target_resolution)
    if template is not None:
        data = data.interp(
            lat=template["lat"],
            lon=template["lon"],
            method="nearest",
        )
    return data


def clean_values(
    values: np.ndarray,
    minimum: float | None = None,
    maximum: float | None = None,
    fill_threshold: float = -9000,
) -> np.ndarray:
    """Replace fill values and values outside the accepted range with NaN."""
    output = np.asarray(values, dtype="float32").copy()
    output[output <= fill_threshold] = np.nan
    if minimum is not None:
        output[output < minimum] = np.nan
    if maximum is not None:
        output[output > maximum] = np.nan
    return output


def area_weight(latitude: np.ndarray) -> np.ndarray:
    """Return cosine-latitude area weights for a regular lon-lat grid."""
    weight = np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))
    weight[~np.isfinite(weight) | (weight <= 0)] = np.nan
    return weight
