"""Small weighted-statistics helpers shared across analysis stages."""

from __future__ import annotations

import numpy as np


def weighted_mean(values, weights, valid=None) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    if not ok.any():
        return np.nan
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok]))


def weighted_fraction(flag, valid, weights) -> float:
    flag = np.asarray(flag, bool)
    valid = np.asarray(valid, bool)
    weights = np.asarray(weights, float)
    ok = valid & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    return float(weights[ok & flag].sum() / weights[ok].sum())


def weighted_quantile(values, quantile: float, weights) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    values, weights = values[ok], weights[ok]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return float(np.interp(quantile, cdf, values))
