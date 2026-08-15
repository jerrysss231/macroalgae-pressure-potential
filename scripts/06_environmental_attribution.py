"""Attribute future response changes to five environmental-change gradients.

Associations are descriptive. Shapley values partition the R² of an area-weighted
linear model and therefore do not imply causal environmental effects.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths


PATHS = ProjectPaths.from_env()
OUTPUT_DIR = PATHS.output_dir
PIXEL_DIR = OUTPUT_DIR / "cross_scenario_comparison_and_mismatch_final" / "recalculated_pixels"
ATTR_DIR = OUTPUT_DIR / "future_environmental_attribution"
DRIVER_CSV = ATTR_DIR / "driver_delta_matrix.csv"

FUTURES = ("ssp245", "ssp370", "ssp585")
DRIVERS = ("Temperature", "NO3", "PO4", "pH", "Salinity")
RESPONSES = ("constrained_potential", "constraint_loss")
BLOCK_SIZE = 5.0
N_PERM = 9_999
N_BOOT = 4_999
N_BOOT_SHAPLEY = 999
SEED = 42
EPS = 1e-12


def read_csv(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def area_weight(lat) -> np.ndarray:
    out = np.cos(np.deg2rad(np.asarray(lat, float)))
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


def weighted_mean(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def weighted_sd(x, w) -> float:
    mean = weighted_mean(x, w)
    return float(np.sqrt(weighted_mean((np.asarray(x, float) - mean) ** 2, w))) if np.isfinite(mean) else np.nan


def standardize(x, w) -> np.ndarray:
    mean, sd = weighted_mean(x, w), weighted_sd(x, w)
    return (np.asarray(x, float) - mean) / sd if np.isfinite(sd) and sd > 0 else np.full(len(x), np.nan)


def weighted_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(x).all(axis=1)
    if ok.sum() <= x.shape[1] + 1:
        return np.nan, np.full(len(y), np.nan), np.full(x.shape[1] + 1, np.nan)
    xx, yy, ww = x[ok], y[ok], w[ok]
    matrix = np.column_stack([np.ones(len(xx)), xx])
    root = np.sqrt(ww)
    beta = np.linalg.lstsq(matrix * root[:, None], yy * root, rcond=None)[0]
    pred = matrix @ beta
    mean = np.sum(yy * ww) / np.sum(ww)
    ss_tot = np.sum(ww * (yy - mean) ** 2)
    ss_res = np.sum(ww * (yy - pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    full_pred = np.full(len(y), np.nan)
    full_pred[ok] = pred
    return float(r2), full_pred, beta


def shapley_r2(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    p = x.shape[1]
    cache = {(): 0.0}
    for size in range(1, p + 1):
        for subset in combinations(range(p), size):
            cache[subset] = weighted_fit(x[:, subset], y, w)[0]
    values = np.zeros(p, float)
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for size in range(p):
            coefficient = factorial(size) * factorial(p - size - 1) / factorial(p)
            for subset in combinations(others, size):
                with_j = tuple(sorted((*subset, j)))
                values[j] += coefficient * (cache[with_j] - cache[tuple(sorted(subset))])
    return cache[tuple(range(p))], values


def weighted_corr(x, y, w) -> float:
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    mx, my = np.sum(x * w) / np.sum(w), np.sum(y * w) / np.sum(w)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def residualize(target, controls, w) -> np.ndarray:
    _, pred, _ = weighted_fit(np.asarray(controls, float), np.asarray(target, float), np.asarray(w, float))
    return np.asarray(target, float) - pred


def partial_corr(x, y, controls, w) -> float:
    return weighted_corr(residualize(x, controls, w), residualize(y, controls, w), w)


def block_ids(lon, lat) -> np.ndarray:
    bx = np.floor((np.asarray(lon, float) + 180) / BLOCK_SIZE).astype(int)
    by = np.floor((np.asarray(lat, float) + 90) / BLOCK_SIZE).astype(int)
    return by * 10_000 + bx


def block_groups(block) -> list[np.ndarray]:
    return [np.flatnonzero(block == key) for key in np.unique(block)]


def partial_permutation_p(x, y, controls, w, block, observed, rng) -> float:
    rx = residualize(x, controls, w)
    ry = residualize(y, controls, w)
    groups = block_groups(block)
    if len(groups) < 2 or not np.isfinite(observed):
        return np.nan
    extreme = 0
    for _ in range(N_PERM):
        signed = ry.copy()
        for sign, rows in zip(rng.choice((-1.0, 1.0), size=len(groups)), groups):
            signed[rows] *= sign
        stat = weighted_corr(rx, signed, w)
        extreme += int(np.isfinite(stat) and abs(stat) >= abs(observed))
    return (extreme + 1) / (N_PERM + 1)


def bootstrap_partial(x, y, controls, w, block, rng) -> tuple[float, float]:
    groups = block_groups(block)
    stats = np.empty(N_BOOT, float)
    for i in range(N_BOOT):
        rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        stats[i] = partial_corr(x[rows], y[rows], controls[rows], w[rows])
    finite = stats[np.isfinite(stats)]
    return tuple(np.quantile(finite, [0.025, 0.975])) if len(finite) else (np.nan, np.nan)


def bh_adjust(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def prepare_data() -> pd.DataFrame:
    drivers = read_csv(DRIVER_CSV)
    drivers["lon_key"], drivers["lat_key"] = drivers["lon"].round(6), drivers["lat"].round(6)
    base = read_csv(PIXEL_DIR / "recalculated_baseline.csv")
    base["lon_key"], base["lat_key"] = base["lon"].round(6), base["lat"].round(6)
    out = drivers.merge(base[["lon_key", "lat_key", "area_weight", *RESPONSES]], on=["lon_key", "lat_key"], how="inner", validate="one_to_one")
    for response in RESPONSES:
        out.rename(columns={response: f"{response}_baseline"}, inplace=True)
    for scenario in FUTURES:
        future = read_csv(PIXEL_DIR / f"recalculated_{scenario}.csv")
        future["lon_key"], future["lat_key"] = future["lon"].round(6), future["lat"].round(6)
        keep = future[["lon_key", "lat_key", *RESPONSES]].rename(columns={r: f"{r}_{scenario}" for r in RESPONSES})
        out = out.merge(keep, on=["lon_key", "lat_key"], how="inner", validate="one_to_one")
        for response in RESPONSES:
            out[f"delta_{response}_{scenario}"] = out[f"{response}_{scenario}"] - out[f"{response}_baseline"]
    out["block_id"] = block_ids(out["lon"], out["lat"])
    return out


def environment_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    w = data["area_weight"].to_numpy(float)
    for scenario in FUTURES:
        for driver in DRIVERS:
            baseline = data[f"{driver}_baseline"].to_numpy(float)
            delta = data[f"delta_{driver}_{scenario}"].to_numpy(float)
            sd = weighted_sd(baseline, w)
            rows.append({"scenario": scenario, "driver": driver, "mean_change": weighted_mean(delta, w), "baseline_spatial_sd": sd, "standardized_mean_change": weighted_mean(delta, w) / sd if sd > 0 else np.nan})
    return pd.DataFrame(rows)


def analyze(data: pd.DataFrame):
    attribution_rows, partial_rows, rank_rows = [], [], []
    rng = np.random.default_rng(SEED)
    for scenario in FUTURES:
        for response in RESPONSES:
            columns = [f"delta_{d}_{scenario}" for d in DRIVERS]
            needed = [*columns, f"delta_{response}_{scenario}", "area_weight", "block_id"]
            valid = np.isfinite(data[needed].apply(pd.to_numeric, errors="coerce")).all(axis=1)
            subset = data.loc[valid].copy()
            w = subset["area_weight"].to_numpy(float)
            x = np.column_stack([standardize(subset[c], w) for c in columns])
            y = subset[f"delta_{response}_{scenario}"].to_numpy(float)
            block = subset["block_id"].to_numpy(int)

            full_r2, contributions = shapley_r2(x, y, w)
            denominator = contributions.sum()
            for driver, value in zip(DRIVERS, contributions):
                attribution_rows.append({"scenario": scenario, "response": response, "driver": driver, "model_r2": full_r2, "shapley_r2": value, "share_of_explained_variance_percent": 100 * value / denominator if abs(denominator) > EPS else np.nan})

            local_partial = []
            for j, driver in enumerate(DRIVERS):
                controls = np.delete(x, j, axis=1)
                observed = partial_corr(x[:, j], y, controls, w)
                p = partial_permutation_p(x[:, j], y, controls, w, block, observed, rng)
                lo, hi = bootstrap_partial(x[:, j], y, controls, w, block, rng)
                local_partial.append({"scenario": scenario, "response": response, "driver": driver, "partial_r": observed, "p": p, "ci_low": lo, "ci_high": hi})
            adjusted = bh_adjust([row["p"] for row in local_partial])
            for row, q in zip(local_partial, adjusted):
                row["p_fdr"] = q
                partial_rows.append(row)

            groups = block_groups(block)
            boot = np.empty((N_BOOT_SHAPLEY, len(DRIVERS)), float)
            for i in range(N_BOOT_SHAPLEY):
                rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                _, values = shapley_r2(x[rows], y[rows], w[rows])
                denom = values.sum()
                boot[i] = values / denom if abs(denom) > EPS else np.nan
            ranks = np.argsort(-boot, axis=1)
            for j, driver in enumerate(DRIVERS):
                finite = boot[:, j][np.isfinite(boot[:, j])]
                rank_rows.append({"scenario": scenario, "response": response, "driver": driver, "mean_share": np.mean(finite) if len(finite) else np.nan, "ci_low": np.quantile(finite, 0.025) if len(finite) else np.nan, "ci_high": np.quantile(finite, 0.975) if len(finite) else np.nan, "top_two_probability": np.mean(np.any(ranks[:, :2] == j, axis=1)) if len(finite) else np.nan})
    return pd.DataFrame(attribution_rows), pd.DataFrame(partial_rows), pd.DataFrame(rank_rows)


def main() -> None:
    ATTR_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    environment_summary(data).to_csv(ATTR_DIR / "environmental_change_summary.csv", index=False)
    attribution, partial, stability = analyze(data)
    attribution.to_csv(ATTR_DIR / "shapley_attribution.csv", index=False)
    partial.to_csv(ATTR_DIR / "partial_correlations.csv", index=False)
    stability.to_csv(ATTR_DIR / "shapley_block_bootstrap.csv", index=False)
    print("Saved environmental attribution outputs to", ATTR_DIR)


if __name__ == "__main__":
    main()
