"""Evaluate ecological heterogeneity in biogeographic constraint.

The analysis covers MEOW realms, Redfield-relative nutrient regimes and regional
candidate-species richness. Nutrient-regime labels describe relative baseline
stoichiometry and are not direct tests of nutrient limitation.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import geopandas as gpd
import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.statistics import weighted_quantile


PATHS = ProjectPaths.from_env()
OUTPUT_DIR = PATHS.output_dir
PIXEL_DIR = OUTPUT_DIR / "cross_scenario_comparison_and_mismatch_final" / "recalculated_pixels"
ATTR_DIR = OUTPUT_DIR / "future_environmental_attribution"
DRIVER_CSV = ATTR_DIR / "driver_delta_matrix.csv"
OUT_DIR = OUTPUT_DIR / "ecological_heterogeneity"
BASELINE_PROJECTION = OUTPUT_DIR / "potential_pixels_baseline_depth50m_025deg.csv"

FUTURES = ("ssp245", "ssp370", "ssp585")
DRIVERS = ("Temperature", "NO3", "PO4", "pH", "Salinity")
BLOCK_SIZE = 5.0
N_BOOT_CORR = 999
N_BOOT_SHAPLEY = 999
MIN_GROUP_PIXELS = 300
MIN_GROUP_BLOCKS = 10
REDFIELD_RATIO = 16.0
SEED = 42
EPS = 1e-12


def read_csv(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def wmean(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def wcorr(x, y, w) -> float:
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 10:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    mx, my = np.sum(x * w) / np.sum(w), np.sum(y * w) / np.sum(w)
    xv, yv = x - mx, y - my
    denominator = np.sqrt(np.sum(w * xv**2) * np.sum(w * yv**2))
    return float(np.sum(w * xv * yv) / denominator) if denominator > EPS else np.nan


def block_id(lon, lat) -> np.ndarray:
    lon = ((np.asarray(lon, float) + 180) % 360) - 180
    lat = np.asarray(lat, float)
    bx = np.floor((lon + 180) / BLOCK_SIZE).astype(int)
    by = np.floor((lat + 90) / BLOCK_SIZE).astype(int)
    return by * 10_000 + bx


def bootstrap_corr(x, y, lon, lat, w, rng) -> tuple[float, float, float]:
    observed = wcorr(x, y, w)
    blocks = block_id(lon, lat)
    groups = [np.flatnonzero(blocks == key) for key in np.unique(blocks)]
    if len(groups) < 2:
        return observed, np.nan, np.nan
    values = np.empty(N_BOOT_CORR, float)
    for i in range(N_BOOT_CORR):
        rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        values[i] = wcorr(np.asarray(x)[rows], np.asarray(y)[rows], np.asarray(w)[rows])
    finite = values[np.isfinite(values)]
    if not len(finite):
        return observed, np.nan, np.nan
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return observed, float(lo), float(hi)


def find_meow_shapefile() -> str:
    files = list(PATHS.meow_dir.rglob("*.shp"))
    if not files:
        raise FileNotFoundError(f"No MEOW shapefile under {PATHS.meow_dir}")
    ordered = sorted(
        files,
        key=lambda path: (
            "meow" in str(path).lower(),
            "prov" in str(path).lower(),
        ),
        reverse=True,
    )
    return str(ordered[0])


def infer_realm_column(frame: gpd.GeoDataFrame) -> str:
    for column in frame.columns:
        low = column.lower()
        if low in {"realm", "real_name", "realm_name"} or ("realm" in low and "id" not in low):
            return column
    raise ValueError(f"Cannot identify MEOW realm column: {list(frame.columns)}")


def add_realm(frame: pd.DataFrame) -> pd.DataFrame:
    meow = gpd.read_file(find_meow_shapefile())
    meow = meow.set_crs("EPSG:4326") if meow.crs is None else meow.to_crs("EPSG:4326")
    realm_column = infer_realm_column(meow)
    points = gpd.GeoDataFrame(
        frame[["row_id", "lon", "lat"]].copy(),
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]), crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, meow[[realm_column, "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates("row_id")
    lookup = joined.set_index("row_id")[realm_column]
    frame["meow_realm"] = frame["row_id"].map(lookup).fillna("Unclassified")
    return frame


def load_pixel_context() -> pd.DataFrame:
    baseline = read_csv(PIXEL_DIR / "recalculated_baseline.csv")
    baseline["lon_key"], baseline["lat_key"] = baseline["lon"].round(6), baseline["lat"].round(6)
    projection = read_csv(BASELINE_PROJECTION)
    projection["lon_key"] = projection["lon"].round(6)
    projection["lat_key"] = projection["lat"].round(6)
    projection = projection[["lon_key", "lat_key", "candidate_species_count", "meow_province"]]
    frame = baseline.merge(projection, on=["lon_key", "lat_key"], how="left", validate="one_to_one")
    frame["row_id"] = np.arange(len(frame))

    no3 = pd.to_numeric(frame["NO3_mmol_m3"], errors="coerce").to_numpy(float)
    po4 = pd.to_numeric(frame["PO4_mmol_m3"], errors="coerce").to_numpy(float)
    ratio = np.full(len(frame), np.nan)
    valid = np.isfinite(no3) & np.isfinite(po4) & (po4 > 0)
    ratio[valid] = no3[valid] / po4[valid]
    frame["baseline_N_to_P_molar"] = ratio
    frame["nutrient_regime"] = "Unclassified"
    frame.loc[
        valid & (ratio < REDFIELD_RATIO), "nutrient_regime"
    ] = "N-depleted relative to Redfield"
    frame.loc[
        valid & (ratio >= REDFIELD_RATIO), "nutrient_regime"
    ] = "P-depleted relative to Redfield"

    richness = pd.to_numeric(frame["candidate_species_count"], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(frame["area_weight"], errors="coerce").to_numpy(float)
    q33 = weighted_quantile(richness, 1 / 3, weights)
    q66 = weighted_quantile(richness, 2 / 3, weights)
    frame["species_pool_group"] = "Unclassified"
    frame.loc[np.isfinite(richness) & (richness <= q33), "species_pool_group"] = "Low species pool"
    frame.loc[
        np.isfinite(richness) & (richness > q33) & (richness <= q66),
        "species_pool_group",
    ] = "Medium species pool"
    frame.loc[np.isfinite(richness) & (richness > q66), "species_pool_group"] = "High species pool"
    frame = add_realm(frame)

    drivers = read_csv(DRIVER_CSV)
    drivers["lon_key"], drivers["lat_key"] = drivers["lon"].round(6), drivers["lat"].round(6)
    delta_columns = [f"delta_{driver}_{scenario}" for scenario in FUTURES for driver in DRIVERS]
    frame = frame.merge(
        drivers[["lon_key", "lat_key", *delta_columns]],
        on=["lon_key", "lat_key"],
        how="left",
        validate="one_to_one",
    )
    for scenario in FUTURES:
        future = read_csv(PIXEL_DIR / f"recalculated_{scenario}.csv")
        future["lon_key"], future["lat_key"] = future["lon"].round(6), future["lat"].round(6)
        future = future[["lon_key", "lat_key", "constraint_loss"]].rename(
            columns={"constraint_loss": f"constraint_loss_{scenario}"}
        )
        frame = frame.merge(future, on=["lon_key", "lat_key"], how="left", validate="one_to_one")
        frame[f"delta_constraint_loss_{scenario}"] = (
            frame[f"constraint_loss_{scenario}"] - frame["constraint_loss"]
        )
    return frame


def richness_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    richness = pd.to_numeric(frame["candidate_species_count"], errors="coerce").to_numpy(float)
    weights = frame["area_weight"].to_numpy(float)
    outcomes = {
        "constrained_potential": frame["constrained_potential"].to_numpy(float),
        "constraint_ratio": frame["constraint_ratio"].to_numpy(float),
        "constraint_loss": frame["constraint_loss"].to_numpy(float),
        "stable_mismatch": frame["stable_mismatch"].to_numpy(float),
    }
    rows = []
    for name, outcome in outcomes.items():
        r, lo, hi = bootstrap_corr(richness, outcome, frame["lon"], frame["lat"], weights, rng)
        rows.append({"outcome": name, "weighted_correlation": r, "ci_2.5": lo, "ci_97.5": hi})
    return pd.DataFrame(rows)


def group_summary(frame: pd.DataFrame, axis: str, column: str) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(column, observed=True):
        if group == "Unclassified" or len(subset) < MIN_GROUP_PIXELS:
            continue
        w = subset["area_weight"].to_numpy(float)
        rows.append(
            {
                "axis": axis, "group": group, "n_pixels": len(subset),
                "constraint_loss_mean": wmean(subset["constraint_loss"], w),
                "constraint_ratio_mean": wmean(subset["constraint_ratio"], w),
                "constrained_potential_mean": wmean(subset["constrained_potential"], w),
                "stable_mismatch_fraction": wmean(subset["stable_mismatch"], w),
            }
        )
    return pd.DataFrame(rows)


def weighted_fit_r2(x, y, w) -> float:
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(x).all(axis=1)
    if ok.sum() <= x.shape[1] + 1:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    design = np.column_stack([np.ones(len(x)), x])
    root = np.sqrt(w)
    beta = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    pred = design @ beta
    mean = np.sum(y * w) / np.sum(w)
    return float(1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - mean) ** 2))


def shapley(x, y, w) -> tuple[float, np.ndarray]:
    p = x.shape[1]
    cache = {(): 0.0}
    for size in range(1, p + 1):
        for subset in combinations(range(p), size):
            cache[subset] = weighted_fit_r2(x[:, subset], y, w)
    values = np.zeros(p)
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for size in range(p):
            coef = factorial(size) * factorial(p - size - 1) / factorial(p)
            for subset in combinations(others, size):
                values[j] += coef * (
                    cache[tuple(sorted((*subset, j)))]
                    - cache[tuple(sorted(subset))]
                )
    return cache[tuple(range(p))], values


def subgroup_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    axes = (
        ("Redfield-relative nutrient regime", "nutrient_regime"),
        ("Species-pool capacity", "species_pool_group"),
    )
    for axis, column in axes:
        for group, subset in frame.groupby(column, observed=True):
            if group == "Unclassified" or len(subset) < MIN_GROUP_PIXELS:
                continue
            blocks = np.unique(block_id(subset["lon"], subset["lat"]))
            if len(blocks) < MIN_GROUP_BLOCKS:
                continue
            for scenario in FUTURES:
                x = (
                    subset[[f"delta_{driver}_{scenario}" for driver in DRIVERS]]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(float)
                )
                y = pd.to_numeric(
                    subset[f"delta_constraint_loss_{scenario}"], errors="coerce"
                ).to_numpy(float)
                w = subset["area_weight"].to_numpy(float)
                ok = np.isfinite(x).all(axis=1) & np.isfinite(y) & np.isfinite(w) & (w > 0)
                r2, values = shapley(x[ok], y[ok], w[ok])
                total = values.sum()
                for driver, value in zip(DRIVERS, values):
                    rows.append(
                        {
                            "axis": axis,
                            "group": group,
                            "scenario": scenario,
                            "response": "constraint_loss",
                            "driver": driver,
                            "model_r2": r2,
                            "share_of_explained_variance_percent": (
                                100 * value / total if abs(total) > EPS else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_pixel_context()
    frame.to_csv(OUT_DIR / "ecological_pixel_context.csv.gz", index=False, compression="gzip")
    richness_associations(frame).to_csv(OUT_DIR / "richness_associations.csv", index=False)
    summary = pd.concat(
        [
            group_summary(frame, "MEOW realm", "meow_realm"),
            group_summary(
                frame,
                "Redfield-relative nutrient regime",
                "nutrient_regime",
            ),
            group_summary(frame, "Species-pool capacity", "species_pool_group"),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUT_DIR / "ecological_group_summary.csv", index=False)
    subgroup_attribution(frame).to_csv(OUT_DIR / "subgroup_driver_attribution.csv", index=False)
    print("Saved ecological heterogeneity outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
