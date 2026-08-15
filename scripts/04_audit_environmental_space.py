"""Audit projected environmental representation using PCA pairwise convex hulls."""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from macroalgae_repro.paths import ProjectPaths, SCENARIOS
from macroalgae_repro.spatial import area_weight, clean_values, open_2d
from macroalgae_repro.statistics import weighted_fraction, weighted_mean


PATHS = ProjectPaths.from_env()
OUTPUT_DIR = PATHS.output_dir
MODEL_DIR = OUTPUT_DIR / "model_validation_no3_po4_5cv_seed42"
TRAINING_CSV = MODEL_DIR / "long_training_after_outlier.csv"
OUT_DIR = OUTPUT_DIR / "environmental_space_coverage"

ENDPOINTS = ("NO3", "PO4")
PREDICTORS = ("Temperature", "pH", "Salinity", "Light intensity", "initial_concentration")
PCA_VARIANCE = 0.90
DEPTH_LIMIT_M = 50.0
PHOTOPERIOD_H = 12.0
CHUNK = 200_000


def projected_environment(scenario: str, template, light, depth) -> pd.DataFrame:
    no3 = clean_values(
        open_2d(
            PATHS.environment(scenario, "no3"),
            ("no3", "nitrate"),
            template=template,
        ).values,
        0,
    )
    po4 = clean_values(
        open_2d(
            PATHS.environment(scenario, "po4"),
            ("po4", "phosphate"),
            template=template,
        ).values,
        0,
    )
    ph = clean_values(
        open_2d(PATHS.environment(scenario, "ph"), ("ph",), template=template).values,
        6.0,
        9.5,
    )
    salinity = clean_values(
        open_2d(
            PATHS.environment(scenario, "salinity"),
            ("so", "salinity"),
            template=template,
        ).values,
        0,
        45,
    )
    temperature = clean_values(
        open_2d(
            PATHS.environment(scenario, "temperature"),
            ("thetao", "temperature", "temp"),
            template=template,
        ).values,
        -3,
        45,
    )
    light_values = clean_values(light.values, 0)
    valid = (
        np.isfinite(no3) & np.isfinite(po4) & np.isfinite(ph)
        & np.isfinite(salinity) & np.isfinite(temperature) & np.isfinite(light_values)
        & np.isfinite(depth) & (depth > 0) & (depth <= DEPTH_LIMIT_M)
    )
    lat2, lon2 = np.meshgrid(template["lat"].values, template["lon"].values, indexing="ij")
    return pd.DataFrame(
        {
            "lon": lon2[valid], "lat": lat2[valid],
            "Temperature": temperature[valid], "pH": ph[valid],
            "Salinity": salinity[valid], "Light intensity": light_values[valid],
            "NO3": no3[valid] * 0.0140067, "PO4": po4[valid] * 0.0309738,
        }
    )


def common_environment():
    template = open_2d(PATHS.environment("baseline", "no3"), ("no3", "nitrate"))
    par = open_2d(PATHS.par_baseline, ("par", "photosynt", "radiation"), template=template)
    light = (par * (1_000_000 / (PHOTOPERIOD_H * 3600))).astype("float32")
    terrain = open_2d(PATHS.terrain, ("bathymetry", "bathy", "depth", "terrain"), template=template)
    terrain_values = terrain.values.astype("float32")
    depth = (
        -terrain_values if np.nanmedian(terrain_values) < 0 else terrain_values
    ).astype("float32")

    frames, indexes = {}, {}
    for scenario in SCENARIOS:
        frame = projected_environment(scenario, template, light, depth)
        frame["_lon"], frame["_lat"] = frame["lon"].round(6), frame["lat"].round(6)
        frames[scenario] = frame
        indexes[scenario] = pd.MultiIndex.from_frame(frame[["_lon", "_lat"]])
    common = indexes["baseline"]
    for scenario in SCENARIOS[1:]:
        common = common.intersection(indexes[scenario], sort=False)
    common = pd.MultiIndex.from_frame(common.to_frame(index=False).sort_values(["_lon", "_lat"]))
    return {
        scenario: (
            frames[scenario]
            .set_index(["_lon", "_lat"])
            .reindex(common)
            .reset_index()
        )
        for scenario in SCENARIOS
    }


def retained_components(pca: PCA) -> int:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, PCA_VARIANCE) + 1)


def hull_equations(points: np.ndarray):
    equations = []
    for first, second in combinations(range(points.shape[1]), 2):
        pair = points[:, [first, second]]
        pair = pair[np.isfinite(pair).all(axis=1)]
        if len(pair) < 3:
            continue
        try:
            equations.append(((first, second), ConvexHull(pair).equations.copy()))
        except QhullError:
            continue
    if not equations:
        raise RuntimeError("No valid pairwise PCA convex hull could be constructed")
    return equations


def coverage(scores: np.ndarray, equations) -> np.ndarray:
    count = np.zeros(len(scores), dtype="int16")
    valid_pairs = np.zeros(len(scores), dtype="int16")
    for (first, second), eq in equations:
        pair = scores[:, [first, second]]
        valid = np.isfinite(pair).all(axis=1)
        valid_pairs[valid] += 1
        index = np.flatnonzero(valid)
        for start in range(0, len(index), CHUNK):
            rows = index[start : start + CHUNK]
            inside = np.all(pair[rows] @ eq[:, :2].T + eq[:, 2] <= 1e-10, axis=1)
            count[rows] += inside.astype("int16")
    out = np.full(len(scores), np.nan, dtype="float32")
    valid = valid_pairs > 0
    out[valid] = count[valid] / valid_pairs[valid]
    return out


def endpoint_audit(endpoint: str, training: pd.DataFrame, environments):
    subset = training[training["nutrient_type"].eq(endpoint)].copy()
    train = subset[list(PREDICTORS)].apply(pd.to_numeric, errors="coerce")
    train = train[np.isfinite(train).all(axis=1)]
    scaler = StandardScaler().fit(train)
    standardized = scaler.transform(train)
    pca_full = PCA().fit(standardized)
    n_components = retained_components(pca_full)
    pca = PCA(n_components=n_components).fit(standardized)
    train_scores = pca.transform(standardized)
    equations = hull_equations(train_scores)

    rows, pixel_rows = [], []
    for scenario in SCENARIOS:
        env = environments[scenario]
        matrix = env[["Temperature", "pH", "Salinity", "Light intensity"]].copy()
        matrix["initial_concentration"] = env[endpoint]
        scores = pca.transform(scaler.transform(matrix))
        cov = coverage(scores, equations)
        weight = area_weight(env["lat"])
        valid = np.isfinite(cov) & np.isfinite(weight)
        rows.append(
            {
                "endpoint": endpoint, "scenario": scenario,
                "n_training": len(train), "n_components": n_components,
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
                "n_pairwise_hulls": len(equations),
                "mean_coverage": weighted_mean(cov, weight, valid),
                "area_fraction_ge_0.90": weighted_fraction(cov >= 0.90, valid, weight),
                "area_fraction_lt_0.50": weighted_fraction(cov < 0.50, valid, weight),
            }
        )
        pixel_rows.append(
            pd.DataFrame(
                {
                    "endpoint": endpoint,
                    "scenario": scenario,
                    "lon": env["lon"],
                    "lat": env["lat"],
                    "coverage": cov,
                }
            )
        )
    return rows, pixel_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(TRAINING_CSV, encoding="utf-8-sig")
    environments = common_environment()
    summaries, pixels = [], []
    for endpoint in ENDPOINTS:
        rows, pixel_rows = endpoint_audit(endpoint, training, environments)
        summaries.extend(rows)
        pixels.extend(pixel_rows)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "environmental_space_summary.csv", index=False)
    pd.concat(pixels, ignore_index=True).to_csv(
        OUT_DIR / "environmental_space_pixel_coverage.csv.gz",
        index=False,
        compression="gzip",
    )
    info = {
        "method": (
            "training-standardized PCA plus pairwise retained-PC "
            "convex-hull coverage"
        ),
        "pca_cumulative_variance_threshold": PCA_VARIANCE,
        "coverage_use": "audit only; predictions are not filtered",
    }
    (OUT_DIR / "environmental_space_run_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
