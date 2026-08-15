"""Project species-level NO3 and PO4 removal rates to the coastal grid.

The script consumes the final model bundle produced by ``01_fit_nutrient_model.py``.
It deliberately stops at species-level prediction and environmental bookkeeping;
all pressure, unconstrained/constrained potential and mismatch calculations are
performed authoritatively in ``03_build_pressure_potential_mismatch.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths, SCENARIOS
from macroalgae_repro.spatial import clean_values, open_2d


PATHS = ProjectPaths.from_env()
OUTPUT_DIR = PATHS.output_dir
RUN_ID = "no3_po4_5cv_seed42_log_duan_iqr3"
MODEL_DIR = OUTPUT_DIR / "model_validation_no3_po4_5cv_seed42"
TRAINING_CSV = MODEL_DIR / "long_training_after_outlier.csv"
MODEL_BUNDLE = OUTPUT_DIR / "cache" / RUN_ID / "model_bundle.joblib"
CACHE_DIR = OUTPUT_DIR / "cache" / RUN_ID / "predictions"
SPECIES_CSV = OUTPUT_DIR / "species_lookup.csv"

TARGET_RESOLUTION_DEG = 0.25
DEPTH_LIMIT_M = 50.0
PHOTOPERIOD_H = 12.0
DENSITY_G_L = 2.5
EXPERIMENTAL_DURATION_H = 24.0
MIN_VALID_RECORDS = 15
MEOW_OCCURRENCE_MIN = 3
PREDICT_BATCH_SIZE = 10_000
REDFIELD_N_TO_P = 16.0

TAXONOMY_COLS = ["Phylum", "Order", "Family", "Genus", "Scientific name"]
NUMERIC_FEATURES = [
    "pH",
    "Temperature",
    "Salinity",
    "Light intensity",
    "Photoperiod",
    "Density",
    "Experimental duration",
    "initial_concentration",
]
CATEGORICAL_FEATURES = TAXONOMY_COLS + ["Data category", "nutrient_type"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_key(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split()).strip().lower()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def build_name_bridge() -> dict[str, tuple[str, str]]:
    """Map model species names to accepted names used by the occurrence summary."""
    if not PATHS.algaetraits_attributes.exists():
        return {}
    traits = read_csv(PATHS.algaetraits_attributes)
    if "accepted_name" not in traits:
        return {}
    name_columns = [
        c
        for c in (
            "original_scientific_name",
            "scientific_name_clean",
            "match_query_name",
            "accepted_name",
        )
        if c in traits
    ]
    bridge: dict[str, tuple[str, str]] = {}
    for _, row in traits.iterrows():
        accepted = row.get("accepted_name")
        accepted_key = normalize_key(accepted)
        if not accepted_key:
            continue
        for column in name_columns:
            key = normalize_key(row.get(column))
            if key and key not in bridge:
                bridge[key] = (str(accepted), column)
    return bridge


def build_species_table(training: pd.DataFrame) -> pd.DataFrame:
    counts = training.dropna(subset=["Scientific name"]).groupby("Scientific name").size()
    keep = counts[counts > MIN_VALID_RECORDS].index
    if keep.empty:
        raise RuntimeError("No candidate species have more than 15 valid modelling records.")

    species = (
        training[TAXONOMY_COLS]
        .dropna(subset=["Scientific name"])
        .drop_duplicates("Scientific name")
    )
    species = species[species["Scientific name"].isin(keep)].copy()
    for column in TAXONOMY_COLS:
        species[column] = species[column].astype("string").fillna("Missing")
    species["valid_record_count"] = species["Scientific name"].map(counts)
    species = species.sort_values("valid_record_count", ascending=False).reset_index(drop=True)

    bridge = build_name_bridge()
    accepted_names, sources = [], []
    for name in species["Scientific name"]:
        mapped = bridge.get(normalize_key(name))
        if mapped is None:
            accepted_names.append(str(name))
            sources.append("original_name_fallback")
        else:
            accepted_names.append(mapped[0])
            sources.append(mapped[1])
    species["meow_match_name"] = accepted_names
    species["meow_match_key"] = [normalize_key(x) for x in accepted_names]
    species["name_bridge_source"] = sources
    species["species_id"] = np.arange(len(species), dtype=int)
    species.to_csv(SPECIES_CSV, index=False, encoding="utf-8-sig")
    return species


def find_meow_shapefile() -> Path:
    candidates = list(PATHS.meow_dir.rglob("*.shp"))
    if not candidates:
        raise FileNotFoundError(f"No MEOW shapefile found under {PATHS.meow_dir}")

    def score(path: Path) -> int:
        name = str(path).lower()
        return (
            (100 if "meow" in name else 0)
            + (50 if "prov" in name or "province" in name else 0)
            - (60 if "ppow" in name else 0)
        )

    return sorted(candidates, key=score, reverse=True)[0]


def infer_province_column(frame: gpd.GeoDataFrame) -> str:
    columns = [c for c in frame.columns if c.lower() != "geometry"]
    exact = ("province", "provinc", "prov_name", "province_name", "meow_province", "meow_prov")
    for target in exact:
        for column in columns:
            if column.lower() == target:
                return column
    for column in columns:
        low = column.lower()
        if ("province" in low or "prov" in low) and not any(x in low for x in ("id", "code", "num")):
            return column
    raise ValueError(f"Cannot identify MEOW province column: {columns}")


def map_meow_provinces(template, depth: np.ndarray) -> np.ndarray:
    meow = gpd.read_file(find_meow_shapefile())
    meow = meow.set_crs("EPSG:4326") if meow.crs is None else meow.to_crs("EPSG:4326")
    province_col = infer_province_column(meow)
    meow = meow[meow.geometry.notna()][[province_col, "geometry"]].copy()

    lat2, lon2 = np.meshgrid(template["lat"].values, template["lon"].values, indexing="ij")
    coastal = np.isfinite(depth) & (depth > 0) & (depth <= DEPTH_LIMIT_M)
    flat_index = np.flatnonzero(coastal.reshape(-1))
    points = pd.DataFrame(
        {
            "flat_index": flat_index,
            "lon": lon2.reshape(-1)[flat_index],
            "lat": lat2.reshape(-1)[flat_index],
        }
    )
    points = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, meow, how="left", predicate="within").drop_duplicates("flat_index")
    out = np.full(depth.size, None, dtype=object)
    valid = joined[province_col].notna()
    out[joined.loc[valid, "flat_index"].astype(int)] = joined.loc[valid, province_col].astype(str)
    return out.reshape(depth.shape)


def load_allowed_species(species: pd.DataFrame) -> dict[str, set[str]]:
    occurrence = read_csv(PATHS.meow_occurrence_summary)
    required = {"accepted_name", "meow_province", "occurrence_count"}
    missing = required.difference(occurrence.columns)
    if missing:
        raise ValueError(f"Occurrence summary missing columns: {sorted(missing)}")
    occurrence["occurrence_count"] = pd.to_numeric(occurrence["occurrence_count"], errors="coerce").fillna(0)
    occurrence = occurrence[occurrence["occurrence_count"] >= MEOW_OCCURRENCE_MIN].copy()
    occurrence["species_key"] = occurrence["accepted_name"].map(normalize_key)
    occurrence["province_key"] = occurrence["meow_province"].map(normalize_key)
    allowed = occurrence.groupby("species_key")["province_key"].apply(set).to_dict()

    status = species[["species_id", "Scientific name", "meow_match_name", "valid_record_count"]].copy()
    status["n_allowed_provinces"] = [len(allowed.get(key, set())) for key in species["meow_match_key"]]
    status.to_csv(OUTPUT_DIR / "candidate_species_meow_status.csv", index=False, encoding="utf-8-sig")
    return allowed


def redfield_weights(no3: np.ndarray, po4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_equivalent = np.asarray(no3, dtype="float32") / REDFIELD_N_TO_P
    p = np.asarray(po4, dtype="float32")
    denominator = n_equivalent + p
    w_n = np.full(denominator.shape, 0.5, dtype="float32")
    valid = np.isfinite(denominator) & (denominator > 0)
    w_n[valid] = n_equivalent[valid] / denominator[valid]
    return w_n, (1.0 - w_n).astype("float32")


def predict_batches(model, matrix: np.ndarray) -> np.ndarray:
    output = []
    for start in range(0, len(matrix), PREDICT_BATCH_SIZE):
        batch = matrix[start : start + PREDICT_BATCH_SIZE]
        try:
            pred = model.predict(batch, output_type="mean")
        except TypeError:
            pred = model.predict(batch)
        output.append(np.asarray(pred).reshape(-1).astype("float32"))
    return np.concatenate(output) if output else np.empty(0, dtype="float32")


def inverse_log1p(pred_log: np.ndarray, smear: float) -> np.ndarray:
    pred = np.exp(np.asarray(pred_log, dtype="float64")) * float(smear) - 1.0
    return np.clip(pred, 0, None).astype("float32")


def feature_frame(base: pd.DataFrame, species: pd.Series, nutrient: str, concentration) -> pd.DataFrame:
    frame = base.copy()
    for column in TAXONOMY_COLS:
        frame[column] = species[column]
    frame["nutrient_type"] = nutrient
    frame["initial_concentration"] = concentration
    return frame[FEATURE_COLS]


def scenario_environment(scenario: str, template, light, depth, meow_grid):
    no3 = open_2d(PATHS.environment(scenario, "no3"), ("no3", "nitrate"), template=template)
    po4 = open_2d(PATHS.environment(scenario, "po4"), ("po4", "phosphate"), template=template)
    ph = open_2d(PATHS.environment(scenario, "ph"), ("ph",), template=template)
    salinity = open_2d(PATHS.environment(scenario, "salinity"), ("so", "salinity"), template=template)
    temperature = open_2d(PATHS.environment(scenario, "temperature"), ("thetao", "temperature", "temp"), template=template)

    no3_raw = clean_values(no3.values, 0)
    po4_raw = clean_values(po4.values, 0)
    ph_raw = clean_values(ph.values, 6.0, 9.5)
    salinity_raw = clean_values(salinity.values, 0.0, 45.0)
    temperature_raw = clean_values(temperature.values, -3.0, 45.0)
    light_raw = clean_values(light.values, 0)

    valid = (
        np.isfinite(no3_raw)
        & np.isfinite(po4_raw)
        & np.isfinite(ph_raw)
        & np.isfinite(salinity_raw)
        & np.isfinite(temperature_raw)
        & np.isfinite(light_raw)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth <= DEPTH_LIMIT_M)
    )
    if not valid.any():
        raise RuntimeError(f"No valid coastal pixels for {scenario}")

    lat2, lon2 = np.meshgrid(template["lat"].values, template["lon"].values, indexing="ij")
    no3_valid = no3_raw[valid]
    po4_valid = po4_raw[valid]
    w_no3, w_po4 = redfield_weights(no3_valid, po4_valid)
    pixels = pd.DataFrame(
        {
            "scenario": scenario,
            "lon": lon2[valid],
            "lat": lat2[valid],
            "NO3_mmol_m3": no3_valid,
            "PO4_mmol_m3": po4_valid,
            "NO3_N_mg_L": no3_valid * 0.0140067,
            "PO4_P_mg_L": po4_valid * 0.0309738,
            "pH": ph_raw[valid],
            "Temperature": temperature_raw[valid],
            "Salinity": salinity_raw[valid],
            "Light_intensity": light_raw[valid],
            "depth_m": depth[valid],
            "meow_province": ["" if x is None else str(x) for x in meow_grid[valid]],
            "w_no3": w_no3,
            "w_po4": w_po4,
        }
    )
    base = pd.DataFrame(
        {
            "pH": ph_raw[valid],
            "Temperature": temperature_raw[valid],
            "Salinity": salinity_raw[valid],
            "Light intensity": light_raw[valid],
            "Photoperiod": PHOTOPERIOD_H,
            "Density": DENSITY_G_L,
            "Experimental duration": EXPERIMENTAL_DURATION_H,
        }
    )
    return pixels, base


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training = read_csv(TRAINING_CSV)
    species = build_species_table(training)
    allowed = load_allowed_species(species)

    if not MODEL_BUNDLE.exists():
        raise FileNotFoundError(MODEL_BUNDLE)
    bundle = joblib.load(MODEL_BUNDLE)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    smear = float(bundle["smear"])
    data_category = training["Data category"].astype("string").fillna("Missing").mode().iloc[0]

    template = open_2d(PATHS.environment("baseline", "no3"), ("no3", "nitrate"))
    par = open_2d(PATHS.par_baseline, ("par", "photosynt", "radiation"), template=template)
    light = (par * (1_000_000 / (PHOTOPERIOD_H * 3600))).astype("float32")
    terrain = open_2d(PATHS.terrain, ("bathymetry", "bathy", "depth", "terrain"), template=template)
    terrain_values = terrain.values.astype("float32")
    depth = (-terrain_values if np.nanmedian(terrain_values) < 0 else terrain_values).astype("float32")
    meow_grid = map_meow_provinces(template, depth)

    for scenario in SCENARIOS:
        print(f"Projecting {scenario}...")
        pixels, base = scenario_environment(scenario, template, light, depth, meow_grid)
        province_keys = np.array([normalize_key(x) for x in pixels["meow_province"]], dtype=object)
        candidate_count = np.zeros(len(pixels), dtype="int16")
        for key in species["meow_match_key"]:
            candidate_count += np.isin(province_keys, tuple(allowed.get(key, ()))).astype("int16")
        pixels["candidate_species_count"] = candidate_count

        for nutrient, concentration_column in (("NO3", "NO3_N_mg_L"), ("PO4", "PO4_P_mg_L")):
            nutrient_dir = CACHE_DIR / scenario / nutrient
            nutrient_dir.mkdir(parents=True, exist_ok=True)
            concentration = pixels[concentration_column].to_numpy("float32")
            for i, row in species.iterrows():
                path = nutrient_dir / f"species_{int(row['species_id']):05d}.npy"
                if path.exists():
                    cached = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
                    if cached.size == len(pixels):
                        continue
                features = feature_frame(base.assign(**{"Data category": data_category}), row, nutrient, concentration)
                transformed = np.asarray(preprocessor.transform(features), dtype="float32")
                prediction = inverse_log1p(predict_batches(model, transformed), smear)
                np.save(path, prediction, allow_pickle=False)
                print(f"  {nutrient}: {i + 1}/{len(species)} {row['Scientific name']}")

        pixel_path = OUTPUT_DIR / f"potential_pixels_{scenario}_depth50m_025deg.csv"
        pixels.to_csv(pixel_path, index=False, encoding="utf-8-sig")
        print(f"Saved {pixel_path}")


if __name__ == "__main__":
    main()
