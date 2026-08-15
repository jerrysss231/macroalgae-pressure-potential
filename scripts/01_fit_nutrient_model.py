"""Fit and validate the pooled NO3/PO4 TabPFN removal model."""

from __future__ import annotations

import json
import os
import random
import sys
from importlib.metadata import PackageNotFoundError, version

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNRegressor

from macroalgae_repro.paths import ProjectPaths

try:
    import torch
except ImportError:
    torch = None


PATHS = ProjectPaths.from_env()
OUTPUT_DIR = PATHS.output_dir
MODEL_DIR = OUTPUT_DIR / "model_validation_no3_po4_5cv_seed42"
RUN_ID = "no3_po4_5cv_seed42_log_duan_iqr3"
BUNDLE_PATH = OUTPUT_DIR / "cache" / RUN_ID / "model_bundle.joblib"

NUTRIENTS = ("NO3", "PO4")
N_SPLITS = 5
SEED = 42
IQR_MULTIPLIER = 3.0
PREDICT_BATCH_SIZE = 10_000
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
EXPECTED_TABPFN_VERSION = "8.0.0"

TAXONOMY = ["Phylum", "Order", "Family", "Genus", "Scientific name"]
RAW_ENVIRONMENT = {
    "pH": "pH",
    "Temperature ℃": "Temperature",
    "Salinity": "Salinity",
    "Light intensity μmol m-2 s-1": "Light intensity",
    "Photoperiod h": "Photoperiod",
    "Density g L-1": "Density",
    "Experimental duration h": "Experimental duration",
}
NUTRIENT_COLUMNS = {
    "NO3": (
        "NO3--N concentration mg L-1",
        "NO3--N removal rate μg g-1 FW h-1",
    ),
    "PO4": (
        "PO43--P concentration mg L-1",
        "PO43--P removal rate μg g-1 FW h-1",
    ),
}
NUMERIC = [
    "pH",
    "Temperature",
    "Salinity",
    "Light intensity",
    "Photoperiod",
    "Density",
    "Experimental duration",
    "initial_concentration",
]
CATEGORICAL = TAXONOMY + ["Data category", "nutrient_type"]
FEATURES = NUMERIC + CATEGORICAL


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "tabpfn": package_version("tabpfn"),
        "torch": package_version("torch"),
        "scikit_learn": package_version("scikit-learn"),
    }


def validate_tabpfn_version() -> None:
    installed = package_version("tabpfn")
    if installed != EXPECTED_TABPFN_VERSION:
        raise RuntimeError(
            "This analysis is frozen to TabPFN "
            f"{EXPECTED_TABPFN_VERSION}; found {installed}."
        )


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    if torch is not None:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().replace({"": pd.NA, "-": pd.NA, "--": pd.NA})
    values = pd.to_numeric(text, errors="coerce")
    missing = values.isna() & text.notna()
    if missing.any():
        extracted = text[missing].str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
        values.loc[missing] = pd.to_numeric(extracted, errors="coerce")
    return values


def build_training_table() -> pd.DataFrame:
    if not PATHS.experimental_xlsx.exists():
        raise FileNotFoundError(PATHS.experimental_xlsx)
    raw = pd.read_excel(PATHS.experimental_xlsx)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw["original_row_id"] = np.arange(len(raw), dtype=int)

    required = ["Number", "Data category", *TAXONOMY, *RAW_ENVIRONMENT]
    for concentration, response in NUTRIENT_COLUMNS.values():
        required.extend((concentration, response))
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError(f"Experimental table is missing columns: {missing}")

    parts = []
    for nutrient in NUTRIENTS:
        concentration, response = NUTRIENT_COLUMNS[nutrient]
        part = pd.DataFrame(
            {
                "original_row_id": raw["original_row_id"],
                "Number": raw["Number"],
                "Data category": raw["Data category"],
                "nutrient_type": nutrient,
                "initial_concentration": numeric(raw[concentration]),
                "removal_rate": numeric(raw[response]),
            }
        )
        for column in TAXONOMY:
            part[column] = raw[column]
        for source, target in RAW_ENVIRONMENT.items():
            part[target] = numeric(raw[source])
        parts.append(part)

    data = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL:
        data[column] = data[column].astype("string").fillna("Missing")
    data = data[data["removal_rate"].notna() & (data["removal_rate"] >= 0)].copy()
    data["target_log1p"] = np.log1p(data["removal_rate"].to_numpy(float))
    return data.reset_index(drop=True)


def remove_upper_outliers(data: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(data), dtype=bool)
    audit = []
    for nutrient, rows in data.groupby("nutrient_type", observed=True).groups.items():
        response = data.loc[rows, "removal_rate"]
        q1, q3 = response.quantile([0.25, 0.75])
        upper = q3 + IQR_MULTIPLIER * (q3 - q1)
        drop = data.index.isin(rows) & (data["removal_rate"].to_numpy(float) > upper)
        keep &= ~drop
        audit.append(
            {
                "nutrient_type": nutrient,
                "q1": q1,
                "q3": q3,
                "upper": upper,
                "n_removed": int(drop.sum()),
            }
        )
    pd.DataFrame(audit).to_csv(MODEL_DIR / "outlier_thresholds.csv", index=False)
    return data.loc[keep].reset_index(drop=True)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def predict(model, matrix: np.ndarray) -> np.ndarray:
    chunks = []
    for start in range(0, len(matrix), PREDICT_BATCH_SIZE):
        batch = matrix[start : start + PREDICT_BATCH_SIZE]
        try:
            values = model.predict(batch, output_type="mean")
        except TypeError:
            values = model.predict(batch)
        chunks.append(np.asarray(values).reshape(-1))
    return np.concatenate(chunks).astype("float32")


def duan_smear(observed_log: np.ndarray, predicted_log: np.ndarray) -> float:
    residual = np.asarray(observed_log, float) - np.asarray(predicted_log, float)
    value = float(np.mean(np.exp(residual[np.isfinite(residual)])))
    return value if np.isfinite(value) and value > 0 else 1.0


def backtransform(predicted_log: np.ndarray, smear: float) -> np.ndarray:
    return np.clip(np.exp(np.asarray(predicted_log, float)) * smear - 1, 0, None).astype("float32")


def metrics(observed, predicted) -> dict[str, float]:
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    span = float(observed.max() - observed.min())
    return {
        "n": int(len(observed)),
        "r2": float(r2_score(observed, predicted)),
        "rmse": rmse,
        "nrmse_range_pct": 100 * rmse / span if span > 0 else np.nan,
    }


def grouped_cross_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = data[
        [
            "original_row_id",
            "Number",
            "Scientific name",
            "nutrient_type",
            "removal_rate",
        ]
    ].copy()
    oof["fold"] = -1
    oof["predicted_removal_rate"] = np.nan
    oof["fold_smear"] = np.nan
    x = data[FEATURES]
    y = data["target_log1p"].to_numpy("float32")
    rows = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(x, data["nutrient_type"], groups=data["original_row_id"]), start=1
    ):
        set_seed()
        prep = preprocessor()
        x_train = np.asarray(prep.fit_transform(x.iloc[train_idx]), dtype="float32")
        x_valid = np.asarray(prep.transform(x.iloc[valid_idx]), dtype="float32")
        model = TabPFNRegressor(device=DEVICE, random_state=SEED)
        model.fit(x_train, y[train_idx])
        smear = duan_smear(y[train_idx], predict(model, x_train))
        prediction = backtransform(predict(model, x_valid), smear)
        oof.loc[
            valid_idx,
            ["fold", "predicted_removal_rate", "fold_smear"],
        ] = fold, prediction, smear
        for nutrient in NUTRIENTS:
            mask = data.iloc[valid_idx]["nutrient_type"].eq(nutrient).to_numpy()
            rows.append(
                {
                    "scope": "fold",
                    "fold": fold,
                    "endpoint": nutrient,
                    **metrics(
                        data.iloc[valid_idx].loc[mask, "removal_rate"],
                        prediction[mask],
                    ),
                    "duan_smear": smear,
                }
            )

    for nutrient in NUTRIENTS:
        subset = oof[oof["nutrient_type"].eq(nutrient)]
        rows.append(
            {
                "scope": "overall_oof",
                "fold": np.nan,
                "endpoint": nutrient,
                **metrics(
                    subset["removal_rate"],
                    subset["predicted_removal_rate"],
                ),
                "duan_smear": np.nan,
            }
        )
    return oof, pd.DataFrame(rows)


def training_ranges(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges = {}
    for column in ("pH", "Temperature", "Salinity", "Light intensity"):
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        ranges[column] = {"min": float(values.min()), "max": float(values.max())}
    for nutrient in NUTRIENTS:
        values = data.loc[data["nutrient_type"].eq(nutrient), "initial_concentration"].dropna()
        ranges[f"initial_concentration_{nutrient}"] = {
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return ranges


def fit_final_model(data: pd.DataFrame, cv_metrics: pd.DataFrame) -> None:
    set_seed()
    prep = preprocessor()
    matrix = np.asarray(prep.fit_transform(data[FEATURES]), dtype="float32")
    target = data["target_log1p"].to_numpy("float32")
    model = TabPFNRegressor(device=DEVICE, random_state=SEED)
    model.fit(matrix, target)
    smear = duan_smear(target, predict(model, matrix))
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": prep,
            "model": model,
            "smear": smear,
            "ranges": training_ranges(data),
            "run_id": RUN_ID,
            "nutrients_for_training": list(NUTRIENTS),
            "cv_folds": N_SPLITS,
            "random_state": SEED,
            "software_versions": software_versions(),
            "cv_metrics": cv_metrics.to_dict(orient="records"),
        },
        BUNDLE_PATH,
    )


def main() -> None:
    validate_tabpfn_version()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = remove_upper_outliers(build_training_table())
    data.to_csv(MODEL_DIR / "long_training_after_outlier.csv", index=False, encoding="utf-8-sig")
    oof, cv_metrics = grouped_cross_validation(data)
    oof.to_csv(MODEL_DIR / "model_5cv_oof_predictions.csv", index=False, encoding="utf-8-sig")
    cv_metrics.to_csv(MODEL_DIR / "model_5cv_metrics.csv", index=False, encoding="utf-8-sig")
    fit_final_model(data, cv_metrics)
    summary = {
        "run_id": RUN_ID,
        "endpoints": list(NUTRIENTS),
        "seed": SEED,
        "folds": N_SPLITS,
        "outlier_rule": "endpoint-specific upper tail > Q3 + 3*IQR",
        "nrmse_definition": "100 * RMSE / observed range",
        "software_versions": software_versions(),
        "model_bundle": str(BUNDLE_PATH),
    }
    (MODEL_DIR / "model_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(cv_metrics[cv_metrics["scope"].eq("overall_oof")].to_string(index=False))


if __name__ == "__main__":
    main()
