"""Construct constrained potential, nutrient pressure and frozen mismatch states."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths, SCENARIOS
from macroalgae_repro.spatial import area_weight
from macroalgae_repro.statistics import weighted_fraction, weighted_mean, weighted_quantile


PATHS = ProjectPaths.from_env()
OUTPUT_DIR = PATHS.output_dir
RESULT_DIR = OUTPUT_DIR / "cross_scenario_comparison_and_mismatch_final"
PIXEL_DIR = RESULT_DIR / "recalculated_pixels"
RUN_ID = "no3_po4_5cv_seed42_log_duan_iqr3"
CACHE_DIR = OUTPUT_DIR / "cache" / RUN_ID / "predictions"
SPECIES_CSV = OUTPUT_DIR / "species_lookup.csv"

FUTURES = ("ssp245", "ssp370", "ssp585")
QUANTILES = (0.70, 0.75, 0.80, 0.85)
ACTIVE_HOURS = 8760.0
MEOW_MIN = 3
LOSS_EPS = 1e-6


def clean(value) -> str:
    return "" if pd.isna(value) else " ".join(str(value).replace("\xa0", " ").split()).strip()


def read_csv(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def right_continuous_ecdf(values, reference, reference_weight) -> np.ndarray:
    values = np.asarray(values, float)
    reference = np.asarray(reference, float)
    reference_weight = np.asarray(reference_weight, float)
    valid = np.isfinite(reference) & np.isfinite(reference_weight) & (reference_weight > 0)
    reference, reference_weight = reference[valid], reference_weight[valid]
    order = np.argsort(reference, kind="mergesort")
    reference, reference_weight = reference[order], reference_weight[order]
    cumulative = np.cumsum(reference_weight)
    out = np.full(values.shape, np.nan, dtype="float32")
    valid = np.isfinite(values)
    index = np.searchsorted(reference, values[valid], side="right") - 1
    percentile = np.zeros(index.shape, float)
    inside = index >= 0
    percentile[inside] = cumulative[index[inside]] / cumulative[-1] * 100
    out[valid] = percentile.astype("float32")
    return out


def load_common_pixels():
    """Load the four scenarios and freeze nutrient ECDFs from the full baseline domain."""
    frames, indexes = {}, {}
    references = {}

    for scenario in SCENARIOS:
        path = OUTPUT_DIR / f"potential_pixels_{scenario}_depth50m_025deg.csv"
        frame = read_csv(path)
        required = ["lon", "lat", "NO3_mmol_m3", "PO4_mmol_m3", "meow_province"]
        if scenario == "baseline":
            required += ["w_no3", "w_po4"]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        frame["_row"] = np.arange(len(frame), dtype=np.int64)
        numeric = ["lon", "lat", "NO3_mmol_m3", "PO4_mmol_m3"]
        if scenario == "baseline":
            numeric += ["w_no3", "w_po4"]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=["lon", "lat"]).copy()
        frame["_lon"] = frame["lon"].round(6)
        frame["_lat"] = frame["lat"].round(6)
        if frame.duplicated(["_lon", "_lat"]).any():
            raise ValueError(f"Duplicate coordinates in {path.name}")

        if scenario == "baseline":
            reference_weight = area_weight(frame["lat"])
            for nutrient in ("NO3", "PO4"):
                column = f"{nutrient}_mmol_m3"
                valid = (
                    np.isfinite(frame[column].to_numpy(float))
                    & np.isfinite(reference_weight)
                    & (reference_weight > 0)
                )
                references[nutrient] = (
                    frame.loc[valid, column].to_numpy(float),
                    np.asarray(reference_weight[valid], float),
                )

        frames[scenario] = frame
        indexes[scenario] = pd.MultiIndex.from_frame(frame[["_lon", "_lat"]])

    if any(nutrient not in references for nutrient in ("NO3", "PO4")):
        raise RuntimeError("Failed to build baseline nutrient references")
    if any(not references[nutrient][0].size for nutrient in ("NO3", "PO4")):
        raise RuntimeError("Baseline nutrient reference is empty")

    common = indexes["baseline"]
    for scenario in FUTURES:
        common = common.intersection(indexes[scenario], sort=False)
    common = pd.MultiIndex.from_frame(
        common.to_frame(index=False).sort_values(["_lon", "_lat"], kind="mergesort")
    )
    for scenario in SCENARIOS:
        frames[scenario] = (
            frames[scenario]
            .set_index(["_lon", "_lat"])
            .reindex(common)
            .reset_index(drop=True)
        )

    return frames, references


def allowed_sets() -> dict[str, set[str]]:
    occurrence = read_csv(PATHS.meow_occurrence_summary)
    occurrence["occurrence_count"] = pd.to_numeric(
        occurrence["occurrence_count"], errors="coerce"
    ).fillna(0)
    occurrence = occurrence[occurrence["occurrence_count"] >= MEOW_MIN].copy()
    occurrence["species"] = occurrence["accepted_name"].map(lambda x: clean(x).lower())
    occurrence["province"] = occurrence["meow_province"].map(lambda x: clean(x).lower())
    return occurrence.groupby("species")["province"].apply(set).to_dict()


def load_species() -> pd.DataFrame:
    species = read_csv(SPECIES_CSV).sort_values("species_id").reset_index(drop=True)
    species["display_name"] = species.get("meow_match_name", species["Scientific name"]).map(clean)
    return species


def build_allowed_mask(species: pd.DataFrame, provinces) -> np.ndarray:
    allowed = allowed_sets()
    provinces = np.array([clean(x).lower() for x in provinces], dtype=object)
    return np.vstack(
        [
            np.isin(provinces, tuple(allowed.get(clean(name).lower(), ())))
            for name in species["display_name"]
        ]
    )


def load_predictions(scenario: str, species_ids, rows) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, np.intp)
    matrices = []
    for nutrient in ("NO3", "PO4"):
        matrix = np.empty((len(species_ids), len(rows)), dtype="float32")
        for i, species_id in enumerate(species_ids):
            path = CACHE_DIR / scenario / nutrient / f"species_{int(species_id):05d}.npy"
            values = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
            matrix[i] = values[rows]
        matrices.append(matrix)
    return matrices[0], matrices[1]


def best_annualized(combined: np.ndarray, allow: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(combined)
    if allow is not None:
        valid &= allow
    available = valid.any(axis=0)
    out = np.full(combined.shape[1], np.nan, dtype="float32")
    columns = np.flatnonzero(available)
    if columns.size:
        best = np.argmax(np.where(valid[:, columns], combined[:, columns], -np.inf), axis=0)
        out[columns] = combined[best, columns] * np.float32(ACTIVE_HOURS / 1000)
    return out


def safe_ratio(numerator, denominator) -> np.ndarray:
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    out = np.full(numerator.shape, np.nan, dtype="float32")
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def calculate(frames, references):
    species = load_species()
    base = frames["baseline"]
    w_no3 = pd.to_numeric(base["w_no3"], errors="coerce").to_numpy("float32")
    w_po4 = pd.to_numeric(base["w_po4"], errors="coerce").to_numpy("float32")
    if not np.allclose(w_no3 + w_po4, 1, atol=1e-5, equal_nan=False):
        raise ValueError("Invalid baseline nutrient weights")

    base_province = base["meow_province"].map(clean).reset_index(drop=True)
    for scenario in FUTURES:
        if not base_province.equals(
            frames[scenario]["meow_province"].map(clean).reset_index(drop=True)
        ):
            raise ValueError(f"MEOW province assignments differ under {scenario}")
    allow = build_allowed_mask(species, base_province)

    weights = area_weight(pd.to_numeric(base["lat"], errors="coerce"))
    no3_ref, no3_ref_weight = references["NO3"]
    po4_ref, po4_ref_weight = references["PO4"]
    results = {}

    for scenario in SCENARIOS:
        frame = frames[scenario]
        pred_no3, pred_po4 = load_predictions(scenario, species["species_id"], frame["_row"])
        combined = pred_no3 * w_no3[None, :] + pred_po4 * w_po4[None, :]
        unconstrained = best_annualized(combined)
        constrained = best_annualized(combined, allow)
        loss = unconstrained - constrained
        loss[np.isfinite(loss) & (np.abs(loss) < LOSS_EPS)] = 0
        if np.any(np.isfinite(loss) & (loss < 0)):
            raise RuntimeError(f"Negative constraint loss under {scenario}")
        pressure = (
            right_continuous_ecdf(frame["NO3_mmol_m3"], no3_ref, no3_ref_weight) * w_no3
            + right_continuous_ecdf(frame["PO4_mmol_m3"], po4_ref, po4_ref_weight) * w_po4
        ).astype("float32")
        results[scenario] = {
            "unconstrained_potential": unconstrained,
            "constrained_potential": constrained,
            "constraint_loss": loss,
            "constraint_ratio": safe_ratio(constrained, unconstrained),
            "nutrient_pressure": pressure,
        }
        del pred_no3, pred_po4, combined
    return results, weights, w_no3, w_po4


def valid_masks(results, weights):
    """Build cross-scenario valid domains separately for each derived metric."""
    base = np.isfinite(weights) & (weights > 0)
    masks = {}
    for metric in (
        "unconstrained_potential",
        "constrained_potential",
        "constraint_loss",
        "constraint_ratio",
        "nutrient_pressure",
    ):
        valid = base.copy()
        for scenario in SCENARIOS:
            valid &= np.isfinite(results[scenario][metric])
        masks[metric] = valid

    mismatch = base.copy()
    for scenario in SCENARIOS:
        mismatch &= np.isfinite(results[scenario]["nutrient_pressure"])
        mismatch &= np.isfinite(results[scenario]["constrained_potential"])
    masks["mismatch"] = mismatch
    return masks


def add_mismatch(results, valid, weights):
    thresholds = []
    for q in QUANTILES:
        pressure_threshold = weighted_quantile(
            results["baseline"]["nutrient_pressure"][valid],
            q,
            weights[valid],
        )
        potential_threshold = weighted_quantile(
            results["baseline"]["constrained_potential"][valid],
            q,
            weights[valid],
        )
        thresholds.append(
            {
                "quantile": q,
                "nutrient_pressure_threshold": pressure_threshold,
                "constrained_potential_threshold": potential_threshold,
            }
        )
        key = f"mismatch_q{int(q * 100)}"
        for scenario in SCENARIOS:
            results[scenario][key] = (
                (results[scenario]["nutrient_pressure"] >= pressure_threshold)
                & (results[scenario]["constrained_potential"] < potential_threshold)
                & valid
            ).astype("int8")
    for scenario in SCENARIOS:
        results[scenario]["stable_mismatch"] = np.logical_and.reduce(
            [results[scenario][f"mismatch_q{int(q * 100)}"].astype(bool) for q in QUANTILES]
        ).astype("int8")
    pd.DataFrame(thresholds).to_csv(RESULT_DIR / "baseline_reference_thresholds.csv", index=False)


def save_outputs(frames, results, weights, w_no3, w_po4, masks):
    PIXEL_DIR.mkdir(parents=True, exist_ok=True)
    summary, transitions = [], []
    for scenario in SCENARIOS:
        frame = frames[scenario]
        out = pd.DataFrame(
            {
                "scenario": scenario,
                "lon": frame["lon"].to_numpy(float),
                "lat": frame["lat"].to_numpy(float),
                "area_weight": weights,
                "NO3_mmol_m3": frame["NO3_mmol_m3"].to_numpy(float),
                "PO4_mmol_m3": frame["PO4_mmol_m3"].to_numpy(float),
                "w_no3_base": w_no3,
                "w_po4_base": w_po4,
                **{k: v for k, v in results[scenario].items()},
                "valid_constrained_potential": masks["constrained_potential"],
                "valid_constraint_loss": masks["constraint_loss"],
                "valid_constraint_ratio": masks["constraint_ratio"],
                "valid_nutrient_pressure": masks["nutrient_pressure"],
                "valid_mismatch": masks["mismatch"],
            }
        )
        out.to_csv(PIXEL_DIR / f"recalculated_{scenario}.csv", index=False, encoding="utf-8-sig")
        summary.append(
            {
                "scenario": scenario,
                "constrained_potential_mean": weighted_mean(
                    results[scenario]["constrained_potential"],
                    weights,
                    masks["constrained_potential"],
                ),
                "constraint_loss_mean": weighted_mean(
                    results[scenario]["constraint_loss"],
                    weights,
                    masks["constraint_loss"],
                ),
                "mismatch_q75_fraction": weighted_fraction(
                    results[scenario]["mismatch_q75"].astype(bool),
                    masks["mismatch"],
                    weights,
                ),
                "stable_mismatch_fraction": weighted_fraction(
                    results[scenario]["stable_mismatch"].astype(bool),
                    masks["mismatch"],
                    weights,
                ),
            }
        )

    baseline = results["baseline"]["mismatch_q75"].astype(bool)
    for scenario in FUTURES:
        future = results[scenario]["mismatch_q75"].astype(bool)
        transitions.extend(
            [
                {
                    "scenario": scenario,
                    "class": "persistent",
                    "area_fraction": weighted_fraction(
                        baseline & future, masks["mismatch"], weights
                    ),
                },
                {
                    "scenario": scenario,
                    "class": "emerging",
                    "area_fraction": weighted_fraction(
                        ~baseline & future, masks["mismatch"], weights
                    ),
                },
                {
                    "scenario": scenario,
                    "class": "recovery",
                    "area_fraction": weighted_fraction(
                        baseline & ~future, masks["mismatch"], weights
                    ),
                },
            ]
        )
    pd.DataFrame(summary).to_csv(RESULT_DIR / "scenario_specific_delta_summary.csv", index=False)
    pd.DataFrame(transitions).to_csv(RESULT_DIR / "transition_area_summary.csv", index=False)

    consistency = []
    for metric in ("constrained_potential", "constraint_loss"):
        delta = np.vstack([results[s][metric] - results["baseline"][metric] for s in FUTURES])
        increase = np.all(delta > 0, axis=0)
        decrease = np.all(delta < 0, axis=0)
        sensitive = ~(increase | decrease)
        for label, flag in (
            ("consistent_increase", increase),
            ("consistent_decrease", decrease),
            ("scenario_sensitive", sensitive),
        ):
            consistency.append(
                {
                    "metric": metric,
                    "class": label,
                    "area_fraction": weighted_fraction(flag, masks[metric], weights),
                }
            )
    pd.DataFrame(consistency).to_csv(
        RESULT_DIR / "cross_scenario_consistency_summary.csv", index=False
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frames, references = load_common_pixels()
    results, weights, w_no3, w_po4 = calculate(frames, references)
    masks = valid_masks(results, weights)
    add_mismatch(results, masks["mismatch"], weights)
    save_outputs(frames, results, weights, w_no3, w_po4, masks)
    manifest = {
        "pressure_reference": "area-weighted right-continuous baseline ECDF",
        "potential_weights": "baseline Redfield-normalized local weights frozen across scenarios",
        "mismatch_quantiles": list(QUANTILES),
        "area_weight": "cos(latitude)",
        "common_domain_pixels": int(masks["mismatch"].sum()),
    }
    (RESULT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
