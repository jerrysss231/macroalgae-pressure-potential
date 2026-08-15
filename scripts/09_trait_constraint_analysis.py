"""Trait selection, Rao's Q, and functional-substitution analyses."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths

PATHS = ProjectPaths.from_env()
ROOT = PATHS.output_dir
TRAIT_DIR = ROOT / "traits"
OUT_DIR = ROOT / "trait_constraint_analysis"
PIXEL_DIR = ROOT / "cross_scenario_comparison_and_mismatch_final" / "recalculated_pixels"
CONTEXT_CSV = ROOT / "ecological_heterogeneity" / "ecological_pixel_context.csv.gz"
PREDICTION_CACHE = (
    ROOT / "cache" / "no3_po4_5cv_seed42_log_duan_iqr3" / "predictions"
)

SCENARIOS = ("baseline", "ssp245", "ssp370", "ssp585")
FUNCTIONAL_DIMENSIONS = ("habitat", "life_history", "morphology", "reproduction")
PRIMARY_COVERAGE_GATE = 0.80
N_PERMUTATIONS = 9_999
N_BOOTSTRAPS = 4_999
RANDOM_SEED = 42
EPS = 1e-12
POSITIVE_LOSS_EPS = 1e-6
SPATIAL_BLOCK_DEGREES = 5.0


def read_csv(path, **kwargs):
    """Read a UTF-8 CSV and fail clearly when the input is absent."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def normalize_key(value):
    """Normalize species and province names for deterministic joins."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split()).strip().lower()


def weighted_mean(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_fraction(flag, valid, weights):
    flag = np.asarray(flag, bool)
    valid = np.asarray(valid, bool)
    weights = np.asarray(weights, float)
    use = valid & np.isfinite(weights) & (weights > 0)
    if not use.any():
        return np.nan
    return float(np.sum(weights[use & flag]) / np.sum(weights[use]))


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    q_values = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    adjusted[order] = np.minimum(q_values, 1.0)
    return adjusted


def spatial_block_id(longitude, latitude):
    longitude = np.asarray(longitude, float)
    latitude = np.asarray(latitude, float)
    block_x = np.floor((longitude + 180.0) % 360.0 / SPATIAL_BLOCK_DEGREES).astype(int)
    block_y = np.floor((latitude + 90.0) / SPATIAL_BLOCK_DEGREES).astype(int)
    return block_x + 10_000 * block_y


def load_traits_and_distances():
    traits = read_csv(TRAIT_DIR / "direct_traits_cleaned.csv")
    metadata = read_csv(TRAIT_DIR / "trait_metadata.csv")
    traits["name"] = traits["meow_match_name"].map(normalize_key)

    distance = pd.read_csv(TRAIT_DIR / "functional_distance.csv", index_col=0)
    distance.index = [normalize_key(value) for value in distance.index]
    distance.columns = [normalize_key(value) for value in distance.columns]

    dimension_distances = {}
    for dimension in FUNCTIONAL_DIMENSIONS:
        matrix = pd.read_csv(
            TRAIT_DIR / f"functional_distance_{dimension}.csv",
            index_col=0,
        )
        matrix.index = [normalize_key(value) for value in matrix.index]
        matrix.columns = [normalize_key(value) for value in matrix.columns]
        dimension_distances[dimension] = matrix

    primary_traits = metadata["clean_trait"].tolist()
    missing_traits = [column for column in primary_traits if column not in traits.columns]
    if missing_traits:
        raise ValueError(f"Cleaned trait table missing primary traits: {missing_traits}")

    # Trait inference uses the frozen direct-trait complete-case species set.
    resolved = traits[primary_traits].notna().all(axis=1).to_numpy(bool)
    traits["trait_resolved"] = resolved
    resolved_names = traits.loc[resolved, "name"].tolist()
    resolved_distance = distance.loc[resolved_names, resolved_names].to_numpy(float)
    if not np.isfinite(resolved_distance).all():
        raise ValueError("Incomplete functional distances among trait-resolved species.")

    return traits, metadata, distance, dimension_distances


def load_model_species_mapping(traits):
    model_species = read_csv(ROOT / "species_lookup.csv")
    model_species = model_species.sort_values("species_id").reset_index(drop=True)
    model_species["name"] = model_species["meow_match_name"].map(normalize_key)

    biological_lookup = {name: i for i, name in enumerate(traits["name"])}
    mapping = np.array(
        [biological_lookup.get(name, -1) for name in model_species["name"]],
        dtype=int,
    )
    if (mapping < 0).any():
        raise ValueError("At least one model species could not be mapped to a biological species.")

    return model_species, mapping


def load_pixel_context():
    baseline = read_csv(PIXEL_DIR / "recalculated_baseline.csv")
    baseline["lon_key"] = baseline["lon"].round(6)
    baseline["lat_key"] = baseline["lat"].round(6)

    context = read_csv(CONTEXT_CSV)
    context["lon_key"] = context["lon"].round(6)
    context["lat_key"] = context["lat"].round(6)
    context_cols = ["lon_key", "lat_key", "meow_province", "meow_realm"]

    pixels = baseline.merge(
        context[context_cols].drop_duplicates(["lon_key", "lat_key"]),
        on=["lon_key", "lat_key"],
        validate="one_to_one",
    )

    response_columns = (
        "unconstrained_potential",
        "constrained_potential",
        "constraint_loss",
        "nutrient_pressure",
        "valid_constraint_loss",
    )
    pixels.rename(
        columns={column: f"{column}_baseline" for column in response_columns},
        inplace=True,
    )

    for scenario in SCENARIOS[1:]:
        future = read_csv(PIXEL_DIR / f"recalculated_{scenario}.csv")
        future["lon_key"] = future["lon"].round(6)
        future["lat_key"] = future["lat"].round(6)
        future = future[["lon_key", "lat_key", *response_columns]].rename(
            columns={column: f"{column}_{scenario}" for column in response_columns}
        )
        pixels = pixels.merge(
            future,
            on=["lon_key", "lat_key"],
            validate="one_to_one",
        )

    pixels["block_id"] = spatial_block_id(pixels["lon"], pixels["lat"])
    return pixels


def build_availability_masks(model_species, mapping, traits, provinces):
    occurrences = read_csv(PATHS.meow_occurrence_summary)
    occurrences["occurrence_count"] = pd.to_numeric(
        occurrences["occurrence_count"], errors="coerce"
    ).fillna(0)
    occurrences = occurrences[occurrences["occurrence_count"] >= 3].copy()

    occurrence_sets = (
        occurrences.assign(
            species_key=occurrences["accepted_name"].map(normalize_key),
            province_key=occurrences["meow_province"].map(normalize_key),
        )
        .groupby("species_key")["province_key"]
        .apply(set)
        .to_dict()
    )

    province_keys = np.array([normalize_key(value) for value in provinces], dtype=object)
    model_available = np.vstack(
        [
            np.isin(province_keys, tuple(occurrence_sets.get(name, ())))
            for name in model_species["name"]
        ]
    )

    biological_available = np.zeros((len(traits), len(province_keys)), dtype=bool)
    for model_index, biological_index in enumerate(mapping):
        biological_available[biological_index] |= model_available[model_index]

    return model_available, biological_available


def calculate_pool_metrics(
    biological_available,
    traits,
    distance,
    dimension_distances,
    provinces,
):
    resolved_species = np.flatnonzero(traits["trait_resolved"].to_numpy(bool))
    resolved_names = traits.loc[resolved_species, "name"].tolist()
    resolved_lookup = {species: i for i, species in enumerate(resolved_species)}

    overall_distance = distance.loc[resolved_names, resolved_names].to_numpy(float)
    per_dimension = {
        dimension: matrix.loc[resolved_names, resolved_names].to_numpy(float)
        for dimension, matrix in dimension_distances.items()
    }

    province_keys, province_codes = np.unique(
        np.array([normalize_key(value) for value in provinces], dtype=object),
        return_inverse=True,
    )

    substitution_gaps = {
        "overall": np.full((len(traits), len(province_keys)), np.nan)
    }
    substitution_gaps.update(
        {
            dimension: np.full_like(substitution_gaps["overall"], np.nan)
            for dimension in FUNCTIONAL_DIMENSIONS
        }
    )

    rows = []
    for code, province_key in enumerate(province_keys):
        representative_pixel = np.flatnonzero(province_codes == code)[0]
        candidate_species = np.flatnonzero(
            biological_available[:, representative_pixel]
        )
        trait_resolved_candidates = np.array(
            [species for species in candidate_species if species in resolved_lookup],
            dtype=int,
        )
        local_indices = np.array(
            [resolved_lookup[species] for species in trait_resolved_candidates],
            dtype=int,
        )

        n_candidates = len(candidate_species)
        n_resolved = len(local_indices)
        row = {
            "province_key": province_key,
            "candidate_richness": n_candidates,
            "trait_resolved_richness": n_resolved,
            "pool_trait_coverage": (
                n_resolved / n_candidates if n_candidates else np.nan
            ),
            "rao_q": _rao_q(overall_distance, local_indices),
        }
        for dimension in FUNCTIONAL_DIMENSIONS:
            row[f"rao_q_{dimension}"] = _rao_q(
                per_dimension[dimension], local_indices
            )

        if n_resolved:
            for biological_species in resolved_species:
                resolved_index = resolved_lookup[biological_species]
                substitution_gaps["overall"][biological_species, code] = np.min(
                    overall_distance[resolved_index, local_indices]
                )
                for dimension in FUNCTIONAL_DIMENSIONS:
                    substitution_gaps[dimension][biological_species, code] = np.min(
                        per_dimension[dimension][resolved_index, local_indices]
                    )

        rows.append(row)

    return pd.DataFrame(rows), province_codes, substitution_gaps


def _rao_q(distance_matrix, local_indices):
    n_species = len(local_indices)
    if n_species == 0:
        return np.nan
    if n_species == 1:
        return 0.0
    local = distance_matrix[np.ix_(local_indices, local_indices)]
    return float(np.sum(local) / n_species**2)


def prediction_rows(pixels, scenario):
    projected = read_csv(ROOT / f"potential_pixels_{scenario}_depth50m_025deg.csv")
    projected["lon_key"] = projected["lon"].round(6)
    projected["lat_key"] = projected["lat"].round(6)
    projected["row"] = np.arange(len(projected))
    joined = pixels[["lon_key", "lat_key"]].merge(
        projected[["lon_key", "lat_key", "row"]],
        on=["lon_key", "lat_key"],
        validate="one_to_one",
    )
    return joined["row"].to_numpy(int)


def combined_species_predictions(model_species, rows, scenario, weight_n, weight_p):
    endpoint_predictions = []
    for nutrient in ("NO3", "PO4"):
        values = np.empty((len(model_species), len(rows)), dtype=np.float32)
        for i, species_id in enumerate(model_species["species_id"]):
            path = (
                PREDICTION_CACHE
                / scenario
                / nutrient
                / f"species_{int(species_id):05d}.npy"
            )
            values[i] = np.load(path, mmap_mode="r")[rows]
        endpoint_predictions.append(values)

    return (
        endpoint_predictions[0] * weight_n[None, :]
        + endpoint_predictions[1] * weight_p[None, :]
    )


def best_species(predictions, availability=None):
    eligible = np.isfinite(predictions)
    if availability is not None:
        eligible &= availability

    has_candidate = eligible.any(axis=0)
    winner = np.full(predictions.shape[1], -1, dtype=int)
    columns = np.flatnonzero(has_candidate)
    if len(columns):
        winner[columns] = np.argmax(
            np.where(eligible[:, columns], predictions[:, columns], -np.inf),
            axis=0,
        )
    return winner, eligible


def collapse_model_mask(mask, mapping, n_biological_species):
    collapsed = np.zeros((n_biological_species, mask.shape[1]), dtype=bool)
    for model_index, biological_index in enumerate(mapping):
        collapsed[biological_index] |= mask[model_index]
    return collapsed


def selection_advantage(winner, eligible, weights, traits, scenario, selection_type):
    n_eligible = eligible.sum(axis=0)
    valid = np.isfinite(weights) & (weights > 0) & (n_eligible > 0)

    expected = eligible[:, valid] @ (weights[valid] / n_eligible[valid])
    winner_valid = valid & (winner >= 0)
    observed = np.bincount(
        winner[winner_valid],
        weights=weights[winner_valid],
        minlength=len(traits),
    )
    pseudocount = max(0.5 * np.min(weights[valid]), EPS)

    output = traits[
        ["species_id", "Scientific name", "name", "Phylum", "trait_resolved"]
    ].copy()
    output["scenario"] = scenario
    output["selection_type"] = selection_type
    output["selection_advantage_log2"] = np.where(
        expected > 0,
        np.log2((observed + pseudocount) / (expected + pseudocount)),
        np.nan,
    )
    return output


def ordinary_r2(y, design):
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    total = np.sum((y - y.mean()) ** 2)
    if total <= 0:
        return np.nan
    return 1.0 - np.sum((y - fitted) ** 2) / total


def encode_multistate_trait(series):
    states = sorted(
        {
            state
            for value in series.astype(str)
            for state in value.split(";")
            if state and state != "nan"
        }
    )
    if len(states) <= 1:
        return np.empty((len(series), 0))

    return np.column_stack(
        [
            series.astype(str).map(
                lambda value, state=state: float(state in value.split(";"))
            )
            for state in states[1:]
        ]
    )


def specific_trait_models(selection_table, traits, metadata, rng):
    trait_columns = metadata["clean_trait"].tolist()
    resolved_traits = traits.loc[traits["trait_resolved"]].copy()
    data = selection_table.merge(
        resolved_traits[["species_id", "Phylum", *trait_columns]],
        on=["species_id", "Phylum"],
        how="inner",
        validate="many_to_one",
    )

    rows = []
    for (scenario, selection_type), group in data.groupby(
        ["scenario", "selection_type"], observed=True
    ):
        local_results = []
        for trait_meta in metadata.itertuples():
            subset = group[
                group["selection_advantage_log2"].notna()
                & group[trait_meta.clean_trait].notna()
            ]
            if len(subset) < 10:
                continue

            y = subset["selection_advantage_log2"].to_numpy(float)
            phylum_terms = pd.get_dummies(
                subset["Phylum"].astype(str), drop_first=True, dtype=float
            )
            base = np.column_stack([np.ones(len(subset)), phylum_terms])
            trait_terms = encode_multistate_trait(subset[trait_meta.clean_trait])
            if not trait_terms.shape[1]:
                continue

            full = np.column_stack([base, trait_terms])
            observed_delta = ordinary_r2(y, full) - ordinary_r2(y, base)

            phylum = subset["Phylum"].to_numpy()
            phylum_groups = [
                np.flatnonzero(phylum == value) for value in np.unique(phylum)
            ]
            extreme = 0
            for _ in range(N_PERMUTATIONS):
                permuted = y.copy()
                for indices in phylum_groups:
                    permuted[indices] = rng.permutation(permuted[indices])
                permuted_delta = ordinary_r2(permuted, full) - ordinary_r2(
                    permuted, base
                )
                extreme += permuted_delta >= observed_delta

            local_results.append(
                {
                    "scenario": scenario,
                    "selection_type": selection_type,
                    "trait": trait_meta.clean_trait,
                    "functional_dimension": trait_meta.functional_dimension,
                    "n_species": len(subset),
                    "trait_added_r2": observed_delta,
                    "p": (extreme + 1) / (N_PERMUTATIONS + 1),
                }
            )

        adjusted = benjamini_hochberg([result["p"] for result in local_results])
        for result, q_value in zip(local_results, adjusted):
            result["p_fdr"] = q_value
            rows.append(result)

    return pd.DataFrame(rows)


def weighted_r2(data, response, features, weight_column):
    numeric = data[[response, weight_column, *features]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = np.isfinite(numeric).all(axis=1) & (numeric[weight_column] > 0)
    numeric = numeric[valid]
    if len(numeric) <= len(features) + 2:
        return np.nan

    design = np.column_stack([np.ones(len(numeric)), numeric[features]])
    y = numeric[response].to_numpy(float)
    weights = numeric[weight_column].to_numpy(float)
    sqrt_weights = np.sqrt(weights)

    coefficients = np.linalg.lstsq(
        design * sqrt_weights[:, None],
        y * sqrt_weights,
        rcond=None,
    )[0]
    fitted = design @ coefficients
    weighted_mean_y = np.sum(y * weights) / np.sum(weights)
    denominator = np.sum(weights * (y - weighted_mean_y) ** 2)
    if denominator <= 0:
        return np.nan
    return 1.0 - np.sum(weights * (y - fitted) ** 2) / denominator


def build_province_block_table(
    pixels,
    pool_metrics,
    province_codes,
    substitution_gaps,
    unconstrained_winners,
):
    pool_lookup = pool_metrics.set_index("province_key")
    province_keys = np.array(
        [normalize_key(value) for value in pixels["meow_province"]], dtype=object
    )
    area_weights = pixels["area_weight"].to_numpy(float)

    pool_columns = (
        "candidate_richness",
        "pool_trait_coverage",
        "rao_q",
        *[f"rao_q_{dimension}" for dimension in FUNCTIONAL_DIMENSIONS],
    )

    rows = []
    grouping = pixels.groupby(
        [pd.Series(province_keys, index=pixels.index), "block_id"], observed=True
    )

    for scenario in SCENARIOS:
        loss = pixels[f"constraint_loss_{scenario}"].to_numpy(float)
        pressure = pixels[f"nutrient_pressure_{scenario}"].to_numpy(float)
        unconstrained = pixels[f"unconstrained_potential_{scenario}"].to_numpy(float)
        winners = unconstrained_winners[scenario]

        for (province_key, block_id), group_indices in grouping.groups.items():
            indices = np.asarray(list(group_indices), dtype=int)
            weights = area_weights[indices]
            block_loss = loss[indices]
            if province_key not in pool_lookup.index:
                continue

            coverage_ok = (
                float(pool_lookup.loc[province_key, "pool_trait_coverage"])
                >= PRIMARY_COVERAGE_GATE
            )
            gap = pixels[f"trait_substitution_gap_{scenario}"].to_numpy(float)[indices]
            gap_valid = pixels[f"trait_gap_valid_{scenario}"].to_numpy(bool)[indices]
            loss_valid = (
                pd.to_numeric(
                    pixels[f"valid_constraint_loss_{scenario}"], errors="coerce"
                )
                .fillna(0)
                .to_numpy(float)[indices]
                > 0.5
            )
            valid = (
                coverage_ok
                & loss_valid
                & gap_valid
                & np.isfinite(block_loss)
                & np.isfinite(weights)
                & (weights > 0)
            )
            positive = valid & (block_loss > POSITIVE_LOSS_EPS)
            if not valid.any():
                continue

            incidence = weighted_fraction(positive, valid, weights)
            incidence_clipped = np.clip(incidence, 1e-4, 1.0 - 1e-4)
            row = {
                "province_key": province_key,
                "block_id": block_id,
                "scenario": scenario,
                "realm": pixels["meow_realm"].iloc[indices[0]],
                "area_weight_sum": float(np.sum(weights[valid])),
                "positive_area_weight_sum": float(np.sum(weights[positive])),
                "logit_loss_incidence": float(
                    np.log(incidence_clipped / (1.0 - incidence_clipped))
                ),
                "mean_positive_log_loss": (
                    weighted_mean(np.log1p(block_loss[positive]), weights[positive])
                    if positive.any()
                    else np.nan
                ),
                "mean_pressure": weighted_mean(pressure[indices][valid], weights[valid]),
                "mean_unconstrained_potential": weighted_mean(
                    unconstrained[indices][valid], weights[valid]
                ),
            }
            for column in pool_columns:
                row[column] = pool_lookup.loc[province_key, column]
            row["log_candidate_richness"] = np.log1p(row["candidate_richness"])

            overall_gap = gap
            row["mean_trait_gap"] = weighted_mean(overall_gap[valid], weights[valid])
            row["mean_positive_trait_gap"] = (
                weighted_mean(overall_gap[positive], weights[positive])
                if positive.any()
                else np.nan
            )
            row["exact_match_share"] = weighted_fraction(
                np.isfinite(overall_gap) & (overall_gap <= EPS),
                valid,
                weights,
            )
            exact_clipped = np.clip(row["exact_match_share"], 1e-4, 1.0 - 1e-4)
            row["logit_exact_match_share"] = float(
                np.log(exact_clipped / (1.0 - exact_clipped))
            )

            valid_winner = winners[indices] >= 0
            for dimension in FUNCTIONAL_DIMENSIONS:
                dimension_gap = np.full(len(indices), np.nan)
                dimension_gap[valid_winner] = substitution_gaps[dimension][
                    winners[indices][valid_winner],
                    province_codes[indices][valid_winner],
                ]
                row[f"mean_gap_{dimension}"] = weighted_mean(
                    dimension_gap[valid], weights[valid]
                )
                row[f"mean_positive_gap_{dimension}"] = (
                    weighted_mean(dimension_gap[positive], weights[positive])
                    if positive.any()
                    else np.nan
                )

            rows.append(row)

    return pd.DataFrame(rows)


def nested_constraint_models(blocks, rng):
    """Fit the frozen two-part nested models on matched complete-case block tables."""
    controls = pd.get_dummies(
        blocks[["scenario", "realm"]].astype(str), drop_first=True, dtype=float
    )
    frame = pd.concat(
        [
            blocks.reset_index(drop=True),
            controls.add_prefix("control_").reset_index(drop=True),
        ],
        axis=1,
    )
    control_columns = [column for column in frame if column.startswith("control_")]
    base_features = [
        "mean_pressure",
        "mean_unconstrained_potential",
        "pool_trait_coverage",
        *control_columns,
    ]

    response_specs = (
        (
            "loss_incidence",
            "logit_loss_incidence",
            "mean_trait_gap",
            "area_weight_sum",
        ),
        (
            "positive_loss_severity",
            "mean_positive_log_loss",
            "mean_positive_trait_gap",
            "positive_area_weight_sum",
        ),
    )

    def complete_case(data, columns, weight_column):
        numeric = data[[*columns, weight_column]].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric).all(axis=1) & (numeric[weight_column] > 0)
        return data.loc[valid].copy()

    def bootstrap_delta(data, response, reduced, full, weight_column):
        groups = [
            group.index.to_numpy()
            for _, group in data.groupby("province_key", observed=True)
        ]
        if not groups:
            return np.nan, np.nan

        values = np.empty(N_BOOTSTRAPS)
        for i in range(N_BOOTSTRAPS):
            sampled = rng.integers(0, len(groups), len(groups))
            indices = np.concatenate([groups[j] for j in sampled])
            sample = data.loc[indices]
            values[i] = (
                weighted_r2(sample, response, full, weight_column)
                - weighted_r2(sample, response, reduced, weight_column)
            )
        finite = values[np.isfinite(values)]
        if not len(finite):
            return np.nan, np.nan
        low, high = np.quantile(finite, [0.025, 0.975])
        return float(low), float(high)

    rows = []
    for component, response, gap_column, weight_column in response_specs:
        overall_features = [
            *base_features,
            "log_candidate_richness",
            "rao_q",
            gap_column,
        ]
        overall = complete_case(
            frame,
            [response, *overall_features],
            weight_column,
        )
        model_steps = (
            ("M0", base_features),
            ("M1_richness", [*base_features, "log_candidate_richness"]),
            (
                "M2_RaoQ",
                [*base_features, "log_candidate_richness", "rao_q"],
            ),
            ("M3_gap", overall_features),
        )

        previous_features = None
        previous_r2 = np.nan
        for model_step, features in model_steps:
            model_r2 = weighted_r2(overall, response, features, weight_column)
            ci_low, ci_high = (np.nan, np.nan)
            if previous_features is not None:
                ci_low, ci_high = bootstrap_delta(
                    overall,
                    response,
                    previous_features,
                    features,
                    weight_column,
                )
            rows.append(
                {
                    "loss_component": component,
                    "functional_dimension": "overall",
                    "model_step": model_step,
                    "n_blocks": len(overall),
                    "r2": model_r2,
                    "delta_r2": (
                        model_r2 - previous_r2
                        if np.isfinite(previous_r2)
                        else np.nan
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
            previous_features = features
            previous_r2 = model_r2

        for dimension in FUNCTIONAL_DIMENSIONS:
            diversity_column = f"rao_q_{dimension}"
            dimension_gap = (
                f"mean_gap_{dimension}"
                if component == "loss_incidence"
                else f"mean_positive_gap_{dimension}"
            )
            reduced = [*base_features, "log_candidate_richness"]
            with_diversity = [*reduced, diversity_column]
            with_gap = [*with_diversity, dimension_gap]
            dimension_data = complete_case(
                frame,
                [response, *with_gap],
                weight_column,
            )

            reduced_r2 = weighted_r2(
                dimension_data, response, reduced, weight_column
            )
            diversity_r2 = weighted_r2(
                dimension_data, response, with_diversity, weight_column
            )
            gap_r2 = weighted_r2(
                dimension_data, response, with_gap, weight_column
            )
            diversity_ci = bootstrap_delta(
                dimension_data,
                response,
                reduced,
                with_diversity,
                weight_column,
            )
            gap_ci = bootstrap_delta(
                dimension_data,
                response,
                with_diversity,
                with_gap,
                weight_column,
            )
            rows.extend(
                [
                    {
                        "loss_component": component,
                        "functional_dimension": dimension,
                        "model_step": "dimension_RaoQ",
                        "n_blocks": len(dimension_data),
                        "r2": diversity_r2,
                        "delta_r2": diversity_r2 - reduced_r2,
                        "ci_low": diversity_ci[0],
                        "ci_high": diversity_ci[1],
                    },
                    {
                        "loss_component": component,
                        "functional_dimension": dimension,
                        "model_step": "dimension_gap",
                        "n_blocks": len(dimension_data),
                        "r2": gap_r2,
                        "delta_r2": gap_r2 - diversity_r2,
                        "ci_low": gap_ci[0],
                        "ci_high": gap_ci[1],
                    },
                ]
            )

    return pd.DataFrame(rows)


def functional_alignment_models(blocks, rng):
    """Test whether overall Rao's Q adds information on functional alignment."""
    controls = pd.get_dummies(
        blocks[["scenario", "realm"]].astype(str), drop_first=True, dtype=float
    )
    frame = pd.concat(
        [
            blocks.reset_index(drop=True),
            controls.add_prefix("control_").reset_index(drop=True),
        ],
        axis=1,
    )
    control_columns = [column for column in frame if column.startswith("control_")]
    background = [
        "mean_pressure",
        "mean_unconstrained_potential",
        "pool_trait_coverage",
        "log_candidate_richness",
        *control_columns,
    ]

    def fit(data, response, features, weight_column):
        numeric = data[[response, weight_column, *features]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid = np.isfinite(numeric).all(axis=1) & (numeric[weight_column] > 0)
        data = data.loc[valid].copy()
        numeric = numeric.loc[valid]
        if len(data) <= len(features) + 2:
            return data, np.array([]), np.nan

        design = np.column_stack([np.ones(len(data)), numeric[features]])
        y = numeric[response].to_numpy(float)
        weights = numeric[weight_column].to_numpy(float)
        sqrt_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * sqrt_weights[:, None],
            y * sqrt_weights,
            rcond=None,
        )[0]
        fitted = design @ coefficients
        mean_y = np.sum(y * weights) / np.sum(weights)
        denominator = np.sum(weights * (y - mean_y) ** 2)
        r2 = (
            1.0 - np.sum(weights * (y - fitted) ** 2) / denominator
            if denominator > 0
            else np.nan
        )
        return data, coefficients, float(r2)

    rows = []
    for component, response in (
        ("trait_gap", "mean_trait_gap"),
        ("exact_match", "logit_exact_match_share"),
    ):
        weight_column = "area_weight_sum"
        required = [response, *background, "rao_q", weight_column]
        numeric = frame[required].apply(pd.to_numeric, errors="coerce")
        complete = np.isfinite(numeric).all(axis=1) & (numeric[weight_column] > 0)
        data = frame.loc[complete].copy()

        _, _, base_r2 = fit(data, response, background, weight_column)
        full_features = [*background, "rao_q"]
        fitted_data, coefficients, full_r2 = fit(
            data, response, full_features, weight_column
        )
        rao_index = 1 + full_features.index("rao_q")
        estimate = coefficients[rao_index] if coefficients.size else np.nan

        province_groups = [
            group.index.to_numpy()
            for _, group in fitted_data.groupby("province_key", observed=True)
        ]
        boot_estimate = np.full(N_BOOTSTRAPS, np.nan)
        boot_delta_r2 = np.full(N_BOOTSTRAPS, np.nan)
        for i in range(N_BOOTSTRAPS):
            if not province_groups:
                break
            sampled = rng.integers(0, len(province_groups), len(province_groups))
            indices = np.concatenate([province_groups[j] for j in sampled])
            sample = fitted_data.loc[indices]
            _, beta, sample_full_r2 = fit(
                sample, response, full_features, weight_column
            )
            _, _, sample_base_r2 = fit(
                sample, response, background, weight_column
            )
            if beta.size:
                boot_estimate[i] = beta[rao_index]
            boot_delta_r2[i] = sample_full_r2 - sample_base_r2

        finite_beta = boot_estimate[np.isfinite(boot_estimate)]
        finite_delta = boot_delta_r2[np.isfinite(boot_delta_r2)]
        beta_ci = (np.nan, np.nan)
        delta_ci = (np.nan, np.nan)
        sign_p = np.nan
        if len(finite_beta):
            beta_ci = tuple(np.quantile(finite_beta, [0.025, 0.975]))
            sign_p = min(
                1.0,
                2.0
                * min(
                    np.mean(finite_beta <= 0),
                    np.mean(finite_beta >= 0),
                ),
            )
        if len(finite_delta):
            delta_ci = tuple(np.quantile(finite_delta, [0.025, 0.975]))

        rows.append(
            {
                "alignment_component": component,
                "n_blocks": len(fitted_data),
                "n_provinces": fitted_data["province_key"].nunique(),
                "background_r2": base_r2,
                "full_r2": full_r2,
                "rao_q_added_r2": full_r2 - base_r2,
                "rao_q_added_r2_ci_low": delta_ci[0],
                "rao_q_added_r2_ci_high": delta_ci[1],
                "rao_q_estimate": estimate,
                "rao_q_ci_low": beta_ci[0],
                "rao_q_ci_high": beta_ci[1],
                "rao_q_bootstrap_sign_p": sign_p,
            }
        )

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    traits, metadata, distance, dimension_distances = load_traits_and_distances()
    model_species, mapping = load_model_species_mapping(traits)
    pixels = load_pixel_context()

    model_available, biological_available = build_availability_masks(
        model_species,
        mapping,
        traits,
        pixels["meow_province"],
    )
    pool_metrics, province_codes, substitution_gaps = calculate_pool_metrics(
        biological_available,
        traits,
        distance,
        dimension_distances,
        pixels["meow_province"],
    )
    pool_metrics.to_csv(OUT_DIR / "regional_functional_pool_metrics.csv", index=False)

    weight_n = pixels["w_no3_base"].to_numpy(float)
    weight_p = pixels["w_po4_base"].to_numpy(float)
    area_weights = pixels["area_weight"].to_numpy(float)

    selection_tables = []
    unconstrained_winners = {}
    pool_lookup = pool_metrics.set_index("province_key")
    province_keys = [normalize_key(value) for value in pixels["meow_province"]]
    pixel_pool_coverage = pool_lookup["pool_trait_coverage"].reindex(province_keys).to_numpy(float)
    pixel_resolved_richness = (
        pool_lookup["trait_resolved_richness"].reindex(province_keys).to_numpy(float)
    )
    pixels["pool_trait_coverage"] = pixel_pool_coverage
    pixels["trait_resolved_richness"] = pixel_resolved_richness
    trait_resolved = traits["trait_resolved"].to_numpy(bool)

    for scenario in SCENARIOS:
        predictions = combined_species_predictions(
            model_species,
            prediction_rows(pixels, scenario),
            scenario,
            weight_n,
            weight_p,
        )
        unconstrained_model, unconstrained_eligible = best_species(predictions)
        constrained_model, constrained_eligible = best_species(
            predictions, model_available
        )

        unconstrained_biological = np.where(
            unconstrained_model >= 0,
            mapping[np.maximum(unconstrained_model, 0)],
            -1,
        )
        constrained_biological = np.where(
            constrained_model >= 0,
            mapping[np.maximum(constrained_model, 0)],
            -1,
        )
        unconstrained_winners[scenario] = unconstrained_biological

        selection_tables.extend(
            [
                selection_advantage(
                    unconstrained_biological,
                    collapse_model_mask(
                        unconstrained_eligible, mapping, len(traits)
                    ),
                    area_weights,
                    traits,
                    scenario,
                    "unconstrained",
                ),
                selection_advantage(
                    constrained_biological,
                    collapse_model_mask(constrained_eligible, mapping, len(traits)),
                    area_weights,
                    traits,
                    scenario,
                    "constrained",
                ),
            ]
        )

        valid_winner_index = np.maximum(unconstrained_biological, 0)
        trait_gap_valid = (
            (unconstrained_biological >= 0)
            & trait_resolved[valid_winner_index]
            & (pixel_resolved_richness > 0)
        )
        gap = np.full(len(pixels), np.nan)
        gap[trait_gap_valid] = substitution_gaps["overall"][
            unconstrained_biological[trait_gap_valid],
            province_codes[trait_gap_valid],
        ]
        trait_gap_valid &= np.isfinite(gap)
        pixels[f"trait_substitution_gap_{scenario}"] = gap
        pixels[f"trait_gap_valid_{scenario}"] = trait_gap_valid
        pixels[f"exact_trait_match_{scenario}"] = trait_gap_valid & (gap <= EPS)

    selection_table = pd.concat(selection_tables, ignore_index=True)
    selection_table.to_csv(OUT_DIR / "species_selection_advantage.csv", index=False)

    specific_trait_models(selection_table, traits, metadata, rng).to_csv(
        OUT_DIR / "specific_trait_selection_models.csv", index=False
    )

    block_metrics = build_province_block_table(
        pixels,
        pool_metrics,
        province_codes,
        substitution_gaps,
        unconstrained_winners,
    )
    block_metrics.to_csv(OUT_DIR / "province_block_trait_metrics.csv", index=False)
    nested_constraint_models(block_metrics, rng).to_csv(
        OUT_DIR / "nested_constraint_loss_models.csv", index=False
    )
    functional_alignment_models(block_metrics, rng).to_csv(
        OUT_DIR / "functional_alignment_models.csv", index=False
    )
    pixels.to_csv(
        OUT_DIR / "trait_alignment_pixel_metrics.csv.gz",
        index=False,
        compression="gzip",
    )

    baseline_valid = pixel_pool_coverage >= PRIMARY_COVERAGE_GATE
    valid_area = np.isfinite(area_weights) & (area_weights > 0)
    summary = {
        "n_biological_species": len(traits),
        "n_trait_resolved_species": int(traits["trait_resolved"].sum()),
        "primary_pool_coverage_gate": PRIMARY_COVERAGE_GATE,
        "baseline_area_fraction_retained": float(
            np.sum(area_weights[baseline_valid]) / np.sum(area_weights[valid_area])
        ),
        "permutations": N_PERMUTATIONS,
        "province_cluster_bootstrap": N_BOOTSTRAPS,
    }
    (OUT_DIR / "trait_constraint_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
