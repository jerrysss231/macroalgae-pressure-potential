"""Prepare directly documented AlgaeTraits states and functional distances."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths


PATHS = ProjectPaths.from_env()
OUT_DIR = PATHS.output_dir / "traits"
INPUT_CSV = PATHS.algaetraits_traits_wide
MIN_SHARED_DIMENSIONS = 2

TRAITS = {
    67: ("environmental_position", "habitat"),
    72: ("life_span", "life_history"),
    73: ("body_shape", "morphology"),
    76: ("cytomorphology", "morphology"),
    78: ("gametophyte_arrangement", "reproduction"),
    79: ("asexual_reproduction", "reproduction"),
    82: ("gamete_type", "reproduction"),
    84: ("life_cycle", "reproduction"),
    85: ("spawning", "reproduction"),
}
STATE_PATTERNS = {
    67: [("epilithic", r"\bepilithic\b"), ("epiphytic", r"\bepiphytic\b"), ("epizoic", r"\bepizoic\b"), ("unattached", r"\bunattached\b")],
    72: [("annual", r"\bannual\b"), ("ephemeral", r"\bephemeral\b"), ("perennial", r"\bperennial\b")],
    73: [(state, rf"\b{state}\b") for state in ("erect", "branched", "tubular", "foliose", "filamentous")],
    76: [("siphonous", r"\bsiphonous\b"), ("non_unicellular", r"non-unicellular")],
    78: [("dioecious", r"\bdioicous\b|\bdioecious\b"), ("monoecious", r"\bmonoicous\b|\bmonoecious\b"), ("mixed", r"\bmixed\b")],
    79: [("yes", r"\byes\b"), ("no", r"\bno\b")],
    82: [("oogamous", r"\boogamous\b"), ("anisogamous", r"\banisogamous\b"), ("isogamous", r"\bisogamous\b")],
    84: [("haplodiplontic", r"\bhaplodiplontic\b"), ("diplontic", r"\bdiplontic\b"), ("isomorphic", r"\bisomorphic\b"), ("heteromorphic", r"\bheteromorphic\b")],
    85: [("water_column", r"fertilisation in the water column"), ("female_gametophyte", r"fertilisation on female gametophyte")],
}


def read_csv(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def trait_columns(frame: pd.DataFrame) -> dict[int, str]:
    output = {}
    for column in frame.columns:
        match = re.match(r"^(\d+)__", str(column))
        if match:
            output[int(match.group(1))] = column
    return output


def normalized_text(value) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip().lower()


def canonical_states(value, attribute_id: int):
    text = re.sub(r"\s*\[[^\]]*\]", "", normalized_text(value))
    if not text:
        return np.nan
    states = [name for name, pattern in STATE_PATTERNS[attribute_id] if re.search(pattern, text)]
    if not states:
        states = sorted({re.sub(r"\s+", " ", part).strip() for part in text.split("||") if part.strip()})
    return ";".join(sorted(set(states))) if states else np.nan


def clean_traits(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = trait_columns(raw)
    missing = [attribute for attribute in TRAITS if attribute not in columns]
    if missing:
        raise ValueError(f"AlgaeTraits table is missing primary attributes: {missing}")
    required = ["species_id", "Scientific name", "meow_match_name", "Phylum"]
    absent = [column for column in required if column not in raw]
    if absent:
        raise ValueError(f"AlgaeTraits table is missing species columns: {absent}")

    clean = raw[required].copy()
    clean["species_id"] = pd.to_numeric(clean["species_id"], errors="raise").astype(int)
    metadata = []
    for attribute, (trait, dimension) in TRAITS.items():
        clean[trait] = raw[columns[attribute]].map(lambda value, aid=attribute: canonical_states(value, aid))
        metadata.append({"attribute_id": attribute, "raw_column": columns[attribute], "clean_trait": trait, "functional_dimension": dimension, "trait_type": "multistate", "coverage": float(clean[trait].notna().mean())})
    return clean, pd.DataFrame(metadata)


def jaccard_dissimilarity(first, second) -> float:
    if pd.isna(first) or pd.isna(second):
        return np.nan
    a = {x for x in str(first).split(";") if x}
    b = {x for x in str(second).split(";") if x}
    if not a or not b:
        return np.nan
    return 1.0 - len(a & b) / len(a | b)


def functional_distance(clean: pd.DataFrame, metadata: pd.DataFrame):
    names = clean["meow_match_name"].astype(str).tolist()
    dimensions = metadata.groupby("functional_dimension")["clean_trait"].apply(list).to_dict()
    distance = np.full((len(clean), len(clean)), np.nan, float)
    shared = np.zeros((len(clean), len(clean)), int)
    dimension_matrices = {dimension: np.full_like(distance, np.nan) for dimension in dimensions}

    for i in range(len(clean)):
        distance[i, i] = 0.0
        for matrix in dimension_matrices.values():
            matrix[i, i] = 0.0
        shared[i, i] = len(dimensions)
        for j in range(i + 1, len(clean)):
            dim_values = []
            for dimension, traits in dimensions.items():
                values = [jaccard_dissimilarity(clean.at[i, trait], clean.at[j, trait]) for trait in traits]
                values = [value for value in values if np.isfinite(value)]
                if values:
                    value = float(np.mean(values))
                    dimension_matrices[dimension][i, j] = dimension_matrices[dimension][j, i] = value
                    dim_values.append(value)
            shared[i, j] = shared[j, i] = len(dim_values)
            if len(dim_values) >= MIN_SHARED_DIMENSIONS:
                distance[i, j] = distance[j, i] = float(np.mean(dim_values))

    return names, distance, shared, dimension_matrices


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_csv(INPUT_CSV)
    clean, metadata = clean_traits(raw)
    names, distance, shared, dimension_matrices = functional_distance(clean, metadata)

    clean.to_csv(OUT_DIR / "direct_traits_cleaned.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(OUT_DIR / "trait_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(distance, index=names, columns=names).to_csv(OUT_DIR / "functional_distance.csv")
    pd.DataFrame(shared, index=names, columns=names).to_csv(OUT_DIR / "shared_functional_dimensions.csv")
    for dimension, matrix in dimension_matrices.items():
        pd.DataFrame(matrix, index=names, columns=names).to_csv(OUT_DIR / f"functional_distance_{dimension}.csv")

    complete = clean[[name for name, _ in TRAITS.values()]].notna().all(axis=1)
    summary = {"n_species": int(len(clean)), "n_complete_primary_trait_species": int(complete.sum()), "primary_traits": [name for name, _ in TRAITS.values()], "functional_dimensions": sorted(metadata["functional_dimension"].unique().tolist()), "minimum_shared_dimensions": MIN_SHARED_DIMENSIONS, "distance_rule": "mean trait dissimilarity within dimension, then mean across shared dimensions", "missing_trait_imputation": False}
    (OUT_DIR / "trait_preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved trait preparation outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
