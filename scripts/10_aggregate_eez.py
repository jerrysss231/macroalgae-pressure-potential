"""Aggregate global pressure–potential metrics to 200-nautical-mile EEZs."""

from __future__ import annotations

import re
import zlib

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import box

try:
    from shapely import make_valid
except ImportError:
    try:
        from shapely.validation import make_valid
    except ImportError:
        make_valid = None

from macroalgae_repro.paths import ProjectPaths


PATHS = ProjectPaths.from_env()
ROOT = PATHS.output_dir
PIXEL_DIR = ROOT / "cross_scenario_comparison_and_mismatch_final" / "recalculated_pixels"
THRESHOLD_CSV = (
    ROOT
    / "cross_scenario_comparison_and_mismatch_final"
    / "baseline_reference_thresholds.csv"
)
OUT_DIR = ROOT / "eez_analysis"

SCENARIOS = ("baseline", "ssp245", "ssp370", "ssp585")
FUTURE_SCENARIOS = SCENARIOS[1:]
SPATIAL_BLOCK_DEGREES = 5.0
MIN_VALID_PIXELS = 100
MIN_SPATIAL_BLOCKS = 5
N_PERMUTATIONS = 9_999
N_BOOTSTRAPS = 4_999
RANDOM_SEED = 42
DIRECTIONAL_EFFECT = 0.01
PIXEL_SIZE_DEGREES = 0.25
GEOD = Geod(ellps="WGS84")


GDP_VALUE_CANDIDATES = (
    "gdp_per_capita_ppp_constant_2021_intl_dollar",
    "sovereign_gdp_per_capita_ppp_constant_2021",
    "gdp_per_capita_ppp_2024",
    "2024",
)
GDP_YEAR_CANDIDATES = ("gdp_year", "sovereign_gdp_year", "year")


def read_csv(path):
    """Read a UTF-8 CSV and fail clearly when the input is absent."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    return frame


def require_columns(frame, columns, label):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


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


def spatial_block_id(longitude, latitude):
    longitude = np.asarray(longitude, float)
    latitude = np.asarray(latitude, float)
    block_x = np.floor(
        (longitude + 180.0) % 360.0 / SPATIAL_BLOCK_DEGREES
    ).astype(int)
    block_y = np.floor((latitude + 90.0) / SPATIAL_BLOCK_DEGREES).astype(int)
    return block_y * 10_000 + block_x


def stable_seed(value, scenario_index):
    """Derive deterministic EEZ-specific random seeds from the frozen global seed."""
    token = zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF
    sequence = np.random.SeedSequence([RANDOM_SEED, scenario_index, token])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def load_common_pixels():
    required = (
        "lon",
        "lat",
        "area_weight",
        "unconstrained_potential",
        "constrained_potential",
        "constraint_loss",
        "nutrient_pressure",
        "mismatch_q75",
        "valid_mismatch",
    )
    indexed = {}

    for scenario in SCENARIOS:
        path = PIXEL_DIR / f"recalculated_{scenario}.csv"
        frame = read_csv(path)
        require_columns(frame, required, path.name)

        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = ((frame["lon"] + 180.0) % 360.0) - 180.0
        for column in (
            "area_weight",
            "unconstrained_potential",
            "constrained_potential",
            "constraint_loss",
            "nutrient_pressure",
            "mismatch_q75",
            "valid_mismatch",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["lon_key"] = frame["lon"].round(6)
        frame["lat_key"] = frame["lat"].round(6)
        if frame.duplicated(["lon_key", "lat_key"]).any():
            raise ValueError(f"Duplicate coordinates in {path.name}")

        expected_weight = np.cos(np.deg2rad(frame["lat"].to_numpy(float)))
        if not np.allclose(
            frame["area_weight"],
            expected_weight,
            atol=1e-6,
            rtol=0,
            equal_nan=True,
        ):
            raise ValueError(f"{path.name}: area_weight differs from cos(latitude)")

        indexed[scenario] = frame.set_index(["lon_key", "lat_key"])

    common = indexed["baseline"].index
    for scenario in FUTURE_SCENARIOS:
        common = common.intersection(indexed[scenario].index, sort=False)
    common = pd.MultiIndex.from_frame(
        common.to_frame(index=False).sort_values(
            ["lat_key", "lon_key"], kind="mergesort"
        )
    )

    data = {
        scenario: indexed[scenario].reindex(common).reset_index()
        for scenario in SCENARIOS
    }
    valid_masks = [
        data[scenario]["valid_mismatch"].fillna(0).to_numpy(bool)
        for scenario in SCENARIOS
    ]
    for other in valid_masks[1:]:
        if not np.array_equal(valid_masks[0], other):
            raise ValueError("valid_mismatch must be identical across all scenarios")
    return data


def load_q75_thresholds():
    thresholds = read_csv(THRESHOLD_CSV)
    q75 = thresholds[np.isclose(thresholds["quantile"], 0.75)]
    if len(q75) != 1:
        raise ValueError("Exactly one Q75 threshold row is required")
    return (
        float(q75["nutrient_pressure_threshold"].iloc[0]),
        float(q75["constrained_potential_threshold"].iloc[0]),
    )


def identify_eez_columns(eez):
    def pick(candidates):
        for column in candidates:
            if column in eez.columns:
                return column
        raise ValueError(f"Missing EEZ field; expected one of {candidates}")

    return (
        pick(("MRGID", "mrgid")),
        pick(("GEONAME", "geoname", "TERRITORY1")),
    )


def find_iso_columns(columns, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}\d*$", re.IGNORECASE)
    return sorted(column for column in columns if pattern.match(str(column)))


def split_iso_values(values):
    found = []
    for value in values:
        if pd.isna(value):
            continue
        for part in re.split(r"[;,|/]+", str(value).strip().upper()):
            if re.fullmatch(r"[A-Z]{3}", part) and part not in found:
                found.append(part)
    return found


def load_eez_layer():
    """Load EEZ v12 and retain sovereign metadata without collapsing disputes."""
    eez = gpd.read_file(PATHS.eez_geopackage)
    if eez.crs is None:
        raise ValueError("EEZ layer has no CRS")
    eez = eez.to_crs("EPSG:4326")

    id_column, name_column = identify_eez_columns(eez)
    eez = eez.rename(columns={id_column: "MRGID", name_column: "GEONAME"})
    require_columns(eez, ("MRGID", "GEONAME", "POL_TYPE"), "EEZ layer")
    if eez["MRGID"].duplicated().any():
        raise ValueError("MRGID is not unique in the EEZ layer")

    invalid = ~eez.geometry.is_valid
    if invalid.any():
        if make_valid is None:
            raise RuntimeError(
                "EEZ layer contains invalid geometries and make_valid is unavailable"
            )
        eez.loc[invalid, "geometry"] = eez.loc[invalid, "geometry"].map(make_valid)
        if (~eez.geometry.is_valid).any():
            raise RuntimeError("Some EEZ geometries remain invalid after make_valid")

    sovereign_columns = find_iso_columns(eez.columns, "ISO_SOV")
    territory_columns = find_iso_columns(eez.columns, "ISO_TER")
    if not sovereign_columns:
        raise ValueError("EEZ layer contains no ISO_SOV* fields")

    eez["sovereign_iso3_list"] = eez[sovereign_columns].apply(
        lambda row: split_iso_values(row.values), axis=1
    )
    eez["territory_iso3_list"] = (
        eez[territory_columns].apply(
            lambda row: split_iso_values(row.values), axis=1
        )
        if territory_columns
        else [[] for _ in range(len(eez))]
    )
    eez["sovereign_iso3_all"] = eez["sovereign_iso3_list"].map(";".join)
    eez["territory_iso3_all"] = eez["territory_iso3_list"].map(";".join)
    eez["sovereign_iso3_count"] = eez["sovereign_iso3_list"].map(len)
    eez["sovereign_iso3"] = eez["sovereign_iso3_list"].map(
        lambda values: values[0] if len(values) == 1 else pd.NA
    )
    eez["included_in_ecological_analysis"] = (
        eez["POL_TYPE"].astype(str).str.strip().eq("200NM")
    )
    return eez


def geodesic_area(geometry):
    if geometry is None or geometry.is_empty:
        return 0.0
    if geometry.geom_type == "GeometryCollection":
        return float(sum(geodesic_area(part) for part in geometry.geoms))
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        return 0.0
    return abs(float(GEOD.geometry_area_perimeter(geometry)[0]))


def build_pixel_eez_overlap(baseline, eez):
    """Intersect model cells with ordinary 200NM EEZs without jurisdictional renormalization."""
    eez_200 = eez.loc[
        eez["included_in_ecological_analysis"],
        ["MRGID", "GEONAME", "POL_TYPE", "geometry"],
    ].copy()
    eez_200 = eez_200[eez_200.geometry.notna()].copy()

    half_pixel = PIXEL_SIZE_DEGREES / 2.0
    cells = baseline[["lon", "lat", "area_weight"]].copy()
    cells["pixel_id"] = np.arange(len(cells), dtype=int)
    cells = gpd.GeoDataFrame(
        cells,
        geometry=[
            box(
                longitude - half_pixel,
                latitude - half_pixel,
                longitude + half_pixel,
                latitude + half_pixel,
            )
            for longitude, latitude in zip(cells["lon"], cells["lat"])
        ],
        crs="EPSG:4326",
    )
    cells["pixel_area_m2"] = np.fromiter(
        (geodesic_area(geometry) for geometry in cells.geometry),
        dtype=float,
        count=len(cells),
    )
    if (cells["pixel_area_m2"] <= 0).any():
        raise RuntimeError("Non-positive geodesic pixel area detected")

    overlap = gpd.overlay(cells, eez_200, how="intersection", keep_geom_type=False)
    overlap["overlap_area_m2"] = np.fromiter(
        (geodesic_area(geometry) for geometry in overlap.geometry),
        dtype=float,
        count=len(overlap),
    )
    overlap = overlap[overlap["overlap_area_m2"] > 0].copy()
    overlap["overlap_fraction"] = (
        overlap["overlap_area_m2"] / overlap["pixel_area_m2"]
    )
    if len(overlap) and float(overlap["overlap_fraction"].max()) > 1.001:
        raise RuntimeError("Pixel–EEZ overlap fraction materially exceeds one")
    overlap["overlap_fraction"] = overlap["overlap_fraction"].clip(0, 1)

    overlap["ecological_weight"] = (
        overlap["area_weight"] * overlap["overlap_fraction"]
    )
    overlap["overlap_area_km2"] = overlap["overlap_area_m2"] / 1e6
    overlap["block_id"] = spatial_block_id(overlap["lon"], overlap["lat"])

    # Overlapping EEZs are intentionally independent; never renormalize these weights.
    return pd.DataFrame(
        overlap[
            [
                "pixel_id",
                "MRGID",
                "GEONAME",
                "POL_TYPE",
                "lon",
                "lat",
                "overlap_fraction",
                "overlap_area_km2",
                "ecological_weight",
                "block_id",
            ]
        ]
    )


def aggregate_scenario_metrics(data, overlap, pressure_threshold, potential_threshold):
    rows = []
    grouped_overlap = overlap.groupby(["MRGID", "GEONAME"], observed=True)

    for scenario in SCENARIOS:
        frame = data[scenario]
        pressure = frame["nutrient_pressure"].to_numpy(float)
        potential = frame["constrained_potential"].to_numpy(float)
        loss = frame["constraint_loss"].to_numpy(float)
        mismatch = frame["mismatch_q75"].to_numpy(bool)
        valid_mismatch = frame["valid_mismatch"].fillna(0).to_numpy(bool)
        opportunity = (
            (pressure >= pressure_threshold)
            & (potential >= potential_threshold)
            & valid_mismatch
        )

        for (mrgid, name), group in grouped_overlap:
            pixel_ids = group["pixel_id"].to_numpy(int)
            weights = group["ecological_weight"].to_numpy(float)
            valid = valid_mismatch[pixel_ids]
            block_ids = group["block_id"].to_numpy(int)

            rows.append(
                {
                    "MRGID": mrgid,
                    "GEONAME": name,
                    "scenario": scenario,
                    "n_pixels": int(np.unique(pixel_ids[valid]).size),
                    "n_blocks": int(np.unique(block_ids[valid]).size),
                    "mean_nutrient_pressure": weighted_mean(
                        pressure[pixel_ids][valid], weights[valid]
                    ),
                    "mean_constrained_potential": weighted_mean(
                        potential[pixel_ids][valid], weights[valid]
                    ),
                    "mean_constraint_loss": weighted_mean(
                        loss[pixel_ids][valid], weights[valid]
                    ),
                    "mismatch_fraction_q75": weighted_fraction(
                        mismatch[pixel_ids], valid, weights
                    ),
                    "opportunity_fraction_q75": weighted_fraction(
                        opportunity[pixel_ids], valid, weights
                    ),
                    "valid_weight": float(np.sum(weights[valid])),
                }
            )

    return pd.DataFrame(rows)


def aggregate_cross_scenario_states(data, overlap):
    """Summarize persistent, emerging and recovered Q75 mismatch on one common domain."""
    valid = np.logical_and.reduce(
        [
            data[scenario]["valid_mismatch"].fillna(0).to_numpy(bool)
            for scenario in SCENARIOS
        ]
    )
    mismatch = {
        scenario: data[scenario]["mismatch_q75"].to_numpy(bool)
        for scenario in SCENARIOS
    }
    persistent = valid.copy()
    for scenario in SCENARIOS:
        persistent &= mismatch[scenario]
    emerging = {
        scenario: valid & ~mismatch["baseline"] & mismatch[scenario]
        for scenario in FUTURE_SCENARIOS
    }
    recovery = {
        scenario: valid & mismatch["baseline"] & ~mismatch[scenario]
        for scenario in FUTURE_SCENARIOS
    }

    rows = []
    for (mrgid, name), group in overlap.groupby(
        ["MRGID", "GEONAME"], observed=True
    ):
        pixel_ids = group["pixel_id"].to_numpy(int)
        weights = group["ecological_weight"].to_numpy(float)
        local_valid = valid[pixel_ids]
        row = {
            "MRGID": mrgid,
            "GEONAME": name,
            "persistent_bottleneck_fraction_q75": weighted_fraction(
                persistent[pixel_ids], local_valid, weights
            ),
        }
        for scenario in FUTURE_SCENARIOS:
            row[f"emerging_fraction_{scenario}"] = weighted_fraction(
                emerging[scenario][pixel_ids], local_valid, weights
            )
            row[f"recovery_fraction_{scenario}"] = weighted_fraction(
                recovery[scenario][pixel_ids], local_valid, weights
            )
        rows.append(row)
    return pd.DataFrame(rows)


def classify_direction(delta):
    if not np.isfinite(delta):
        return np.nan
    if delta >= DIRECTIONAL_EFFECT:
        return 1.0
    if delta <= -DIRECTIONAL_EFFECT:
        return -1.0
    return 0.0


def classify_trajectory(delta):
    directions = np.array([classify_direction(value) for value in delta], float)
    if not np.isfinite(directions).all():
        return "no valid data"
    if np.all(directions == 0):
        return "stable"
    if np.array_equal(directions, np.array([0.0, 0.0, 1.0])):
        return "high-forcing-sensitive"
    if (directions == 1).any() and (directions == -1).any():
        return "scenario-sensitive"
    if (directions == 1).any():
        return "directionally worsening"
    if (directions == -1).any():
        return "directionally improving"
    return "stable"


def eez_inference(data, future_scenario, overlap_group, seed):
    """Test future-minus-baseline mismatch change using independent 5° blocks."""
    pixel_ids = overlap_group["pixel_id"].to_numpy(int)
    block_ids = overlap_group["block_id"].to_numpy(int)
    weights = overlap_group["ecological_weight"].to_numpy(float)

    common_valid = np.logical_and.reduce(
        [
            data[scenario]["valid_mismatch"].fillna(0).to_numpy(bool)[pixel_ids]
            for scenario in SCENARIOS
        ]
    )
    pixel_ids = pixel_ids[common_valid]
    block_ids = block_ids[common_valid]
    weights = weights[common_valid]

    baseline_state = data["baseline"]["mismatch_q75"].to_numpy(float)[pixel_ids]
    future_state = data[future_scenario]["mismatch_q75"].to_numpy(float)[pixel_ids]
    delta = future_state - baseline_state

    unique_blocks = np.unique(block_ids)
    if not len(unique_blocks):
        return np.nan, np.nan, np.nan, np.nan, 0, 0, "no_valid_data"

    block_numerator = np.array(
        [
            np.sum(weights[block_ids == block] * delta[block_ids == block])
            for block in unique_blocks
        ],
        dtype=float,
    )
    block_weight = np.array(
        [np.sum(weights[block_ids == block]) for block in unique_blocks],
        dtype=float,
    )
    observed = float(np.sum(block_numerator) / np.sum(block_weight))
    n_pixels = int(np.unique(pixel_ids).size)
    n_blocks = int(len(unique_blocks))

    if n_pixels < MIN_VALID_PIXELS or n_blocks < MIN_SPATIAL_BLOCKS:
        return (
            observed,
            np.nan,
            np.nan,
            np.nan,
            n_pixels,
            n_blocks,
            "insufficient_spatial_support",
        )

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        signs = rng.choice((-1.0, 1.0), n_blocks)
        permuted = float(np.sum(signs * block_numerator) / np.sum(block_weight))
        extreme += abs(permuted) >= abs(observed)
    p_value = (extreme + 1) / (N_PERMUTATIONS + 1)

    bootstrap = np.empty(N_BOOTSTRAPS)
    for index in range(N_BOOTSTRAPS):
        sampled = rng.integers(0, n_blocks, n_blocks)
        bootstrap[index] = (
            np.sum(block_numerator[sampled]) / np.sum(block_weight[sampled])
        )
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    return (
        observed,
        float(p_value),
        float(ci_low),
        float(ci_high),
        n_pixels,
        n_blocks,
        "ok",
    )


def build_cross_scenario_metrics(data, overlap, scenario_metrics, state_metrics):
    mismatch_table = scenario_metrics.pivot(
        index=["MRGID", "GEONAME"],
        columns="scenario",
        values="mismatch_fraction_q75",
    ).reset_index()
    overlap_groups = {
        key: group
        for key, group in overlap.groupby(["MRGID", "GEONAME"], observed=True)
    }

    rows = []
    for record in mismatch_table.itertuples(index=False):
        delta = np.array(
            [
                getattr(record, scenario) - record.baseline
                for scenario in FUTURE_SCENARIOS
            ],
            dtype=float,
        )
        row = {
            "MRGID": record.MRGID,
            "GEONAME": record.GEONAME,
            "trajectory_class": classify_trajectory(delta),
        }
        overlap_group = overlap_groups[(record.MRGID, record.GEONAME)]

        for scenario_index, (scenario, effect) in enumerate(
            zip(FUTURE_SCENARIOS, delta), start=1
        ):
            stats = eez_inference(
                data,
                scenario,
                overlap_group,
                stable_seed(record.MRGID, scenario_index),
            )
            observed, p_value, ci_low, ci_high, n_pixels, n_blocks, status = stats
            row[f"delta_{scenario}"] = effect
            row[f"delta_mismatch_pp_{scenario}"] = 100.0 * effect
            row[f"inference_delta_{scenario}"] = observed
            row[f"p_{scenario}"] = p_value
            row[f"ci_low_{scenario}"] = ci_low
            row[f"ci_high_{scenario}"] = ci_high
            row[f"n_valid_pixels_{scenario}"] = n_pixels
            row[f"n_spatial_blocks_{scenario}"] = n_blocks
            row[f"inference_status_{scenario}"] = status
        rows.append(row)

    output = pd.DataFrame(rows)
    for scenario in FUTURE_SCENARIOS:
        output[f"p_fdr_{scenario}"] = benjamini_hochberg(output[f"p_{scenario}"])
        direction = np.array(
            [classify_direction(value) for value in output[f"delta_{scenario}"]],
            dtype=float,
        )
        ci_support = (
            ((direction < 0) & (output[f"ci_high_{scenario}"] < 0))
            | ((direction > 0) & (output[f"ci_low_{scenario}"] > 0))
        )
        output[f"formally_supported_{scenario}"] = (
            np.isfinite(direction)
            & (direction != 0)
            & (output[f"p_fdr_{scenario}"] < 0.05)
            & ci_support
        )

    output = output.merge(
        state_metrics,
        on=["MRGID", "GEONAME"],
        how="left",
        validate="one_to_one",
    )
    output["persistent_mismatch_fraction"] = output[
        "persistent_bottleneck_fraction_q75"
    ]
    return output


def choose_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def add_gdp_context(portfolio, eez):
    """Attach GDP only where an EEZ resolves to one sovereign ISO3 code."""
    attributes = eez[
        [
            "MRGID",
            "territory_iso3_all",
            "sovereign_iso3_all",
            "sovereign_iso3_count",
            "sovereign_iso3",
            "included_in_ecological_analysis",
        ]
    ].copy()
    output = portfolio.merge(attributes, on="MRGID", how="left", validate="one_to_one")

    if not PATHS.gdp_2024_csv.exists():
        output["gdp_join_status"] = "gdp_file_not_available"
        output["economic_context_eligible"] = False
        output["gdp_per_capita_ppp_2024"] = np.nan
        return output

    gdp = read_csv(PATHS.gdp_2024_csv)
    iso_column = choose_column(gdp, ("iso3", "sovereign_iso3", "Country Code"))
    value_column = choose_column(gdp, GDP_VALUE_CANDIDATES)
    year_column = choose_column(gdp, GDP_YEAR_CANDIDATES)
    if iso_column is None or value_column is None:
        raise ValueError("GDP table requires an ISO3 column and the 2024 PPP indicator")

    gdp = gdp.copy()
    gdp[iso_column] = gdp[iso_column].astype(str).str.strip().str.upper()
    if gdp[iso_column].duplicated().any():
        raise ValueError("GDP table contains duplicate ISO3 codes")
    gdp[value_column] = pd.to_numeric(gdp[value_column], errors="coerce")
    keep = [iso_column, value_column]
    if year_column is not None:
        gdp[year_column] = pd.to_numeric(gdp[year_column], errors="coerce")
        keep.append(year_column)
    for optional in ("country_or_economy", "region", "income_group"):
        if optional in gdp.columns:
            keep.append(optional)

    rename = {
        iso_column: "sovereign_iso3",
        value_column: "sovereign_gdp_per_capita_ppp_constant_2021",
    }
    if year_column is not None:
        rename[year_column] = "sovereign_gdp_year"
    rename.update(
        {
            "country_or_economy": "sovereign_economic_unit",
            "region": "world_bank_region",
        }
    )
    gdp = gdp[keep].rename(columns=rename)
    if "sovereign_gdp_year" not in gdp.columns:
        gdp["sovereign_gdp_year"] = 2024
    output = output.merge(gdp, on="sovereign_iso3", how="left", validate="many_to_one")

    matched = output["sovereign_gdp_per_capita_ppp_constant_2021"].notna()
    output["economic_context_eligible"] = (
        output["included_in_ecological_analysis"].fillna(False)
        & output["sovereign_iso3_count"].eq(1)
        & matched
    )
    output["gdp_join_status"] = np.select(
        [
            ~output["included_in_ecological_analysis"].fillna(False),
            ~output["sovereign_iso3_count"].eq(1),
            matched,
        ],
        [
            "excluded_non_200NM",
            "missing_or_multiple_sovereign_iso3",
            "matched_sovereign_iso3",
        ],
        default="sovereign_gdp_missing",
    )
    output["gdp_per_capita_ppp_2024"] = output[
        "sovereign_gdp_per_capita_ppp_constant_2021"
    ]
    return output


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_common_pixels()
    pressure_threshold, potential_threshold = load_q75_thresholds()
    eez = load_eez_layer()
    overlap = build_pixel_eez_overlap(data["baseline"], eez)
    overlap.to_csv(
        OUT_DIR / "pixel_eez_overlap.csv.gz",
        index=False,
        compression="gzip",
    )

    scenario_metrics = aggregate_scenario_metrics(
        data,
        overlap,
        pressure_threshold,
        potential_threshold,
    )
    scenario_metrics.to_csv(OUT_DIR / "eez_scenario_metrics.csv", index=False)

    state_metrics = aggregate_cross_scenario_states(data, overlap)
    cross_scenario = build_cross_scenario_metrics(
        data,
        overlap,
        scenario_metrics,
        state_metrics,
    )
    cross_scenario.to_csv(OUT_DIR / "eez_cross_scenario_metrics.csv", index=False)

    baseline = scenario_metrics[scenario_metrics["scenario"].eq("baseline")][
        [
            "MRGID",
            "GEONAME",
            "mean_nutrient_pressure",
            "mean_constrained_potential",
            "mean_constraint_loss",
            "opportunity_fraction_q75",
            "mismatch_fraction_q75",
        ]
    ].rename(
        columns={
            "mean_nutrient_pressure": "baseline_mean_nutrient_pressure",
            "mean_constrained_potential": "baseline_mean_constrained_potential",
            "mean_constraint_loss": "baseline_mean_constraint_loss",
            "mismatch_fraction_q75": "baseline_mismatch_fraction_q75",
        }
    )
    portfolio = cross_scenario.merge(
        baseline,
        on=["MRGID", "GEONAME"],
        how="left",
        validate="one_to_one",
    )
    portfolio = add_gdp_context(portfolio, eez)
    portfolio.to_csv(OUT_DIR / "eez_portfolio.csv", index=False)

    print("Saved EEZ outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
