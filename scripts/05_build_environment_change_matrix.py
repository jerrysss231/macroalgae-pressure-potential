"""Assemble harmonized absolute and future-minus-baseline environmental fields."""

from __future__ import annotations

import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths, SCENARIOS
from macroalgae_repro.spatial import clean_values, open_2d


PATHS = ProjectPaths.from_env()
OUTPUT_DIR = PATHS.output_dir
ATTR_DIR = OUTPUT_DIR / "future_environmental_attribution"
OUT_CSV = ATTR_DIR / "driver_delta_matrix.csv"
FUTURES = ("ssp245", "ssp370", "ssp585")
DRIVERS = ("Temperature", "NO3", "PO4", "pH", "Salinity")
DEPTH_LIMIT_M = 50.0
PHOTOPERIOD_H = 12.0


def load_scenario(scenario: str, template, light, depth) -> pd.DataFrame:
    no3 = clean_values(open_2d(PATHS.environment(scenario, "no3"), ("no3", "nitrate"), template=template).values, 0)
    po4 = clean_values(open_2d(PATHS.environment(scenario, "po4"), ("po4", "phosphate"), template=template).values, 0)
    ph = clean_values(open_2d(PATHS.environment(scenario, "ph"), ("ph",), template=template).values, 6.0, 9.5)
    salinity = clean_values(open_2d(PATHS.environment(scenario, "salinity"), ("so", "salinity"), template=template).values, 0.0, 45.0)
    temperature = clean_values(open_2d(PATHS.environment(scenario, "temperature"), ("thetao", "temperature", "temp"), template=template).values, -3.0, 45.0)
    light_values = clean_values(light.values, 0)
    valid = (
        np.isfinite(no3)
        & np.isfinite(po4)
        & np.isfinite(ph)
        & np.isfinite(salinity)
        & np.isfinite(temperature)
        & np.isfinite(light_values)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth <= DEPTH_LIMIT_M)
    )
    lat2, lon2 = np.meshgrid(template["lat"].values, template["lon"].values, indexing="ij")
    return pd.DataFrame(
        {
            "lon": lon2[valid],
            "lat": lat2[valid],
            "Temperature": temperature[valid],
            "NO3": no3[valid],
            "PO4": po4[valid],
            "pH": ph[valid],
            "Salinity": salinity[valid],
        }
    )


def main() -> None:
    ATTR_DIR.mkdir(parents=True, exist_ok=True)
    template = open_2d(PATHS.environment("baseline", "no3"), ("no3", "nitrate"))
    par = open_2d(PATHS.par_baseline, ("par", "photosynt", "radiation"), template=template)
    light = (par * (1_000_000 / (PHOTOPERIOD_H * 3600))).astype("float32")
    terrain = open_2d(PATHS.terrain, ("bathymetry", "bathy", "depth", "terrain"), template=template)
    values = terrain.values.astype("float32")
    depth = (-values if np.nanmedian(values) < 0 else values).astype("float32")

    frames = {}
    indexes = {}
    for scenario in SCENARIOS:
        frame = load_scenario(scenario, template, light, depth)
        frame["_lon"] = frame["lon"].round(6)
        frame["_lat"] = frame["lat"].round(6)
        if frame.duplicated(["_lon", "_lat"]).any():
            raise ValueError(f"Duplicate coordinates in {scenario}")
        frames[scenario] = frame
        indexes[scenario] = pd.MultiIndex.from_frame(frame[["_lon", "_lat"]])

    common = indexes["baseline"]
    for scenario in FUTURES:
        common = common.intersection(indexes[scenario], sort=False)
    common = pd.MultiIndex.from_frame(common.to_frame(index=False).sort_values(["_lon", "_lat"]))

    aligned = {
        scenario: frames[scenario].set_index(["_lon", "_lat"]).reindex(common)
        for scenario in SCENARIOS
    }
    out = pd.DataFrame({"lon": aligned["baseline"]["lon"], "lat": aligned["baseline"]["lat"]}).reset_index(drop=True)
    for scenario in SCENARIOS:
        for driver in DRIVERS:
            out[f"{driver}_{scenario}"] = aligned[scenario][driver].to_numpy(float)
    for scenario in FUTURES:
        for driver in DRIVERS:
            out[f"delta_{driver}_{scenario}"] = out[f"{driver}_{scenario}"] - out[f"{driver}_baseline"]

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {OUT_CSV} ({len(out):,} common pixels)")


if __name__ == "__main__":
    main()
