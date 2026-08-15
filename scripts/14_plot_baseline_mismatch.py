"""Plot baseline Q75 pressure–potential states from finalized pixel products."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.plotting import (
    load_land,
    save_figure,
    set_publication_style,
    world_axes,
)

PATHS = ProjectPaths.from_env()
RESULT_DIR = PATHS.output_dir / "cross_scenario_comparison_and_mismatch_final"
OUT_DIR = PATHS.output_dir / "manuscript_figures"


def main():
    set_publication_style()

    pixels = pd.read_csv(
        RESULT_DIR / "recalculated_pixels" / "recalculated_baseline.csv"
    )
    thresholds = pd.read_csv(RESULT_DIR / "baseline_reference_thresholds.csv")
    q75 = thresholds[np.isclose(thresholds["quantile"], 0.75)]
    if len(q75) != 1:
        raise ValueError("Exactly one Q75 threshold row is required.")
    q75 = q75.iloc[0]

    high_pressure = (
        pixels["nutrient_pressure"].to_numpy(float)
        >= q75["nutrient_pressure_threshold"]
    )
    high_potential = (
        pixels["constrained_potential"].to_numpy(float)
        >= q75["constrained_potential_threshold"]
    )
    state = np.select(
        [
            high_pressure & ~high_potential,
            high_pressure & high_potential,
            ~high_pressure & high_potential,
        ],
        [1, 2, 3],
        default=0,
    )

    land = load_land(PATHS.land_shapefile)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    world_axes(ax, land)
    scatter = ax.scatter(
        pixels["lon"],
        pixels["lat"],
        c=state,
        s=0.35,
        vmin=0,
        vmax=3,
        rasterized=True,
    )
    ax.set_title("Baseline Q75 pressure–potential states")
    fig.colorbar(
        scatter,
        ax=ax,
        ticks=[0, 1, 2, 3],
        label="0 other · 1 mismatch · 2 opportunity · 3 high potential",
    )
    save_figure(fig, OUT_DIR / "baseline_pressure_potential_mismatch")


if __name__ == "__main__":
    main()
