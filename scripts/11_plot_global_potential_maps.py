"""Plot constrained potential and biogeographic constraint loss by scenario."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.plotting import (
    load_land,
    panel_label,
    save_figure,
    set_publication_style,
    world_axes,
)

PATHS = ProjectPaths.from_env()
PIXEL_DIR = (
    PATHS.output_dir
    / "cross_scenario_comparison_and_mismatch_final"
    / "recalculated_pixels"
)
OUT_DIR = PATHS.output_dir / "manuscript_figures"

SCENARIOS = ("baseline", "ssp245", "ssp370", "ssp585")
SCENARIO_LABELS = ("Baseline", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5")


def read_pixels(scenario):
    return pd.read_csv(
        PIXEL_DIR / f"recalculated_{scenario}.csv",
        encoding="utf-8-sig",
    )


def robust_upper(frames, column, quantile=0.98):
    values = np.concatenate(
        [frame[column].to_numpy(float) for frame in frames]
    )
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"No finite values available for {column}.")
    return float(np.quantile(values, quantile))


def main():
    set_publication_style()
    land = load_land(PATHS.land_shapefile)
    frames = [read_pixels(scenario) for scenario in SCENARIOS]

    potential_max = robust_upper(frames, "constrained_potential")
    loss_max = robust_upper(frames, "constraint_loss")
    metrics = (
        ("constrained_potential", potential_max, "Constrained potential"),
        ("constraint_loss", loss_max, "Biogeographic constraint loss"),
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.2, 4.0),
        sharex=True,
        sharey=True,
    )

    for column_index, (frame, scenario_label) in enumerate(
        zip(frames, SCENARIO_LABELS)
    ):
        for row_index, (column, vmax, metric_label) in enumerate(metrics):
            ax = axes[row_index, column_index]
            world_axes(ax, land)
            scatter = ax.scatter(
                frame["lon"],
                frame["lat"],
                c=frame[column],
                s=0.25,
                vmin=0,
                vmax=vmax,
                rasterized=True,
                zorder=0,
            )

            if row_index == 0:
                ax.set_title(scenario_label)
            if column_index == 0:
                ax.set_ylabel(metric_label)
            else:
                ax.set_ylabel("")
            if row_index == 0:
                ax.set_xlabel("")

            panel = chr(ord("A") + row_index * 4 + column_index)
            panel_label(ax, panel)
            fig.colorbar(
                scatter,
                ax=ax,
                orientation="horizontal",
                fraction=0.05,
                pad=0.08,
            )

    fig.subplots_adjust(wspace=0.12, hspace=0.18)
    save_figure(fig, OUT_DIR / "global_constrained_potential_and_loss")


if __name__ == "__main__":
    main()
