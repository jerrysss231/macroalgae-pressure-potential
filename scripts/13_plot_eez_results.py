"""Plot EEZ trajectories, opportunity, persistent mismatch and economic context."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.plotting import panel_label, save_figure, set_publication_style

PATHS = ProjectPaths.from_env()
EEZ_DIR = PATHS.output_dir / "eez_analysis"
OUT_DIR = PATHS.output_dir / "manuscript_figures"

TRAJECTORY_ORDER = (
    "stable",
    "directionally improving",
    "directionally worsening",
    "scenario-sensitive",
    "high-forcing-sensitive",
)
FUTURE_SCENARIOS = ("ssp245", "ssp370", "ssp585")


def identify_mrgid_column(eez):
    for column in ("MRGID", "mrgid"):
        if column in eez.columns:
            return column
    raise ValueError("EEZ layer is missing MRGID.")


def main():
    set_publication_style()

    table = pd.read_csv(EEZ_DIR / "eez_portfolio.csv")
    eez = gpd.read_file(PATHS.eez_geopackage)
    id_column = identify_mrgid_column(eez)
    eez = eez.rename(columns={id_column: "MRGID"})
    eez = eez.merge(table, on="MRGID", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ax_map, ax_opportunity, ax_change, ax_gdp = axes.ravel()

    trajectory_codes = {
        trajectory: index for index, trajectory in enumerate(TRAJECTORY_ORDER)
    }
    eez["trajectory_code"] = eez["trajectory_class"].map(trajectory_codes)
    eez.plot(
        column="trajectory_code",
        ax=ax_map,
        categorical=True,
        legend=False,
        missing_kwds={"color": "0.9"},
    )
    ax_map.set_axis_off()
    ax_map.set_title("EEZ future trajectories")

    ax_opportunity.scatter(
        100 * table["opportunity_fraction_q75"],
        100 * table["persistent_mismatch_fraction"],
        s=12,
        alpha=0.7,
    )
    ax_opportunity.set_xlabel("Baseline biophysical opportunity (%)")
    ax_opportunity.set_ylabel("Persistent mismatch (%)")
    ax_opportunity.set_title("Persistent mismatch and opportunity")

    non_stable = table[table["trajectory_class"].ne("stable")].copy()
    y_positions = np.arange(len(non_stable))
    ax_change.axvline(0, color="0.5", linewidth=0.7)
    offsets = (-0.18, 0.0, 0.18)
    for offset, scenario in zip(offsets, FUTURE_SCENARIOS):
        ax_change.scatter(
            100 * non_stable[f"delta_{scenario}"],
            y_positions + offset,
            s=10,
            label=scenario,
        )
    ax_change.set_yticks(y_positions)
    ax_change.set_yticklabels(non_stable["GEONAME"], fontsize=6)
    ax_change.set_xlabel("Δ mismatch (percentage points)")
    ax_change.set_title("Non-stable EEZ trajectories")
    ax_change.legend(frameon=False)

    valid_gdp = (
        table["gdp_per_capita_ppp_2024"].notna()
        & table["persistent_mismatch_fraction"].notna()
    )
    ax_gdp.scatter(
        table.loc[valid_gdp, "gdp_per_capita_ppp_2024"],
        100 * table.loc[valid_gdp, "persistent_mismatch_fraction"],
        s=12,
        alpha=0.7,
    )
    ax_gdp.set_xscale("log")
    ax_gdp.set_xlabel("GDP per capita, PPP")
    ax_gdp.set_ylabel("Persistent mismatch (%)")
    ax_gdp.set_title("Economic context")

    for ax, label in zip(axes.ravel(), "ABCD"):
        panel_label(ax, label)

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "eez_pressure_potential_summary")


if __name__ == "__main__":
    main()
