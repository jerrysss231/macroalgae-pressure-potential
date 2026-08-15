"""Plot trait-selection and functional-substitution summaries."""

import matplotlib.pyplot as plt
import pandas as pd

from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.plotting import panel_label, save_figure, set_publication_style

PATHS = ProjectPaths.from_env()
TRAIT_DIR = PATHS.output_dir / "trait_constraint_analysis"
OUT_DIR = PATHS.output_dir / "manuscript_figures"


def main():
    set_publication_style()

    trait_models = pd.read_csv(TRAIT_DIR / "specific_trait_selection_models.csv")
    nested_models = pd.read_csv(TRAIT_DIR / "nested_constraint_loss_models.csv")
    block_metrics = pd.read_csv(TRAIT_DIR / "province_block_trait_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ax_trait, ax_nested, ax_gap, ax_match = axes.ravel()

    constrained = trait_models[trait_models["selection_type"].eq("constrained")]
    trait_matrix = constrained.pivot_table(
        index="trait",
        columns="scenario",
        values="trait_added_r2",
        aggfunc="mean",
    )
    image = ax_trait.imshow(trait_matrix.to_numpy(), aspect="auto")
    ax_trait.set_yticks(range(len(trait_matrix)))
    ax_trait.set_yticklabels(
        [trait.replace("_", " ") for trait in trait_matrix.index]
    )
    ax_trait.set_xticks(range(len(trait_matrix.columns)))
    ax_trait.set_xticklabels(
        trait_matrix.columns,
        rotation=30,
        ha="right",
    )
    ax_trait.set_title("Trait-specific selection signal")
    fig.colorbar(image, ax=ax_trait, label="Added R²")

    overall = nested_models[nested_models["functional_dimension"].eq("overall")]
    overall = overall[overall["model_step"].ne("M0")]
    for component, group in overall.groupby("loss_component", observed=True):
        ax_nested.plot(
            group["model_step"],
            group["delta_r2"],
            marker="o",
            label=component.replace("_", " "),
        )
    ax_nested.axhline(0, linewidth=0.7, color="0.5")
    ax_nested.tick_params(axis="x", rotation=25)
    ax_nested.set_ylabel("ΔR²")
    ax_nested.set_title("Nested functional models")
    ax_nested.legend(frameon=False)

    province_summary = block_metrics.groupby("province_key", as_index=False).agg(
        rao_q=("rao_q", "first"),
        gap=("mean_trait_gap", "mean"),
        exact=("exact_match_share", "mean"),
    )
    ax_gap.scatter(
        province_summary["rao_q"],
        province_summary["gap"],
        s=10,
        alpha=0.65,
    )
    ax_gap.set_xlabel("Rao's Q")
    ax_gap.set_ylabel("Mean trait-substitution gap")
    ax_gap.set_title("Functional breadth and substitution")

    ax_match.scatter(
        province_summary["rao_q"],
        100 * province_summary["exact"],
        s=10,
        alpha=0.65,
    )
    ax_match.set_xlabel("Rao's Q")
    ax_match.set_ylabel("Exact-match share (%)")
    ax_match.set_title("Functional breadth and exact matches")

    for ax, label in zip(axes.ravel(), "ABCD"):
        panel_label(ax, label)

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "trait_and_functional_results")


if __name__ == "__main__":
    main()
