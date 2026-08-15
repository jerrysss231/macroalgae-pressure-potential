# A global mismatch between nutrient pressure and macroalgal removal potential

Reproducible analysis code for the manuscript **“A global mismatch between nutrient pressure and macroalgal removal potential.”**

The workflow separates three quantities that should not be conflated: physiological nutrient-removal performance, removal potential attainable within present-day regional species pools, and the spatial alignment of that potential with nutrient pressure.

## Repository design

This is a publication-oriented refactor of the frozen analysis rather than a dump of development scripts. Machine-specific paths, temporary debugging code, obsolete outputs and duplicated exploratory variants have been removed. Scientific definitions, frozen thresholds, model settings and spatial inference rules used by the manuscript are retained.

The numbered scripts form the analysis pipeline:

| Stage | Script | Purpose |
|---|---|---|
| 01 | `01_fit_nutrient_model.py` | Fit and validate the pooled NO3/PO4 TabPFN model with grouped five-fold CV and Duan retransformation. |
| 02 | `02_project_species_rates.py` | Project species-level NO3 and PO4 removal rates to the harmonized 0.25° coastal grid. |
| 03 | `03_build_pressure_potential_mismatch.py` | Calculate unconstrained and biogeographically constrained potential, nutrient pressure, mismatch and threshold robustness. |
| 04 | `04_audit_environmental_space.py` | Quantify PCA–pairwise convex-hull environmental-space coverage. |
| 05 | `05_build_environment_change_matrix.py` | Assemble future-minus-baseline environmental-change fields. |
| 06 | `06_environmental_attribution.py` | Run area-weighted Shapley attribution, partial associations and spatial-block uncertainty analyses. |
| 07 | `07_ecological_heterogeneity.py` | Analyse MEOW, Redfield-relative nutrient-regime and species-pool heterogeneity. |
| 08 | `08_prepare_traits.py` | Clean directly documented traits and construct the functional-distance space. |
| 09 | `09_trait_constraint_analysis.py` | Analyse trait selection, Rao's Q, functional substitution and constraint loss. |
| 10 | `10_aggregate_eez.py` | Aggregate to ordinary 200-nautical-mile EEZs and run spatial-block inference. |
| 11–14 | plotting scripts | Produce compact publication-style visual summaries from finalized analysis tables without redefining analytical metrics. |

Detailed stage dependencies and frozen definitions are described in [`docs/workflow.md`](docs/workflow.md).

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

TabPFN is the computationally demanding component of the workflow; a CUDA-capable environment is recommended for the full global prediction stage.

## External data

Large and third-party datasets are not redistributed in this repository. Set `MACROALGAE_DATA_ROOT` to a local directory containing the inputs listed in [`docs/data_layout.md`](docs/data_layout.md):

```bash
export MACROALGAE_DATA_ROOT=/path/to/macroalgae-data
```

Generated outputs default to:

```text
$MACROALGAE_DATA_ROOT/outputs/macroalgae_potential_025deg/
```

A different output location can be supplied with `MACROALGAE_OUTPUT_DIR`. Bio-ORACLE files can optionally be stored elsewhere with `MACROALGAE_BIOORACLE_DIR`.

## Running the workflow

After installing the package and arranging the external inputs:

```bash
python scripts/01_fit_nutrient_model.py
python scripts/02_project_species_rates.py
python scripts/03_build_pressure_potential_mismatch.py
python scripts/04_audit_environmental_space.py
python scripts/05_build_environment_change_matrix.py
python scripts/06_environmental_attribution.py
python scripts/07_ecological_heterogeneity.py
python scripts/08_prepare_traits.py
python scripts/09_trait_constraint_analysis.py
python scripts/10_aggregate_eez.py
```

Plotting scripts can then be run independently from the finalized analysis tables.

## Frozen analytical definitions

The public code preserves the manuscript analysis choices. In particular:

- NO3 and PO4 are modelled jointly with a pooled TabPFN regressor on a `log1p` target; retransformation uses a pooled Duan smearing factor.
- Candidate species require more than 15 valid modelling records.
- Global projection uses a 0.25° grid and waters with `0 < depth <= 50 m`.
- Projection constants are 12 h photoperiod, 2.5 g L-1 algal density and 24 h experimental duration.
- NO3 and PO4 predictions are combined with local baseline Redfield-normalized nutrient weights, which are held fixed in future scenarios.
- Unconstrained potential is the best candidate across the full modelled species set. Constrained potential is the best candidate present in the regional MEOW species pool. The unconstrained optimum is a counterfactual benchmark, not a recommendation to introduce non-local species.
- Regional availability requires at least three occurrence records within a MEOW province and is held fixed across future scenarios.
- Nutrient pressure uses area-weighted, right-continuous baseline nutrient reference distributions. Those references are frozen from the complete valid baseline projection table before the four-scenario spatial intersection.
- Q75 mismatch is defined as nutrient pressure at or above its baseline Q75 threshold with constrained potential below its baseline Q75 threshold. Q70, Q75, Q80 and Q85 are used for threshold robustness.
- Environmental attribution uses area-weighted models, Shapley decomposition and partial correlations; these are interpreted as spatial associations rather than causal effects.
- Trait inference uses directly documented traits without imputation. All biological species remain in opportunity and winner calculations, while functional inference uses the direct-trait complete-case species set and requires at least 80% trait-resolved coverage of the regional candidate pool.
- EEZ aggregation retains ordinary `POL_TYPE == "200NM"` features, applies latitude-adjusted geodesic overlap weights and does not renormalize overlapping jurisdictions.
- EEZ formal inference uses 5° spatial blocks and is restricted to EEZs with at least 100 valid pixels and five independent blocks. Directional trajectory classes are kept separate from formal statistical support.

Annualized removal potential is a standardized continuous-operation, per-biomass quantity. It is **not** realized annual field removal.

## Data sources

The workflow uses the global macroalgal nutrient-removal dataset, Bio-ORACLE v3, OBIS/WoRMS-derived occurrence summaries, Marine Ecoregions of the World, AlgaeTraits, Marine Regions EEZ v12, World Bank World Development Indicators and Natural Earth basemap layers. Please cite the original data providers when reusing these inputs.

## Validation of the public refactor

The publication refactor is checked for Python syntax and AST validity, consistent line-length formatting, and removal of machine-specific paths and obsolete development identifiers. The uploaded analytical files are byte-matched against the locally checked clean versions using Git blob hashes.

A complete numerical rerun is not performed as part of these repository checks because the large external environmental, occurrence and jurisdictional datasets are not redistributed here. Reproducing numerical outputs therefore requires arranging the external inputs described in `docs/data_layout.md` and running the numbered workflow.

## Reproducibility note

The publication repository reorganizes the frozen analysis into a linear, portable workflow. Species-level global prediction is separated from the authoritative pressure–potential calculation so that model prediction, biogeographic filtering and mismatch definitions remain auditable as distinct stages. Numerical outputs are not committed because the underlying environmental and geospatial inputs are large and, in several cases, externally licensed.

## Citation

A formal citation and archived release DOI will be added when the manuscript version and repository release are frozen.
