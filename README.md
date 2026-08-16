# Biogeographic constraints shape global macroalgal nutrient-removal potential

Analysis code for the manuscript **“Biogeographic constraints shape global macroalgal nutrient-removal potential.”**

This repository contains the workflow used to model macroalgal nitrogen and phosphorus removal, project species-level removal rates globally, account for regional species availability, quantify nutrient pressure–potential mismatch, and evaluate future and EEZ-scale patterns.

## Workflow

The numbered scripts should be run in order.

| Stage | Script | Purpose |
|---|---|---|
| 01 | `01_fit_nutrient_model.py` | Fit and validate the pooled NO3/PO4 TabPFN model using grouped five-fold cross-validation and Duan retransformation. |
| 02 | `02_project_species_rates.py` | Project species-level NO3 and PO4 removal rates to the global 0.25° coastal grid. |
| 03 | `03_build_pressure_potential_mismatch.py` | Calculate unconstrained and biogeographically constrained removal potential, nutrient pressure, mismatch and threshold sensitivity. |
| 04 | `04_audit_environmental_space.py` | Assess environmental-space coverage using PCA and pairwise convex hulls. |
| 05 | `05_build_environment_change_matrix.py` | Calculate future-minus-baseline changes in environmental conditions. |
| 06 | `06_environmental_attribution.py` | Relate environmental change to future changes in constrained potential and biogeographic constraint loss. |
| 07 | `07_ecological_heterogeneity.py` | Analyse variation among biogeographic realms, nutrient regimes and regional species pools. |
| 08 | `08_prepare_traits.py` | Prepare directly documented functional traits and calculate functional distances. |
| 09 | `09_trait_constraint_analysis.py` | Analyse trait associations, functional breadth, functional substitution and biogeographic constraint. |
| 10 | `10_aggregate_eez.py` | Aggregate global results to EEZs and perform EEZ-scale spatial inference. |
| 11–14 | Plotting scripts | Generate manuscript figures from finalized analysis outputs. |

Additional workflow details are provided in [`docs/workflow.md`](docs/workflow.md).

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

A CUDA-capable environment is recommended for the global TabPFN prediction stage.

## Data

Large external datasets are not included in this repository.

Set `MACROALGAE_DATA_ROOT` to a local directory containing the required inputs described in [`docs/data_layout.md`](docs/data_layout.md):

```bash
export MACROALGAE_DATA_ROOT=/path/to/macroalgae-data
```

Generated outputs are written by default to:

```text
$MACROALGAE_DATA_ROOT/outputs/macroalgae_potential_025deg/
```

The output directory can be changed with `MACROALGAE_OUTPUT_DIR`. Bio-ORACLE files can be stored separately using `MACROALGAE_BIOORACLE_DIR`.

The analysis uses data from:

- the global macroalgal nutrient-removal dataset;
- Bio-ORACLE v3;
- OBIS and WoRMS;
- Marine Ecoregions of the World;
- AlgaeTraits;
- Marine Regions EEZ v12;
- World Bank World Development Indicators; and
- Natural Earth.

Please cite the original data providers when reusing these datasets.

## Running the analysis

After installing the package and arranging the external data, run:

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

The plotting scripts can then be run independently from the finalized analysis tables.

## Key analysis settings

The main settings used in the manuscript are:

- NO3 and PO4 removal rates are modelled jointly with TabPFN 8.0.0 using a pooled regressor on a `log1p` target. Predictions are retransformed using a pooled Duan smearing factor.
- Candidate species require more than 15 valid modelling records.
- Global projections use a 0.25° grid and coastal waters with `0 < depth <= 50 m`.
- Projection constants are a 12 h photoperiod, algal density of 2.5 g L-1 and experimental duration of 24 h.
- NO3 and PO4 predictions are combined using local baseline Redfield-normalized nutrient weights. These weights are retained across future scenarios.
- Unconstrained potential is the maximum predicted removal potential across all candidate species.
- Biogeographically constrained potential is the maximum among candidate species present in the corresponding regional MEOW species pool.
- Regional availability requires at least three occurrence records within a MEOW province. Regional species pools are held fixed across future scenarios.
- Nutrient pressure is calculated from area-weighted baseline nutrient reference distributions.
- Primary mismatch is defined using baseline Q75 thresholds. Q70, Q75, Q80 and Q85 are used to evaluate threshold sensitivity.
- Environmental attribution uses area-weighted models, Shapley variance decomposition and partial correlations. These analyses are interpreted as spatial associations rather than causal effects.
- Trait analyses use directly documented traits without imputation. Functional analyses require at least 80% trait-resolved coverage of the regional candidate pool.
- EEZ aggregation uses ordinary 200-nautical-mile EEZ features and latitude-adjusted overlap weights.
- Formal EEZ inference uses 5° spatial blocks and is restricted to EEZs containing at least 100 valid pixels and five independent blocks.

The unconstrained optimum is a counterfactual benchmark and does not imply that non-local species should be introduced.

Annualized removal potential represents standardized continuous-operation potential per unit biomass and should not be interpreted as realized annual field removal.

## Reproducing manuscript outputs

Scripts 01–10 generate the analysis outputs used by the plotting scripts. Scripts 11–14 reproduce the main publication figures from these finalized outputs.

A full numerical rerun requires the external environmental, occurrence and geospatial datasets described in [`docs/data_layout.md`](docs/data_layout.md).

## Code and data availability

All analysis code required to reproduce the workflow is provided in this repository. Large third-party datasets and generated global outputs are not redistributed because of their size and external licensing or distribution conditions.

## Citation

A formal citation and archived release DOI will be added when the manuscript and repository release are finalized.
