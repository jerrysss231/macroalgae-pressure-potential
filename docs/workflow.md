# Reproducibility workflow

The public workflow separates prediction, ecological constraints and statistical interpretation so each scientific definition can be audited independently.

## 1. Experimental model

`scripts/01_fit_nutrient_model.py`

- builds the NO3/PO4 long-format training table;
- removes only endpoint-specific upper-tail observations above `Q3 + 3 × IQR`;
- fits preprocessing within each grouped cross-validation fold;
- uses stratified grouped five-fold CV with original experimental rows as groups and seed 42;
- fits one pooled TabPFN model on `log1p` removal rate;
- applies one pooled Duan smearing factor per fold;
- refits the final model on all cleaned NO3/PO4 records.

## 2. Global species prediction

`scripts/02_project_species_rates.py`

Bio-ORACLE variables are mean-coarsened to 0.25° and aligned to the baseline nitrate template. The coastal domain requires valid NO3, PO4, pH, salinity, temperature and light values and `0 < depth <= 50 m`. Projection constants are 12 h photoperiod, 2.5 g L-1 density and 24 h experimental duration. Species with more than 15 valid modelling records are projected for NO3 and PO4 and cached as compact arrays.

The script also assigns each grid cell to a MEOW province. Regional candidate pools require at least three occurrence records per species and province. Present-day pools are retained under all future scenarios.

## 3. Pressure–potential framework

`scripts/03_build_pressure_potential_mismatch.py`

This is the authoritative stage for the paper's principal spatial metrics.

For each pixel, baseline nutrient weights are

```text
wN = (NO3 / 16) / ((NO3 / 16) + PO4)
wP = 1 - wN
```

with `wN = wP = 0.5` when the denominator is zero. The baseline weights are frozen in every future scenario.

Species-specific composite removal is the weighted sum of predicted NO3 and PO4 removal. Unconstrained potential is the maximum over all modelled candidate species; constrained potential is the maximum over species in the regional pool. Cells without an eligible regional species retain missing constrained potential rather than zero. Constraint loss is `unconstrained - constrained`.

Nutrient pressure is calculated from area-weighted, right-continuous baseline ECDFs of NO3 and PO4, with `cos(latitude)` weights. Future nutrient values are evaluated against those baseline reference distributions and combined using the same frozen baseline nutrient weights.

Q70, Q75, Q80 and Q85 pressure and constrained-potential thresholds are area-weighted baseline quantiles and remain fixed in future scenarios. Q75 mismatch is high nutrient pressure (`>= Q75`) with constrained potential below the baseline Q75 potential threshold. Threshold-stable mismatch satisfies the mismatch definition at all four quantiles.

## 4. Environmental-space representation

`scripts/04_audit_environmental_space.py`

The audit is run separately for NO3 and PO4 using temperature, pH, salinity, light intensity and endpoint-specific concentration. Predictors are standardized from the experimental training data, PCA is fitted to the training environment, and the minimum number of PCs explaining at least 90% of variance is retained. Coverage is the fraction of all retained-PC pairwise convex hulls that contain a projected pixel. This diagnostic never gates the primary predictions.

## 5–7. Future environmental attribution and ecological heterogeneity

`scripts/05_build_environment_change_matrix.py` produces harmonized absolute and future-minus-baseline environmental fields.

`scripts/06_environmental_attribution.py` relates changes in temperature, NO3, PO4, pH and salinity separately to changes in constrained potential and constraint loss. Area-weighted linear models, Shapley decomposition and partial correlations are interpreted as spatial associations, not causal effects. Spatial robustness uses 5° blocks.

`scripts/07_ecological_heterogeneity.py` evaluates MEOW realms, regional species-pool richness and Redfield-relative nutrient regimes. Baseline molar `NO3:PO4 < 16` is labelled N-depleted relative to Redfield and `>= 16` P-depleted relative to Redfield; these labels describe relative stoichiometry, not experimentally demonstrated nutrient limitation.

## 8–9. Traits and functional substitution

`scripts/08_prepare_traits.py` keeps directly documented primary traits, constructs dimension-aware functional distances and does not impute missing primary trait states.

`scripts/09_trait_constraint_analysis.py` evaluates trait selection, Rao's Q and functional substitution. Primary inference is restricted to regional candidate pools with at least 80% trait-resolved coverage. The trait-substitution gap is the minimum functional distance from the unconstrained best-performing species to a trait-resolved species in the regional pool; zero denotes an exact functional match.

## 10. EEZ aggregation

`scripts/10_aggregate_eez.py`

Each 0.25° model cell is intersected with 200-nautical-mile EEZ polygons and weighted by `cos(latitude) × geodesic overlap fraction`. Global baseline Q75 thresholds are retained. Overlapping EEZ polygons are aggregated independently and are not renormalized between jurisdictions.

Formal inference requires at least 100 valid pixels and five independent 5° blocks. Spatial-block sign-flip permutations and block bootstrap intervals are combined with scenario-specific Benjamini–Hochberg correction. A formally supported directional change must exceed 1 percentage point, have a same-direction 95% bootstrap interval excluding zero, and have FDR-adjusted `P < 0.05`.

## Plotting

Scripts 11–14 consume finalized analysis tables only. They do not refit models or redefine scientific metrics.
