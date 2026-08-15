# Repository notes

This repository is a publication-oriented refactor of the frozen analysis. It intentionally differs from the development workspace in presentation while preserving the manuscript calculations.

Removed from the public version:

- machine-specific absolute paths;
- duplicated exploratory script variants;
- one-off debugging output and temporary audit files;
- obsolete model configurations;
- large third-party data and generated binary caches.

Retained explicitly:

- endpoint and feature definitions;
- grouped cross-validation design and random seed;
- upper-tail response filtering rule;
- Duan retransformation;
- environmental cleaning and spatial-domain rules;
- species-pool occurrence threshold;
- baseline-fixed nutrient weights and full-baseline pressure reference distributions;
- frozen Q70–Q85 mismatch thresholds;
- spatial-block resampling rules;
- direct-trait complete-case, coverage-gate and functional-distance definitions;
- 200NM EEZ aggregation, transition metrics and formal-support criteria.

The refactor is intended to make the computational logic easier to inspect, not to create a new analysis.
