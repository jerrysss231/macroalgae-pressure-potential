# External data layout

The repository does not redistribute third-party environmental, occurrence, trait, boundary or socioeconomic datasets. Place them below `MACROALGAE_DATA_ROOT` using the controlled names below. The controlled names deliberately remove provider-specific download hashes from the analysis code.

```text
$MACROALGAE_DATA_ROOT/
├── experimental/
│   └── macroalgae_nutrient_removal.xlsx
├── biooracle/
│   ├── no3_baseline.nc
│   ├── no3_ssp245.nc
│   ├── no3_ssp370.nc
│   ├── no3_ssp585.nc
│   ├── po4_baseline.nc
│   ├── po4_ssp245.nc
│   ├── po4_ssp370.nc
│   ├── po4_ssp585.nc
│   ├── ph_baseline.nc
│   ├── ph_ssp245.nc
│   ├── ph_ssp370.nc
│   ├── ph_ssp585.nc
│   ├── salinity_baseline.nc
│   ├── salinity_ssp245.nc
│   ├── salinity_ssp370.nc
│   ├── salinity_ssp585.nc
│   ├── temperature_baseline.nc
│   ├── temperature_ssp245.nc
│   ├── temperature_ssp370.nc
│   ├── temperature_ssp585.nc
│   ├── par_baseline.nc
│   └── terrain.nc
├── biogeography/
│   ├── species_meow_province_summary.csv
│   └── meow/
│       └── ... MEOW shapefile distribution ...
├── traits/
│   ├── algaetraits_attributes.csv
│   └── algaetraits_traits_wide.csv
├── eez/
│   └── eez_v12.gpkg
├── socioeconomic/
│   └── world_bank_gdp_per_capita_ppp_2024.csv
└── basemap/
    ├── ne_10m_land/
    ├── ne_10m_coastline/
    ├── ne_10m_admin_0_countries/
    └── ne_10m_ocean/
```

## Source mapping

- Experimental nutrient-removal data: the global marine macroalgae nutrient-removal compilation cited in the manuscript.
- Environmental grids: Bio-ORACLE v3. Baseline and SSP2-4.5, SSP3-7.0 and SSP5-8.5 layers are stored separately.
- Regional species pools: OBIS occurrence records harmonized to WoRMS taxonomy and summarized within MEOW provinces. `species_meow_province_summary.csv` must contain `accepted_name`, `meow_province` and `occurrence_count`.
- Traits: directly documented AlgaeTraits records. Missing primary trait states are not imputed by the public workflow.
- EEZs: Marine Regions Maritime Boundaries and Exclusive Economic Zones, version 12. The analysis retains ordinary 200-nautical-mile EEZ polygons.
- GDP: World Bank World Development Indicators, 2024 GDP per capita at purchasing-power parity in constant 2021 international dollars.
- Basemap layers: Natural Earth; these affect visualization only.

The scripts validate required columns/files before analysis and stop rather than silently substituting missing inputs.
