"""Portable project paths controlled by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SCENARIOS = ("baseline", "ssp245", "ssp370", "ssp585")


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve external inputs and generated outputs without machine-specific paths."""

    data_root: Path
    output_dir: Path
    biooracle_dir: Path

    @classmethod
    def from_env(cls) -> "ProjectPaths":
        root = Path(os.environ.get("MACROALGAE_DATA_ROOT", "data")).expanduser().resolve()
        output = Path(
            os.environ.get(
                "MACROALGAE_OUTPUT_DIR",
                root / "outputs" / "macroalgae_potential_025deg",
            )
        ).expanduser().resolve()
        biooracle = Path(
            os.environ.get("MACROALGAE_BIOORACLE_DIR", root / "biooracle")
        ).expanduser().resolve()
        return cls(root, output, biooracle)

    @property
    def experimental_xlsx(self) -> Path:
        return self.data_root / "experimental" / "macroalgae_nutrient_removal.xlsx"

    @property
    def meow_occurrence_summary(self) -> Path:
        return self.data_root / "biogeography" / "species_meow_province_summary.csv"

    @property
    def meow_dir(self) -> Path:
        return self.data_root / "biogeography" / "meow"

    @property
    def algaetraits_attributes(self) -> Path:
        return self.data_root / "traits" / "algaetraits_attributes.csv"

    @property
    def algaetraits_traits_wide(self) -> Path:
        return self.data_root / "traits" / "algaetraits_traits_wide.csv"

    @property
    def eez_geopackage(self) -> Path:
        return self.data_root / "eez" / "eez_v12.gpkg"

    @property
    def gdp_2024_csv(self) -> Path:
        return self.data_root / "socioeconomic" / "world_bank_gdp_per_capita_ppp_2024.csv"

    @property
    def basemap_dir(self) -> Path:
        return self.data_root / "basemap"

    @property
    def land_shapefile(self) -> Path:
        return self.basemap_dir / "ne_10m_land" / "ne_10m_land.shp"

    @property
    def coastline_shapefile(self) -> Path:
        return self.basemap_dir / "ne_10m_coastline" / "ne_10m_coastline.shp"

    @property
    def country_shapefile(self) -> Path:
        return self.basemap_dir / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp"

    @property
    def ocean_shapefile(self) -> Path:
        return self.basemap_dir / "ne_10m_ocean" / "ne_10m_ocean.shp"

    @property
    def par_baseline(self) -> Path:
        return self.biooracle_dir / "par_baseline.nc"

    @property
    def terrain(self) -> Path:
        return self.biooracle_dir / "terrain.nc"

    def environment(self, scenario: str, variable: str) -> Path:
        """Return a standardized Bio-ORACLE file name used by the public workflow."""
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        aliases = {
            "no3": "no3",
            "po4": "po4",
            "ph": "ph",
            "salinity": "salinity",
            "temperature": "temperature",
        }
        try:
            stem = aliases[variable]
        except KeyError as exc:
            raise ValueError(f"Unknown environmental variable: {variable}") from exc
        return self.biooracle_dir / f"{stem}_{scenario}.nc"
