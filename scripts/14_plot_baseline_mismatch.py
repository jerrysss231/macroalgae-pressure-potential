"""Plot baseline Q75 pressure–potential states from finalized pixel products."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.plotting import set_publication_style,load_land,world_axes,save_figure
P=ProjectPaths.from_env(); R=P.output_dir/"cross_scenario_comparison_and_mismatch_final"; OUT=P.output_dir/"manuscript_figures"
def main():
    set_publication_style(); f=pd.read_csv(R/"recalculated_pixels"/"recalculated_baseline.csv"); t=pd.read_csv(R/"baseline_reference_thresholds.csv"); q=t[np.isclose(t.quantile,.75)].iloc[0]; pressure=f.nutrient_pressure.to_numpy(float)>=q.nutrient_pressure_threshold; potential=f.constrained_potential.to_numpy(float)>=q.constrained_potential_threshold; state=np.select([pressure&~potential,pressure&potential,~pressure&potential],[1,2,3],default=0); land=load_land(P.land_shapefile); fig,ax=plt.subplots(figsize=(7.2,3.2)); world_axes(ax,land); sc=ax.scatter(f.lon,f.lat,c=state,s=.35,vmin=0,vmax=3,rasterized=True); ax.set_title("Baseline Q75 pressure–potential states"); fig.colorbar(sc,ax=ax,ticks=[0,1,2,3],label="0 other · 1 mismatch · 2 opportunity · 3 high potential"); save_figure(fig,OUT/"baseline_pressure_potential_mismatch")
if __name__=="__main__": main()
