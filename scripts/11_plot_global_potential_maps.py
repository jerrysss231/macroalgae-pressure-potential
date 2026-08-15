"""Plot constrained potential and biogeographic constraint loss for all scenarios."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.plotting import set_publication_style,load_land,world_axes,panel_label,save_figure
P=ProjectPaths.from_env(); PIX=P.output_dir/"cross_scenario_comparison_and_mismatch_final"/"recalculated_pixels"; OUT=P.output_dir/"manuscript_figures"
SCENS=("baseline","ssp245","ssp370","ssp585"); LABELS=("Baseline","SSP2-4.5","SSP3-7.0","SSP5-8.5")
def read(s): return pd.read_csv(PIX/f"recalculated_{s}.csv",encoding="utf-8-sig")
def main():
    set_publication_style(); land=load_land(P.land_shapefile); frames=[read(s) for s in SCENS]
    vmax_p=np.nanquantile(np.concatenate([f.constrained_potential.to_numpy(float) for f in frames]),.98); vmax_l=np.nanquantile(np.concatenate([f.constraint_loss.to_numpy(float) for f in frames]),.98)
    fig,axes=plt.subplots(2,4,figsize=(7.2,4.0),sharex=True,sharey=True)
    for j,(f,label) in enumerate(zip(frames,LABELS)):
        for i,(column,vmax,title) in enumerate((("constrained_potential",vmax_p,"Constrained potential"),("constraint_loss",vmax_l,"Biogeographic loss"))):
            ax=axes[i,j]; world_axes(ax,land); sc=ax.scatter(f.lon,f.lat,c=f[column],s=.25,vmin=0,vmax=vmax,rasterized=True,zorder=0); ax.set_title(label if i==0 else title if j==0 else "")
            if j>0: ax.set_ylabel("")
            if i==0: ax.set_xlabel("")
            panel_label(ax,chr(ord('A')+i*4+j)); fig.colorbar(sc,ax=ax,orientation="horizontal",fraction=.05,pad=.08)
    fig.subplots_adjust(wspace=.12,hspace=.18); save_figure(fig,OUT/"global_constrained_potential_and_loss")
if __name__=="__main__": main()
