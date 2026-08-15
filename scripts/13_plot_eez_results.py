"""Plot EEZ trajectories, persistent mismatch, opportunity and economic context."""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.plotting import set_publication_style,panel_label,save_figure
P=ProjectPaths.from_env(); D=P.output_dir/"eez_analysis"; OUT=P.output_dir/"manuscript_figures"
def main():
    set_publication_style(); table=pd.read_csv(D/"eez_portfolio.csv"); eez=gpd.read_file(P.eez_geopackage); idc="MRGID" if "MRGID" in eez else "mrgid"; eez=eez.rename(columns={idc:"MRGID"}).merge(table,on="MRGID",how="left")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.2)); a,b,c,d=axes.ravel(); codes={k:i for i,k in enumerate(["stable","directionally improving","directionally worsening","scenario-sensitive","high-forcing-sensitive"])}; eez["trajectory_code"]=eez.trajectory_class.map(codes); eez.plot(column="trajectory_code",ax=a,categorical=True,legend=False,missing_kwds={"color":"0.9"}); a.set_axis_off(); a.set_title("EEZ future trajectories")
    b.scatter(100*table.opportunity_fraction_q75,100*table.persistent_mismatch_fraction,s=12,alpha=.7); b.set_xlabel("Baseline biophysical opportunity (%)"); b.set_ylabel("Persistent mismatch (%)"); b.set_title("Persistent mismatch and opportunity")
    non=table[table.trajectory_class.ne("stable")].copy(); y=np.arange(len(non)); c.axvline(0,c="0.5",lw=.7); offsets=(-.18,0,.18)
    for off,s in zip(offsets,("ssp245","ssp370","ssp585")): c.scatter(100*non[f"delta_{s}"],y+off,s=10,label=s)
    c.set_yticks(y); c.set_yticklabels(non.GEONAME,fontsize=6); c.set_xlabel("Δ mismatch (percentage points)"); c.set_title("Non-stable EEZ trajectories"); c.legend(frameon=False)
    valid=table.gdp_per_capita_ppp_2024.notna()&table.persistent_mismatch_fraction.notna(); d.scatter(table.loc[valid,"gdp_per_capita_ppp_2024"],100*table.loc[valid,"persistent_mismatch_fraction"],s=12,alpha=.7); d.set_xscale("log"); d.set_xlabel("GDP per capita, PPP"); d.set_ylabel("Persistent mismatch (%)"); d.set_title("Economic context")
    for ax,label in zip(axes.ravel(),"ABCD"): panel_label(ax,label)
    fig.tight_layout(); save_figure(fig,OUT/"eez_pressure_potential_summary")
if __name__=="__main__": main()
