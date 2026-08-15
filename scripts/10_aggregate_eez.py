"""Aggregate global pressure–potential metrics to 200-nautical-mile EEZs."""
from __future__ import annotations
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import box
from macroalgae_repro.paths import ProjectPaths

P=ProjectPaths.from_env(); ROOT=P.output_dir
PIX=ROOT/"cross_scenario_comparison_and_mismatch_final"/"recalculated_pixels"
THR=ROOT/"cross_scenario_comparison_and_mismatch_final"/"baseline_reference_thresholds.csv"
OUT=ROOT/"eez_analysis"
SCENS=("baseline","ssp245","ssp370","ssp585"); FUT=SCENS[1:]
BLOCK=5.0; MIN_PIX=100; MIN_BLOCK=5; NPERM=9999; NBOOT=4999; SEED=42; EFFECT=.01
GEOD=Geod(ellps="WGS84")

def read(path):
    if not path.exists(): raise FileNotFoundError(path)
    return pd.read_csv(path,encoding="utf-8-sig",low_memory=False)
def bh(p):
    p=np.asarray(p,float); out=np.full(len(p),np.nan); ok=np.flatnonzero(np.isfinite(p))
    if not len(ok): return out
    order=ok[np.argsort(p[ok])]; q=p[order]*len(order)/np.arange(1,len(order)+1); q=np.minimum.accumulate(q[::-1])[::-1]; out[order]=np.minimum(q,1); return out
def wmean(x,w):
    x=np.asarray(x,float); w=np.asarray(w,float); ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    return float(np.sum(x[ok]*w[ok])/np.sum(w[ok])) if ok.any() else np.nan
def wfrac(flag,valid,w):
    flag=np.asarray(flag,bool); valid=np.asarray(valid,bool); w=np.asarray(w,float); ok=valid&np.isfinite(w)&(w>0)
    return float(np.sum(w[ok&flag])/np.sum(w[ok])) if ok.any() else np.nan
def block_id(lon,lat):
    bx=np.floor((((np.asarray(lon,float)+180)%360))/BLOCK).astype(int); by=np.floor((np.asarray(lat,float)+90)/BLOCK).astype(int); return by*10000+bx

def load_pixels():
    data={}
    for s in SCENS:
        f=read(PIX/f"recalculated_{s}.csv"); f["lon_key"]=f.lon.round(6); f["lat_key"]=f.lat.round(6)
        if f.duplicated(["lon_key","lat_key"]).any(): raise ValueError(f"Duplicate coordinates in {s}")
        data[s]=f.set_index(["lon_key","lat_key"])
    common=data["baseline"].index
    for s in FUT: common=common.intersection(data[s].index,sort=False)
    common=pd.MultiIndex.from_frame(common.to_frame(index=False).sort_values(["lat_key","lon_key"]))
    return {s:data[s].reindex(common).reset_index() for s in SCENS}
def q75_thresholds():
    t=read(THR); q=t[np.isclose(t["quantile"],.75)]
    if len(q)!=1: raise ValueError("Exactly one Q75 threshold row is required")
    return float(q.nutrient_pressure_threshold.iloc[0]),float(q.constrained_potential_threshold.iloc[0])
def eez_columns(eez):
    def pick(candidates):
        for c in candidates:
            if c in eez.columns:return c
        raise ValueError(f"Missing EEZ field: {candidates}")
    return pick(("MRGID","mrgid")),pick(("GEONAME","geoname","TERRITORY1"))
def geodesic_area(geom):
    if geom is None or geom.is_empty:return 0.0
    return abs(GEOD.geometry_area_perimeter(geom)[0])
def build_overlap(baseline):
    eez=gpd.read_file(P.eez_geopackage); eez=eez.set_crs("EPSG:4326") if eez.crs is None else eez.to_crs("EPSG:4326"); idc,namec=eez_columns(eez)
    eez=eez[[idc,namec,"geometry"]].rename(columns={idc:"MRGID",namec:"GEONAME"}); eez=eez[eez.geometry.notna()].copy()
    cells=gpd.GeoDataFrame({"pixel_id":np.arange(len(baseline)),"lon":baseline.lon.to_numpy(float),"lat":baseline.lat.to_numpy(float)},geometry=[box(lon-.125,lat-.125,lon+.125,lat+.125) for lon,lat in zip(baseline.lon,baseline.lat)],crs="EPSG:4326")
    candidates=gpd.sjoin(cells,eez,how="inner",predicate="intersects")[["pixel_id","MRGID","GEONAME","index_right"]]
    rows=[]
    for item in candidates.itertuples(index=False):
        cell=cells.geometry.iloc[item.pixel_id]; region=eez.geometry.loc[item.index_right]; inter=cell.intersection(region); fraction=geodesic_area(inter)/geodesic_area(cell)
        if fraction>0: rows.append((item.pixel_id,item.MRGID,item.GEONAME,fraction))
    out=pd.DataFrame(rows,columns=["pixel_id","MRGID","GEONAME","overlap_fraction"]); lat=baseline.lat.to_numpy(float); out["ecological_weight"]=np.cos(np.deg2rad(lat[out.pixel_id]))*out.overlap_fraction; out["block_id"]=block_id(baseline.lon.to_numpy(float)[out.pixel_id],lat[out.pixel_id]); return out
def scenario_metrics(data,overlap,pthr,cthr):
    rows=[]
    for s in SCENS:
        f=data[s]; pressure=f.nutrient_pressure.to_numpy(float); potential=f.constrained_potential.to_numpy(float); loss=f.constraint_loss.to_numpy(float); mismatch=f.mismatch_q75.to_numpy(bool); valid=f.valid_mismatch.to_numpy(bool); opportunity=(pressure>=pthr)&(potential>=cthr)&valid
        for (mrgid,name),g in overlap.groupby(["MRGID","GEONAME"],observed=True):
            idx=g.pixel_id.to_numpy(int); w=g.ecological_weight.to_numpy(float); v=valid[idx]; rows.append({"MRGID":mrgid,"GEONAME":name,"scenario":s,"n_pixels":int(np.unique(idx[v]).size),"n_blocks":int(np.unique(g.block_id.to_numpy()[v]).size),"mean_nutrient_pressure":wmean(pressure[idx][v],w[v]),"mean_constrained_potential":wmean(potential[idx][v],w[v]),"mean_constraint_loss":wmean(loss[idx][v],w[v]),"mismatch_fraction_q75":wfrac(mismatch[idx],v,w),"opportunity_fraction_q75":wfrac(opportunity[idx],v,w),"valid_weight":float(np.sum(w[v]))})
    return pd.DataFrame(rows)
def persistent_metrics(data,overlap):
    mismatch=np.vstack([data[s].mismatch_q75.to_numpy(bool) for s in SCENS]); valid=np.vstack([data[s].valid_mismatch.to_numpy(bool) for s in SCENS]).all(0); persistent=mismatch.all(0); rows=[]
    for (mrgid,name),g in overlap.groupby(["MRGID","GEONAME"],observed=True):
        idx=g.pixel_id.to_numpy(int); w=g.ecological_weight.to_numpy(float); rows.append({"MRGID":mrgid,"GEONAME":name,"persistent_mismatch_fraction":wfrac(persistent[idx],valid[idx],w)})
    return pd.DataFrame(rows)
def trajectory(delta):
    state=np.where(delta>=EFFECT,1,np.where(delta<=-EFFECT,-1,0))
    if np.all(state==0):return "stable"
    if np.array_equal(state,np.array([0,0,1])):return "high-forcing-sensitive"
    if (state==1).any() and (state==-1).any():return "scenario-sensitive"
    if (state==1).any():return "directionally worsening"
    if (state==-1).any():return "directionally improving"
    return "stable"
def inference_for_eez(base,future,g,rng):
    idx=g.pixel_id.to_numpy(int); blocks=g.block_id.to_numpy(int); w=g.ecological_weight.to_numpy(float); valid=base.valid_mismatch.to_numpy(bool)[idx]&future.valid_mismatch.to_numpy(bool)[idx]; idx=idx[valid]; blocks=blocks[valid]; w=w[valid]
    if np.unique(idx).size<MIN_PIX or np.unique(blocks).size<MIN_BLOCK:return np.nan,np.nan,np.nan,np.nan,"insufficient_spatial_support"
    y=future.mismatch_q75.to_numpy(float)[idx]-base.mismatch_q75.to_numpy(float)[idx]; keys=np.unique(blocks); groups=[np.flatnonzero(blocks==k) for k in keys]; observed=wmean(y,w); extreme=0
    for _ in range(NPERM):
        signed=y.copy()
        for sign,rows in zip(rng.choice((-1.,1.),len(groups)),groups):signed[rows]*=sign
        extreme += abs(wmean(signed,w))>=abs(observed)
    boot=np.empty(NBOOT)
    for i in range(NBOOT):
        choose=rng.integers(0,len(groups),len(groups)); rows=np.concatenate([groups[j] for j in choose]); boot[i]=wmean(y[rows],w[rows])
    lo,hi=np.quantile(boot[np.isfinite(boot)],[.025,.975]); return observed,(extreme+1)/(NPERM+1),float(lo),float(hi),"tested"
def cross_scenario(data,overlap,metrics):
    rng=np.random.default_rng(SEED); base=data["baseline"]; table=metrics.pivot(index=["MRGID","GEONAME"],columns="scenario",values="mismatch_fraction_q75").reset_index(); rows=[]; groups={(m,n):g for (m,n),g in overlap.groupby(["MRGID","GEONAME"],observed=True)}
    for r in table.itertuples(index=False):
        delta=np.array([getattr(r,s)-r.baseline for s in FUT],float); row={"MRGID":r.MRGID,"GEONAME":r.GEONAME,"trajectory_class":trajectory(delta)}
        for s,d in zip(FUT,delta):
            obs,p,lo,hi,status=inference_for_eez(base,data[s],groups[(r.MRGID,r.GEONAME)],rng); row[f"delta_{s}"]=d; row[f"inference_delta_{s}"]=obs; row[f"p_{s}"]=p; row[f"ci_low_{s}"]=lo; row[f"ci_high_{s}"]=hi; row[f"inference_status_{s}"]=status
        rows.append(row)
    out=pd.DataFrame(rows)
    for s in FUT:
        out[f"p_fdr_{s}"]=bh(out[f"p_{s}"]); direction=np.sign(out[f"delta_{s}"]); out[f"formally_supported_{s}"]=(np.abs(out[f"delta_{s}"])>=EFFECT)&(out[f"p_fdr_{s}"]<.05)&(((direction<0)&(out[f"ci_high_{s}"]<0))|((direction>0)&(out[f"ci_low_{s}"]>0)))
    return out
def add_gdp(portfolio):
    if not P.gdp_2024_csv.exists():return portfolio
    gdp=read(P.gdp_2024_csv); iso=next((c for c in ("sovereign_iso3","iso3","Country Code") if c in gdp.columns),None); value=next((c for c in ("sovereign_gdp_per_capita_ppp_constant_2021","gdp_per_capita_ppp_2024","2024") if c in gdp.columns),None)
    if iso is None or value is None:return portfolio
    eez=gpd.read_file(P.eez_geopackage); idc,_=eez_columns(eez); eez=eez.rename(columns={idc:"MRGID"}); eez_iso=next((c for c in ("ISO_SOV1","ISO_TER1","SOVEREIGN1") if c in eez.columns),None)
    if eez_iso is None:return portfolio
    lookup=eez[["MRGID",eez_iso]].drop_duplicates("MRGID").rename(columns={eez_iso:"sovereign_iso3"}); out=portfolio.merge(lookup,on="MRGID",how="left"); return out.merge(gdp[[iso,value]].rename(columns={iso:"sovereign_iso3",value:"gdp_per_capita_ppp_2024"}),on="sovereign_iso3",how="left")
def main():
    OUT.mkdir(parents=True,exist_ok=True); data=load_pixels(); pthr,cthr=q75_thresholds(); overlap=build_overlap(data["baseline"]); overlap.to_csv(OUT/"pixel_eez_overlap.csv.gz",index=False,compression="gzip"); scen=scenario_metrics(data,overlap,pthr,cthr); scen.to_csv(OUT/"eez_scenario_metrics.csv",index=False); cross=cross_scenario(data,overlap,scen).merge(persistent_metrics(data,overlap),on=["MRGID","GEONAME"],how="left"); cross.to_csv(OUT/"eez_cross_scenario_metrics.csv",index=False); base=scen[scen.scenario.eq("baseline")][["MRGID","GEONAME","opportunity_fraction_q75","mismatch_fraction_q75"]]; add_gdp(cross.merge(base,on=["MRGID","GEONAME"],how="left")).to_csv(OUT/"eez_portfolio.csv",index=False); print("Saved EEZ outputs to",OUT)
if __name__=="__main__":main()
