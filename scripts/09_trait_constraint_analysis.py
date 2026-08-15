"""Trait selection, Rao's Q and functional substitution analyses."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from macroalgae_repro.paths import ProjectPaths

P=ProjectPaths.from_env(); ROOT=P.output_dir
TD=ROOT/"traits"; OUT=ROOT/"trait_constraint_analysis"
PIX=ROOT/"cross_scenario_comparison_and_mismatch_final"/"recalculated_pixels"
CTX=ROOT/"ecological_heterogeneity"/"ecological_pixel_context.csv.gz"
CACHE=ROOT/"cache"/"no3_po4_5cv_seed42_log_duan_iqr3"/"predictions"
SCENS=("baseline","ssp245","ssp370","ssp585"); DIMS=("habitat","life_history","morphology","reproduction")
GATE=.80; NPERM=9999; NBOOT=4999; SEED=42; EPS=1e-12; LOSS_EPS=1e-6

def read(path,**kw):
    if not path.exists(): raise FileNotFoundError(path)
    return pd.read_csv(path,encoding="utf-8-sig",low_memory=False,**kw)
def key(x): return " ".join(str(x).replace("\xa0"," ").split()).strip().lower() if pd.notna(x) else ""
def wmean(x,w):
    x=np.asarray(x,float); w=np.asarray(w,float); m=np.isfinite(x)&np.isfinite(w)&(w>0)
    return float(np.sum(x[m]*w[m])/np.sum(w[m])) if m.any() else np.nan
def wfrac(flag,valid,w):
    flag=np.asarray(flag,bool); valid=np.asarray(valid,bool); w=np.asarray(w,float); m=valid&np.isfinite(w)&(w>0)
    return float(np.sum(w[m&flag])/np.sum(w[m])) if m.any() else np.nan
def bh(p):
    p=np.asarray(p,float); out=np.full(len(p),np.nan); ok=np.flatnonzero(np.isfinite(p))
    if not len(ok): return out
    order=ok[np.argsort(p[ok])]; q=p[order]*len(order)/np.arange(1,len(order)+1); q=np.minimum.accumulate(q[::-1])[::-1]
    out[order]=np.minimum(q,1); return out
def block(lon,lat):
    return np.floor((((np.asarray(lon,float)+180)%360))/5).astype(int)+10000*np.floor((np.asarray(lat,float)+90)/5).astype(int)

def traits_and_distances():
    t=read(TD/"direct_traits_cleaned.csv"); meta=read(TD/"trait_metadata.csv"); t["name"]=t["meow_match_name"].map(key)
    dist=pd.read_csv(TD/"functional_distance.csv",index_col=0); dist.index=[key(x) for x in dist.index]; dist.columns=[key(x) for x in dist.columns]
    dd={}
    for d in DIMS:
        m=pd.read_csv(TD/f"functional_distance_{d}.csv",index_col=0); m.index=[key(x) for x in m.index]; m.columns=[key(x) for x in m.columns]; dd[d]=m
    groups=meta.groupby("functional_dimension")["clean_trait"].apply(list).to_dict(); resolved=np.ones(len(t),bool)
    for d in DIMS: resolved &= t[groups[d]].notna().any(axis=1).to_numpy(bool)
    t["trait_resolved"]=resolved
    names=t.loc[resolved,"name"].tolist()
    if not np.isfinite(dist.loc[names,names].to_numpy(float)).all(): raise ValueError("Incomplete resolved trait distance")
    return t,meta,dist,dd

def model_mapping(t):
    m=read(ROOT/"species_lookup.csv").sort_values("species_id").reset_index(drop=True); m["name"]=m["meow_match_name"].map(key)
    lookup={n:i for i,n in enumerate(t["name"])}; mp=np.array([lookup.get(n,-1) for n in m["name"]],int)
    if (mp<0).any(): raise ValueError("Unmapped model species")
    return m,mp

def pixels():
    b=read(PIX/"recalculated_baseline.csv"); b["lk"]=b.lon.round(6); b["ak"]=b.lat.round(6)
    c=read(CTX); c["lk"]=c.lon.round(6); c["ak"]=c.lat.round(6)
    x=b.merge(c[["lk","ak","meow_province","meow_realm"]].drop_duplicates(["lk","ak"]),on=["lk","ak"],validate="one_to_one")
    for col in ("unconstrained_potential","constrained_potential","constraint_loss","nutrient_pressure"): x.rename(columns={col:f"{col}_baseline"},inplace=True)
    for s in SCENS[1:]:
        f=read(PIX/f"recalculated_{s}.csv"); f["lk"]=f.lon.round(6); f["ak"]=f.lat.round(6)
        cols=("unconstrained_potential","constrained_potential","constraint_loss","nutrient_pressure")
        x=x.merge(f[["lk","ak",*cols]].rename(columns={z:f"{z}_{s}" for z in cols}),on=["lk","ak"],validate="one_to_one")
    x["block_id"]=block(x.lon,x.lat); return x

def allowed(model,mapping,t,province):
    o=read(P.meow_occurrence_summary); o["occurrence_count"]=pd.to_numeric(o.occurrence_count,errors="coerce").fillna(0); o=o[o.occurrence_count>=3]
    sets=o.assign(sp=o.accepted_name.map(key),pr=o.meow_province.map(key)).groupby("sp")["pr"].apply(set).to_dict(); pr=np.array([key(x) for x in province],object)
    am=np.vstack([np.isin(pr,tuple(sets.get(n,()))) for n in model.name]); ab=np.zeros((len(t),len(pr)),bool)
    for i,j in enumerate(mapping): ab[j]|=am[i]
    return am,ab

def pool_metrics(ab,t,dist,dd,province):
    resolved=np.flatnonzero(t.trait_resolved.to_numpy(bool)); names=t.loc[resolved,"name"].tolist(); gl={g:i for i,g in enumerate(resolved)}
    D=dist.loc[names,names].to_numpy(float); DD={d:m.loc[names,names].to_numpy(float) for d,m in dd.items()}
    unique,code=np.unique(np.array([key(x) for x in province],object),return_inverse=True); gaps={"overall":np.full((len(t),len(unique)),np.nan)}; gaps.update({d:np.full_like(gaps["overall"],np.nan) for d in DIMS}); rows=[]
    for c,pv in enumerate(unique):
        j=np.flatnonzero(code==c)[0]; allsp=np.flatnonzero(ab[:,j]); rsp=np.array([x for x in allsp if x in gl],int); loc=np.array([gl[x] for x in rsp],int); n=len(allsp); nr=len(loc)
        row={"province_key":pv,"candidate_richness":n,"trait_resolved_richness":nr,"pool_trait_coverage":nr/n if n else np.nan,"rao_q":np.nan if nr==0 else 0 if nr==1 else np.sum(D[np.ix_(loc,loc)])/nr**2}
        for d in DIMS: row[f"rao_q_{d}"]=np.nan if nr==0 else 0 if nr==1 else np.sum(DD[d][np.ix_(loc,loc)])/nr**2
        if nr:
            for g in resolved:
                l=gl[g]; gaps["overall"][g,c]=np.min(D[l,loc])
                for d in DIMS: gaps[d][g,c]=np.min(DD[d][l,loc])
        rows.append(row)
    return pd.DataFrame(rows),code,gaps

def pred_rows(x,s):
    p=read(ROOT/f"potential_pixels_{s}_depth50m_025deg.csv"); p["lk"]=p.lon.round(6); p["ak"]=p.lat.round(6); p["row"]=np.arange(len(p))
    return x[["lk","ak"]].merge(p[["lk","ak","row"]],on=["lk","ak"],validate="one_to_one").row.to_numpy(int)
def combined(model,rows,s,wn,wp):
    mats=[]
    for nutrient in ("NO3","PO4"):
        a=np.empty((len(model),len(rows)),np.float32)
        for i,sid in enumerate(model.species_id): a[i]=np.load(CACHE/s/nutrient/f"species_{int(sid):05d}.npy",mmap_mode="r")[rows]
        mats.append(a)
    return mats[0]*wn[None,:]+mats[1]*wp[None,:]
def best(a,mask=None):
    ok=np.isfinite(a); ok=ok if mask is None else ok&mask; has=ok.any(0); idx=np.full(a.shape[1],-1,int); cols=np.flatnonzero(has)
    if len(cols): idx[cols]=np.argmax(np.where(ok[:,cols],a[:,cols],-np.inf),0)
    return idx,ok
def collapse(mask,mp,n):
    out=np.zeros((n,mask.shape[1]),bool)
    for i,j in enumerate(mp): out[j]|=mask[i]
    return out

def selection(winner,eligible,w,t,s,kind):
    n=eligible.sum(0); valid=np.isfinite(w)&(w>0)&(n>0); expected=eligible[:,valid]@(w[valid]/n[valid]); obs=np.bincount(winner[valid&(winner>=0)],weights=w[valid&(winner>=0)],minlength=len(t)); eps=max(.5*np.min(w[valid]),EPS)
    out=t[["species_id","Scientific name","name","Phylum","trait_resolved"]].copy(); out["scenario"]=s; out["selection_type"]=kind; out["selection_advantage_log2"]=np.where(expected>0,np.log2((obs+eps)/(expected+eps)),np.nan); return out

def r2(y,X):
    b=np.linalg.lstsq(X,y,rcond=None)[0]; f=X@b; ss=np.sum((y-y.mean())**2); return 1-np.sum((y-f)**2)/ss if ss>0 else np.nan
def encode(series):
    states=sorted({z for v in series.astype(str) for z in v.split(";") if z and z!="nan"}); return np.column_stack([series.astype(str).map(lambda v,z=z:float(z in v.split(";"))) for z in states[1:]]) if len(states)>1 else np.empty((len(series),0))
def trait_models(sel,t,meta,rng):
    cols=meta.clean_trait.tolist(); data=sel.merge(t[["species_id","Phylum",*cols]],on=["species_id","Phylum"],how="left"); rows=[]
    for (s,k),g in data.groupby(["scenario","selection_type"],observed=True):
        local=[]
        for m in meta.itertuples():
            d=g[g.selection_advantage_log2.notna()&g[m.clean_trait].notna()]
            if len(d)<10: continue
            y=d.selection_advantage_log2.to_numpy(float); base=np.column_stack([np.ones(len(d)),pd.get_dummies(d.Phylum.astype(str),drop_first=True,dtype=float)]); e=encode(d[m.clean_trait])
            if not e.shape[1]: continue
            full=np.column_stack([base,e]); delta=r2(y,full)-r2(y,base); ph=d.Phylum.to_numpy(); groups=[np.flatnonzero(ph==z) for z in np.unique(ph)]; extreme=0
            for _ in range(NPERM):
                yp=y.copy()
                for ix in groups: yp[ix]=rng.permutation(yp[ix])
                extreme += (r2(yp,full)-r2(yp,base)>=delta)
            local.append({"scenario":s,"selection_type":k,"trait":m.clean_trait,"functional_dimension":m.functional_dimension,"n_species":len(d),"trait_added_r2":delta,"p":(extreme+1)/(NPERM+1)})
        q=bh([z["p"] for z in local])
        for z,v in zip(local,q): z["p_fdr"]=v; rows.append(z)
    return pd.DataFrame(rows)

def weighted_r2(df,y,features,wcol):
    z=df[[y,wcol,*features]].apply(pd.to_numeric,errors="coerce"); ok=np.isfinite(z).all(1)&(z[wcol]>0); z=z[ok]
    if len(z)<=len(features)+2:return np.nan
    X=np.column_stack([np.ones(len(z)),z[features]]); yy=z[y].to_numpy(float); w=z[wcol].to_numpy(float); sw=np.sqrt(w); b=np.linalg.lstsq(X*sw[:,None],yy*sw,rcond=None)[0]; f=X@b; mu=np.sum(yy*w)/np.sum(w); return 1-np.sum(w*(yy-f)**2)/np.sum(w*(yy-mu)**2)
def block_table(x,pool,code,gaps,winners):
    lookup=pool.set_index("province_key"); pk=np.array([key(v) for v in x.meow_province],object); rows=[]
    for s in SCENS:
        loss=x[f"constraint_loss_{s}"].to_numpy(float); pressure=x[f"nutrient_pressure_{s}"].to_numpy(float); u=x[f"unconstrained_potential_{s}"].to_numpy(float); win=winners[s]
        for (pv,bid),idx in x.groupby([pd.Series(pk,index=x.index),"block_id"],observed=True).groups.items():
            idx=np.array(list(idx),int); w=x.area_weight.to_numpy(float)[idx]; lv=loss[idx]; valid=np.isfinite(lv)&np.isfinite(w)&(w>0); pos=valid&(lv>LOSS_EPS)
            if not valid.any() or pv not in lookup.index: continue
            inc=wfrac(pos,valid,w); row={"province_key":pv,"block_id":bid,"scenario":s,"realm":x.meow_realm.iloc[idx[0]],"area_weight_sum":np.sum(w[valid]),"positive_area_weight_sum":np.sum(w[pos]),"logit_loss_incidence":np.log(np.clip(inc,1e-4,1-1e-4)/(1-np.clip(inc,1e-4,1-1e-4))),"mean_positive_log_loss":wmean(np.log1p(lv[pos]),w[pos]) if pos.any() else np.nan,"mean_pressure":wmean(pressure[idx],w),"mean_unconstrained_potential":wmean(u[idx],w)}
            for c in ("candidate_richness","pool_trait_coverage","rao_q",*[f"rao_q_{d}" for d in DIMS]): row[c]=lookup.loc[pv,c]
            row["log_candidate_richness"]=np.log1p(row["candidate_richness"]); validwin=win[idx]>=0; gap=np.full(len(idx),np.nan); gap[validwin]=gaps["overall"][win[idx][validwin],code[idx][validwin]]; row["mean_trait_gap"]=wmean(gap,w); row["mean_positive_trait_gap"]=wmean(gap[pos],w[pos]) if pos.any() else np.nan; row["exact_match_share"]=wfrac(np.isfinite(gap)&(gap<=EPS),np.isfinite(gap),w)
            for d in DIMS:
                gg=np.full(len(idx),np.nan); gg[validwin]=gaps[d][win[idx][validwin],code[idx][validwin]]; row[f"mean_gap_{d}"]=wmean(gg,w); row[f"mean_positive_gap_{d}"]=wmean(gg[pos],w[pos]) if pos.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
def nested(blocks,rng):
    controls=pd.get_dummies(blocks[["scenario","realm"]].astype(str),drop_first=True,dtype=float); frame=pd.concat([blocks.reset_index(drop=True),controls.add_prefix("ctl_").reset_index(drop=True)],axis=1); ctl=[c for c in frame if c.startswith("ctl_")]; base=["mean_pressure","mean_unconstrained_potential","pool_trait_coverage",*ctl]; rows=[]
    specs=(("loss_incidence","logit_loss_incidence","mean_trait_gap","area_weight_sum"),("positive_loss_severity","mean_positive_log_loss","mean_positive_trait_gap","positive_area_weight_sum"))
    def delta_ci(y,red,full,wcol):
        groups=[g.index.to_numpy() for _,g in frame.groupby("province_key")]; vals=np.empty(NBOOT)
        for i in range(NBOOT):
            ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]); sm=frame.loc[ix]; vals[i]=weighted_r2(sm,y,full,wcol)-weighted_r2(sm,y,red,wcol)
        v=vals[np.isfinite(vals)]; return tuple(np.quantile(v,[.025,.975])) if len(v) else (np.nan,np.nan)
    for comp,y,gap,wcol in specs:
        steps=(("M0",base),("M1_richness",[*base,"log_candidate_richness"]),("M2_RaoQ",[*base,"log_candidate_richness","rao_q"]),("M3_gap",[*base,"log_candidate_richness","rao_q",gap])); prev=None; pr2=np.nan
        for name,features in steps:
            val=weighted_r2(frame,y,features,wcol); lo,hi=(np.nan,np.nan) if prev is None else delta_ci(y,prev,features,wcol); rows.append({"loss_component":comp,"functional_dimension":"overall","model_step":name,"r2":val,"delta_r2":val-pr2 if np.isfinite(pr2) else np.nan,"ci_low":lo,"ci_high":hi}); prev,pr2=features,val
        for d in DIMS:
            red=[*base,"log_candidate_richness"]; fd=[*red,f"rao_q_{d}"]; gp=[*fd,f"mean_gap_{d}" if comp=="loss_incidence" else f"mean_positive_gap_{d}"]; r0=weighted_r2(frame,y,red,wcol); r1=weighted_r2(frame,y,fd,wcol); r2v=weighted_r2(frame,y,gp,wcol); lo1,hi1=delta_ci(y,red,fd,wcol); lo2,hi2=delta_ci(y,fd,gp,wcol); rows += [{"loss_component":comp,"functional_dimension":d,"model_step":"dimension_RaoQ","r2":r1,"delta_r2":r1-r0,"ci_low":lo1,"ci_high":hi1},{"loss_component":comp,"functional_dimension":d,"model_step":"dimension_gap","r2":r2v,"delta_r2":r2v-r1,"ci_low":lo2,"ci_high":hi2}]
    return pd.DataFrame(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(SEED); t,meta,dist,dd=traits_and_distances(); model,mp=model_mapping(t); x=pixels(); am,ab=allowed(model,mp,t,x.meow_province); pool,code,gaps=pool_metrics(ab,t,dist,dd,x.meow_province); pool.to_csv(OUT/"regional_functional_pool_metrics.csv",index=False)
    wn=x.w_no3_base.to_numpy(float); wp=x.w_po4_base.to_numpy(float); w=x.area_weight.to_numpy(float); selections=[]; winners={}
    for s in SCENS:
        a=combined(model,pred_rows(x,s),s,wn,wp); um,uv=best(a); cm,cv=best(a,am); uw=np.where(um>=0,mp[np.maximum(um,0)],-1); cw=np.where(cm>=0,mp[np.maximum(cm,0)],-1); winners[s]=uw; selections += [selection(uw,collapse(uv,mp,len(t)),w,t,s,"unconstrained"),selection(cw,collapse(cv,mp,len(t)),w,t,s,"constrained")]
        cov=pool.set_index("province_key").pool_trait_coverage.reindex([key(v) for v in x.meow_province]).to_numpy(float); gap=np.full(len(x),np.nan); valid=(uw>=0)&t.trait_resolved.to_numpy(bool)[np.maximum(uw,0)]&(cov>=GATE); gap[valid]=gaps["overall"][uw[valid],code[valid]]; x[f"trait_substitution_gap_{s}"]=gap; x[f"exact_trait_match_{s}"]=np.isfinite(gap)&(gap<=EPS)
    sel=pd.concat(selections,ignore_index=True); sel.to_csv(OUT/"species_selection_advantage.csv",index=False); trait_models(sel,t,meta,rng).to_csv(OUT/"specific_trait_selection_models.csv",index=False); blocks=block_table(x,pool,code,gaps,winners); blocks.to_csv(OUT/"province_block_trait_metrics.csv",index=False); nested(blocks,rng).to_csv(OUT/"nested_constraint_loss_models.csv",index=False); x.to_csv(OUT/"trait_alignment_pixel_metrics.csv.gz",index=False,compression="gzip")
    valid=np.isfinite(x.trait_substitution_gap_baseline); summary={"n_biological_species":len(t),"n_trait_resolved_species":int(t.trait_resolved.sum()),"primary_pool_coverage_gate":GATE,"baseline_area_fraction_retained":float(np.sum(w[valid])/np.sum(w[np.isfinite(w)&(w>0)])),"permutations":NPERM,"province_cluster_bootstrap":NBOOT}; (OUT/"trait_constraint_summary.json").write_text(json.dumps(summary,indent=2))
if __name__=="__main__": main()
