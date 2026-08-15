"""Plot trait-selection and functional-substitution summaries."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from macroalgae_repro.paths import ProjectPaths
from macroalgae_repro.plotting import set_publication_style,panel_label,save_figure
P=ProjectPaths.from_env(); D=P.output_dir/"trait_constraint_analysis"; OUT=P.output_dir/"manuscript_figures"
def main():
    set_publication_style(); traits=pd.read_csv(D/"specific_trait_selection_models.csv"); nested=pd.read_csv(D/"nested_constraint_loss_models.csv"); blocks=pd.read_csv(D/"province_block_trait_metrics.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.2)); a,b,c,d=axes.ravel()
    x=traits[traits.selection_type.eq("constrained")].pivot_table(index="trait",columns="scenario",values="trait_added_r2",aggfunc="mean"); im=a.imshow(x.to_numpy(),aspect="auto"); a.set_yticks(range(len(x))); a.set_yticklabels([z.replace('_',' ') for z in x.index]); a.set_xticks(range(len(x.columns))); a.set_xticklabels(x.columns,rotation=30,ha="right"); a.set_title("Trait-specific selection signal"); fig.colorbar(im,ax=a,label="Added R²")
    n=nested[nested.functional_dimension.eq("overall")]; forplot=n[n.model_step.ne("M0")]
    for comp,g in forplot.groupby("loss_component"): b.plot(g.model_step,g.delta_r2,marker="o",label=comp.replace('_',' '))
    b.axhline(0,lw=.7,c="0.5"); b.tick_params(axis="x",rotation=25); b.set_ylabel("ΔR²"); b.set_title("Nested functional models"); b.legend(frameon=False)
    q=blocks.groupby("province_key",as_index=False).agg(rao_q=("rao_q","first"),gap=("mean_trait_gap","mean"),exact=("exact_match_share","mean")); c.scatter(q.rao_q,q.gap,s=10,alpha=.65); c.set_xlabel("Rao's Q"); c.set_ylabel("Mean trait-substitution gap"); c.set_title("Functional breadth and substitution")
    d.scatter(q.rao_q,100*q.exact,s=10,alpha=.65); d.set_xlabel("Rao's Q"); d.set_ylabel("Exact-match share (%)"); d.set_title("Functional breadth and exact matches")
    for ax,label in zip(axes.ravel(),"ABCD"): panel_label(ax,label)
    fig.tight_layout(); save_figure(fig,OUT/"trait_and_functional_results")
if __name__=="__main__": main()
