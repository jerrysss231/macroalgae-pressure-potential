"""Shared publication-figure helpers."""
from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family":"Times New Roman","font.size":8,"axes.titlesize":9.5,
        "axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
        "legend.fontsize":7,"axes.linewidth":0.8,"axes.spines.top":False,
        "axes.spines.right":False,"svg.fonttype":"none","savefig.dpi":600,
    })


def load_land(path: Path):
    if not path.exists(): return None
    land=gpd.read_file(path)
    return land.set_crs("EPSG:4326") if land.crs is None else land.to_crs("EPSG:4326")


def world_axes(ax, land=None):
    if land is not None: land.plot(ax=ax,facecolor="0.93",edgecolor="0.65",linewidth=0.2,zorder=1)
    ax.set_xlim(-180,180); ax.set_ylim(-70,80); ax.set_xticks([-120,-60,0,60,120]); ax.set_yticks([-60,-30,0,30,60]); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")


def panel_label(ax, label: str) -> None:
    ax.text(-0.08,1.04,label,transform=ax.transAxes,fontweight="bold",fontsize=10,ha="left",va="bottom")


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix(".png"),dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(path.with_suffix(".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)
