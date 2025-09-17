# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import RUN_KEY, RUN, blocks

def _to_float_safe(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def color_for(val):
    v = _to_float_safe(val)
    if np.isnan(v): return "lightgrey"
    if v <= 25: return "#FF4C4C"
    if v <= 50: return "#FF8C00"
    if v <= 75: return "#FFD700"
    return "#228B22"

def render_card_visual(player, team, pos, age,
                       scores, sec_index, overall, verdict,
                       run_scores=None, run_abs=None, run_index=np.nan, final_index=None, w_run=0.0):
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("off")
    ax.text(0.02,0.96, f"{player} (věk {age})", fontsize=20, fontweight="bold", va="top", color="black")
    ax.text(0.02,0.93, f"Klub: {team}   Pozice: {pos}", fontsize=13, va="top", color="black")

    # levý sloupec – 4 sekce
    y0=0.88
    for display_title, lst, key in blocks:
        ax.text(0.02,y0,display_title,fontsize=15,fontweight="bold",va="top",color="black")
        y=y0-0.04
        col_x_left = 0.04; col_x_right = 0.26
        for i,(_,label) in enumerate(lst):
            val = scores[key].get(label, np.nan)
            c = color_for(val)
            x = col_x_left if i%2==0 else col_x_right
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=c,alpha=0.85,lw=0))
            ax.text(x+0.005,y-0.001,f"{label}: {'n/a' if np.isnan(val) else str(int(round(val)))+'%'}",
                    fontsize=9,va="center",ha="left",color="black")
            if i%2==1: y-=0.038
        y0 = y-0.025

    # běžecká sekce
    if run_scores is not None and RUN_KEY in run_scores and len(run_scores[RUN_KEY])>0:
        ax.text(0.02,y0,"Běžecká data",fontsize=15,fontweight="bold",va="top",color="black")
        y = y0 - 0.04
        col_x_left = 0.04; col_x_right = 0.26
        for i,(_,label) in enumerate(RUN):
            val_pct = run_scores[RUN_KEY].get(label, np.nan)
            val_abs = run_abs.get(label, np.nan) if run_abs else np.nan
            c = color_for(val_pct)
            x = col_x_left if i%2==0 else col_x_right
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=c,alpha=0.85,lw=0))
            txt_abs = "n/a" if np.isnan(val_abs) else (f"{val_abs:.2f}" if isinstance(val_abs,(int,float,np.number)) else str(val_abs))
            txt_pct = "n/a" if np.isnan(val_pct) else f"{int(round(val_pct))}%"
            ax.text(x+0.005,y-0.001,f"{label}: {txt_abs} ({txt_pct})",fontsize=9,va="center",ha="left",color="black")
            if i%2==1: y-=0.038
        y0 = y - 0.025

    # pravý sloupec – souhrny
    ax.text(0.55,0.9,"Souhrnné indexy (0–100 %) – vážené",fontsize=16,fontweight="bold",va="top",color="black")
    y=0.85
    for key_disp in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        val = sec_index.get(key_disp, np.nan)
        c = color_for(val)
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c,alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"{key_disp}: {'n/a' if np.isnan(val) else str(int(round(val)))+'%'}",
                fontsize=13,va="center",ha="left",color="black")
        y -= 0.075

    if not np.isnan(run_index):
        c_run = color_for(run_index)
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c_run,alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"Běžecký index: {int(round(run_index))}%",fontsize=13,va="center",ha="left",color="black")
        y -= 0.075

    label_total = "Celkový role-index (vážený)" if (final_index is None) else "Celkový index (herní + běžecký)"
    value_total = overall if (final_index is None) else final_index
    c_over = color_for(value_total)
    ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c_over,alpha=0.7,lw=0))
    ax.text(0.56,y-0.005,f"{label_total}: {'n/a' if np.isnan(value_total) else str(int(round(value_total)))+'%'}",
            fontsize=14,fontweight="bold",va="center",ha="left",color="black")

    ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color='lightgrey',alpha=0.35,lw=0))
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center",color="black")
    return fig
