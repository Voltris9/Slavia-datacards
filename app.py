# app.py — Slavia datové karty (herní + vyhledávání + běžecká karta s verdiktem)
# -----------------------------------------------------------------------------
import re, unicodedata, io, zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

# -----------------------------------------------------------------------------
# ZÁKLADNÍ UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Karty – Slavia (herní + běžecká)", layout="wide")
st.title("⚽ Slavia – Generátor datových karet (herní + vyhledávání + běžecká)")

@st.cache_data
def load_xlsx(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))

def color_for(v):
    if pd.isna(v): return "lightgrey"
    if v <= 25: return "#FF4C4C"
    if v <= 50: return "#FF8C00"
    if v <= 75: return "#FFD700"
    return "#228B22"

# -----------------------------------------------------------------------------
# HERNÍ BLOKY + ALIASY
# -----------------------------------------------------------------------------
DEF = [
    ("Defensive duels per 90","Defenzivní duely /90"),
    ("Defensive duels won, %","Úspěšnost obr. duelů %"),
    ("Interceptions per 90","Interceptions /90"),
    ("Sliding tackles per 90","Sliding tackles /90"),
    ("Aerial duels won, %","Úspěšnost vzdušných %"),
    ("Fouls per 90","Fauly /90"),
]
OFF = [
    ("Goals per 90","Góly /90"),
    ("xG per 90","xG /90"),
    ("Shots on target, %","Střely na branku %"),
    ("Assists per 90","Asistence /90"),
    ("xA per 90","xA /90"),
    ("Shot assists per 90","Shot assists /90"),
]
PAS = [
    ("Accurate passes, %","Přesnost přihrávek %"),
    ("Key passes per 90","Klíčové přihrávky /90"),
    ("Smart passes per 90","Smart passes /90"),
    ("Progressive passes per 90","Progresivní přihrávky /90"),
    ("Passes to final third per 90","Do finální třetiny /90"),
    ("Cross accuracy, %","Úspěšnost centrů %"),
    ("Second assists per 90","Second assists /90"),
]
ONE = [
    ("Dribbles per 90","Driblingy /90"),
    ("Successful dribbles, %","Úspěšnost dribblingu %"),
    ("Offensive duels won, %","Úspěšnost of. duelů %"),
    ("Progressive runs per 90","Progresivní běhy /90"),
]
blocks = [("Defenziva", DEF, "Defenziva"),
          ("Ofenziva", OFF, "Ofenziva"),
          ("Přihrávky", PAS, "Přihrávky"),
          ("1v1", ONE, "1v1")]

ALIASES = {
    "Cross accuracy, %": ["Accurate crosses, %","Cross accuracy, %"],
    "Progressive passes per 90": ["Progressive passes per 90","Progressive passes/90"],
    "Passes to final third per 90": ["Passes to final third per 90","Passes to final third/90"],
    "Dribbles per 90": ["Dribbles per 90","Dribbles/90"],
    "Progressive runs per 90": ["Progressive runs per 90","Progressive runs/90"],
    "Second assists per 90": ["Second assists per 90","Second assists/90"],
}
def get_value_with_alias(row, key):
    if key in row.index: return row[key]
    for cand in ALIASES.get(key, []):
        if cand in row.index: return row[cand]
    if key == "Cross accuracy, %" and "Accurate crosses, %" in row.index:
        return row["Accurate crosses, %"]
    return np.nan

# -----------------------------------------------------------------------------
# BĚŽECKÉ METRIKY + ALIASY (+ auto normalizace názvů)
# -----------------------------------------------------------------------------
RUN = [
    ("Total distance per 90", "Total distance /90"),
    ("High-intensity runs per 90", "High-intensity runs /90"),
    ("Sprints per 90", "Sprints /90"),
    ("Max speed (km/h)", "Max speed (km/h)"),
    ("Average speed (km/h)", "Average speed (km/h)"),
    ("Accelerations per 90", "Accelerations /90"),
    ("Decelerations per 90", "Decelerations /90"),
    ("High-speed distance per 90", "High-speed distance /90"),
]
RUN_KEY = "Běh"

ALIASES_RUN = {
    "Total distance per 90": [
        "Total distance per 90","Total distance/90","Total distance /90",
        "Distance per 90","Total distance (km) per 90","Distance P90"
    ],
    "High-intensity runs per 90": [
        "High-intensity runs per 90","High intensity runs per 90",
        "High intensity runs/90","High intensity runs /90","HIR/90","HI Count P90"
    ],
    "Sprints per 90": [
        "Sprints per 90","Sprints/90","Sprints /90",
        "Number of sprints per 90","Sprint Count P90"
    ],
    "Max speed (km/h)": [
        "Max speed (km/h)","Top speed","Max velocity","Max speed","PSV-99","TOP 5 PSV-99"
    ],
    "Average speed (km/h)": [
        "Average speed (km/h)","Avg speed","Average velocity","M/min P90"
    ],
    "Accelerations per 90": [
        "Accelerations per 90","Accelerations/90","Accelerations /90","Accels per 90",
        "High Acceleration Count P90","Medium Acceleration Count P90",
        "High Acceleration Count P90 + Medium Acceleration Count P90"
    ],
    "Decelerations per 90": [
        "Decelerations per 90","Decelerations/90","Decelerations /90","Decels per 90",
        "High Deceleration Count P90","Medium Deceleration Count P90",
        "High Deceleration Count P90 + Medium Deceleration Count P90"
    ],
    "High-speed distance per 90": [
        "High-speed distance per 90","HS distance/90","HS distance /90",
        "High speed distance per 90","HSR Distance P90"
    ],
}

def _normalize_run_colname(name: str) -> str:
    if not isinstance(name, str): return name
    s = name.strip()
    s = re.sub(r"\s*/\s*90", "/90", s)
    s = re.sub(r"\s+per\s+90", " per 90", s, flags=re.I)
    s = re.sub(r"/90\b", " per 90", s)
    s = s.replace("High intensity", "High-intensity")
    return s

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy(); df.columns = [_normalize_run_colname(c) for c in df.columns]; return df

def _best_col(df, names):
    return next((n for n in names if n in df.columns), None)

def get_pos_col(df: pd.DataFrame):
    return _best_col(df, ["Position","Pos","position","Role","Primary position"])

def get_player_col(df: pd.DataFrame):
    return _best_col(df, ["Player","Name","player","name","Short Name"])

def ensure_run_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    if "Metric" in df.columns and "Value" in df.columns:
        pcol = get_pos_col(df); plcol = get_player_col(df) or "Player"
        idx = [c for c in [plcol, "Team", pcol, "Age"] if c and c in df.columns]
        wide = df.pivot_table(index=idx, columns="Metric", values="Value", aggfunc="mean").reset_index()
        if plcol != "Player" and plcol in wide.columns: wide = wide.rename(columns={plcol:"Player"})
        if pcol and pcol != "Position" and pcol in wide.columns: wide = wide.rename(columns={pcol:"Position"})
        return wide
    pcol = get_pos_col(df); plcol = get_player_col(df)
    if pcol and pcol!="Position": df = df.rename(columns={pcol:"Position"})
    if plcol and plcol!="Player": df = df.rename(columns={plcol:"Player"})
    return df

def series_for_alias_run(df: pd.DataFrame, eng_key: str):
    if df is None or df.empty: return None
    if eng_key in df.columns: return df[eng_key]
    for cand in ALIASES_RUN.get(eng_key, []):
        if cand in df.columns: return df[cand]
    return None

def value_with_alias_run(row: pd.Series, eng_key: str):
    if eng_key in row.index: return row[eng_key]
    for cand in ALIASES_RUN.get(eng_key, []):
        if cand in row.index: return row[cand]
    return np.nan

def auto_fix_run_df(run_df: pd.DataFrame, game_df: pd.DataFrame=None) -> pd.DataFrame:
    """Sjednotí názvy, konvertuje long->wide, dopočítá M/min->km/h a Acc/Dec, doplní ID sloupce/pozici (pokud je herní DF)."""
    if run_df is None or run_df.empty: return run_df
    run_df = _normalize_columns(run_df)
    run_df = ensure_run_wide(run_df)

    if "Average speed (km/h)" not in run_df.columns and "M/min P90" in run_df.columns:
        run_df["Average speed (km/h)"] = pd.to_numeric(run_df["M/min P90"], errors="coerce") * 0.06
    if "Accelerations per 90" not in run_df.columns:
        cols = [c for c in ["High Acceleration Count P90","Medium Acceleration Count P90"] if c in run_df.columns]
        if cols:
            s = pd.to_numeric(run_df[cols[0]], errors="coerce")
            for c in cols[1:]: s = s.add(pd.to_numeric(run_df[c], errors="coerce"), fill_value=0)
            run_df["Accelerations per 90"] = s
    if "Decelerations per 90" not in run_df.columns:
        cols = [c for c in ["High Deceleration Count P90","Medium Deceleration Count P90"] if c in run_df.columns]
        if cols:
            s = pd.to_numeric(run_df[cols[0]], errors="coerce")
            for c in cols[1:]: s = s.add(pd.to_numeric(run_df[c], errors="coerce"), fill_value=0)
            run_df["Decelerations per 90"] = s

    # doplnění ident sloupců
    id_map = {}
    if "Player" not in run_df.columns:
        c = _best_col(run_df, ["Name","player","name","Short Name"])
        if c: id_map[c] = "Player"
    if "Team" not in run_df.columns:
        c = _best_col(run_df, ["Club","team","Team"])
        if c: id_map[c] = "Team"
    if id_map: run_df = run_df.rename(columns=id_map)

    # doplnění Position z herních dat (pokud chybí)
    if "Position" not in run_df.columns and game_df is not None and not game_df.empty:
        g = game_df.copy()
        if "Player" not in g.columns:
            pcol = _best_col(g, ["Name","player","name"])
            if pcol: g = g.rename(columns={pcol:"Player"})
        if "Position" in g.columns and "Player" in g.columns:
            g2 = g[["Player","Position"]].dropna().groupby("Player", as_index=False).first()
            def _std(s): 
                if not isinstance(s,str): s=str(s)
                s = unicodedata.normalize("NFKD", s); s="".join(c for c in s if not unicodedata.combining(c))
                return re.sub(r"\s+"," ",s).strip().lower()
            run_df["_k"] = run_df["Player"].astype(str).map(_std)
            g2["_k"] = g2["Player"].astype(str).map(_std)
            run_df = run_df.merge(g2[["_k","Position"]], on="_k", how="left").drop(columns=["_k"])
    return run_df

# -----------------------------------------------------------------------------
# POZICE + SLAVIA PEERS
# -----------------------------------------------------------------------------
POS_REGEX = {
    "CB/DF": r"(CB|DF)", "RB": r"(RB)", "LB": r"(LB)", "WB/RWB/LWB": r"(WB|RWB|LWB)",
    "DM": r"(DM)", "CM": r"(CM)", "AM": r"(AM)", "RW": r"(RW)", "LW": r"(LW)", "CF/ST": r"(CF|ST|FW)",
}
def resolve_pos_group(pos_str: str) -> str:
    p = (pos_str or "").upper()
    if any(k in p for k in ["CB","DF"]): return "CB/DF"
    if "RB" in p: return "RB"
    if "LB" in p: return "LB"
    if any(k in p for k in ["RWB","LWB","WB"]): return "WB/RWB/LWB"
    if "DM" in p: return "DM"
    if "CM" in p: return "CM"
    if "AM" in p: return "AM"
    if "RW" in p: return "RW"
    if "LW" in p: return "LW"
    if any(k in p for k in ["CF","ST","FW"]): return "CF/ST"
    return "CM"

SLAVIA_PEERS = {
    "RB": ["D. Douděra","D. Hashioka"],
    "LB": ["O. Zmrzlý","J. Bořil"],
    "WB/RWB/LWB": ["D. Douděra","D. Hashioka","O. Zmrzlý"],
    "CB/DF": ["I. Ogbu","D. Zima","T. Holeš","J. Bořil"],
    "DM": ["T. Holeš","O. Dorley","M. Sadílek"],
    "CM": ["C. Zafeiris","L. Provod","E. Prekop","M. Sadílek"],
    "AM": ["C. Zafeiris","L. Provod","E. Prekop"],
    "RW": ["I. Schranz","Y. Sanyang","V. Kušej"],
    "LW": ["I. Schranz","V. Kušej"],
    "CF/ST": ["M. Chytil","T. Chorý"],
}
def peers_for_pos_group(pos_group: str):
    return SLAVIA_PEERS.get(pos_group, [])

# pro běžecký verdikt použijeme pevný seznam (musí existovat v CZ run benchmarku)
SLAVIA_RUN_PEERS = ["D. Douděra","O. Zmrzlý","I. Ogbu","T. Holeš","C. Zafeiris","L. Provod","M. Chytil","I. Schranz"]

# -----------------------------------------------------------------------------
# HERNÍ VÝPOČTY
# -----------------------------------------------------------------------------
def series_for_alias(agg: pd.DataFrame, eng_key: str):
    if eng_key in agg.columns: return agg[eng_key]
    for cand in ALIASES.get(eng_key, []):
        if cand in agg.columns: return agg[cand]
    if eng_key == "Cross accuracy, %" and "Accurate crosses, %" in agg.columns:
        return agg["Accurate crosses, %"]
    return None

def normalize_metric(agg: pd.DataFrame, eng_key: str, value):
    s = series_for_alias(agg, eng_key)
    if s is None or pd.isna(value): return np.nan
    s = pd.to_numeric(s, errors="coerce").dropna()
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn, mx = s.min(), s.max()
    if mx == mn: return 50.0
    return float(np.clip((v - mn) / (mx - mn) * 100.0, 0, 100))

def compute_section_scores(player_row: pd.Series, agg: pd.DataFrame, blocks, metric_weights=None):
    sec_scores, sec_index = {}, {}
    for title, lst, key in blocks:
        part = {}
        for eng, label in lst:
            val = get_value_with_alias(player_row, eng)
            part[label] = normalize_metric(agg, eng, val)
        sec_scores[key] = part
        if metric_weights and metric_weights.get(key):
            wsum = 0.0; acc = 0.0
            for label, w in metric_weights[key].items():
                v = part.get(label, np.nan)
                if not pd.isna(v): acc += v*w; wsum += w
            sec_index[key] = float(acc/wsum) if wsum>0 else np.nan
        else:
            vals = [v for v in part.values() if not pd.isna(v)]
            sec_index[key] = float(np.mean(vals)) if vals else np.nan
    return sec_scores, sec_index

def weighted_role_index(sec_index: dict, sec_weights: dict):
    totw = 0.0; acc = 0.0
    for sec in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v = sec_index.get(sec, np.nan)
        if not pd.isna(v):
            w = sec_weights.get(sec, 0)/100.0
            acc += v*w; totw += w
    return float(acc/totw) if totw>0 else np.nan

# -----------------------------------------------------------------------------
# BĚŽECKÉ VÝPOČTY
# -----------------------------------------------------------------------------
def normalize_run_metric(cz_agg: pd.DataFrame, eng_key: str, value):
    s = series_for_alias_run(cz_agg, eng_key)
    if s is None or pd.isna(value): return np.nan
    s = pd.to_numeric(s, errors="coerce").dropna()
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn, mx = s.min(), s.max()
    if mx == mn: return 50.0
    return float(np.clip((v - mn) / (mx - mn) * 100.0, 0, 100))

def compute_run_scores(player_row: pd.Series, cz_run_agg: pd.DataFrame):
    if cz_run_agg is None or cz_run_agg.empty:
        return {RUN_KEY:{}}, {}, np.nan
    run_scores, run_abs = {}, {}
    for eng, label in RUN:
        val_abs = value_with_alias_run(player_row, eng)
        if pd.isna(val_abs) and eng=="Average speed (km/h)" and "M/min P90" in player_row.index:
            val_abs = pd.to_numeric(player_row["M/min P90"], errors="coerce") * 0.06
        run_abs[label] = val_abs if not pd.isna(val_abs) else np.nan
        run_scores[label] = normalize_run_metric(cz_run_agg, eng, val_abs)
    vals = [v for v in run_scores.values() if not pd.isna(v)]
    run_index = float(np.mean(vals)) if vals else np.nan
    return {RUN_KEY: run_scores}, run_abs, run_index

# -----------------------------------------------------------------------------
# RENDER KARET
# -----------------------------------------------------------------------------
def render_card_visual(player, team, pos, age, scores, sec_index, overall, verdict,
                       run_scores=None, run_abs=None, run_index=np.nan, final_index=None):
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("off")
    ax.text(0.02,0.96, f"{player} (věk {age})", fontsize=20, fontweight="bold", va="top")
    ax.text(0.02,0.93, f"Klub: {team}   Pozice: {pos}", fontsize=13, va="top")

    # levý sloupec – 4 herní sekce
    y0=0.88
    for display_title, lst, key in blocks:
        ax.text(0.02,y0,display_title,fontsize=15,fontweight="bold",va="top")
        y=y0-0.04
        col_x_left = 0.04; col_x_right = 0.26
        for i,(_,label) in enumerate(lst):
            val = scores[key].get(label, np.nan)
            c = color_for(val); x = col_x_left if i%2==0 else col_x_right
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=c,alpha=0.85,lw=0))
            ax.text(x+0.005,y-0.001,f"{label}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",fontsize=9,va="center",ha="left")
            if i%2==1: y-=0.038
        y0 = y-0.025

    # běžecká sekce (pokud máme)
    if run_scores is not None and RUN_KEY in run_scores and len(run_scores[RUN_KEY])>0:
        ax.text(0.02,y0,"Běžecká data",fontsize=15,fontweight="bold",va="top")
        y = y0 - 0.04; col_x_left = 0.04; col_x_right = 0.26
        for i,(_,label) in enumerate(RUN):
            val_pct = run_scores[RUN_KEY].get(label, np.nan)
            val_abs = run_abs.get(label, np.nan) if run_abs else np.nan
            c = color_for(val_pct); x = col_x_left if i%2==0 else col_x_right
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=c,alpha=0.85,lw=0))
            txt_abs = "n/a" if pd.isna(val_abs) else (f"{val_abs:.2f}" if isinstance(val_abs,(int,float,np.number)) else str(val_abs))
            txt_pct = "n/a" if pd.isna(val_pct) else f"{int(round(val_pct))}%"
            ax.text(x+0.005,y-0.001,f"{label}: {txt_abs} ({txt_pct})",fontsize=9,va="center",ha="left")
            if i%2==1: y-=0.038
        y0 = y - 0.025

    # pravý sloupec – souhrny
    ax.text(0.55,0.9,"Souhrnné indexy (0–100 %) – vážené",fontsize=16,fontweight="bold",va="top")
    y=0.85
    for key_disp in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        val = sec_index.get(key_disp, np.nan); c = color_for(val)
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c,alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"{key_disp}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",fontsize=13,va="center",ha="left")
        y -= 0.075

    if not pd.isna(run_index):
        c_run = color_for(run_index)
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c_run,alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"Běžecký index: {int(round(run_index))}%",fontsize=13,va="center",ha="left")
        y -= 0.075

    label_total = "Celkový role-index (vážený)" if (final_index is None) else "Celkový index (herní + běžecký)"
    value_total = sec_overall if (final_index is None) else final_index  # (sec_overall je doplněn v místě volání)
    # POZOR: v místě volání nastavíme globální proměnnou sec_overall pro jednoduchost renderu
    ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(value_total),alpha=0.7,lw=0))
    ax.text(0.56,y-0.005,f"{label_total}: {'n/a' if pd.isna(value_total) else str(int(round(value_total)))+'%'}",
            fontsize=14,fontweight="bold",va="center",ha="left")

    ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color='lightgrey',alpha=0.35,lw=0))
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center")
    return fig

def render_run_only_card(player, team, age, run_scores, run_abs, run_index, verdict):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.text(0.02,0.94, f"{player} (věk {age})", fontsize=20, fontweight="bold", va="top")
    ax.text(0.02,0.91, f"Klub: {team}", fontsize=12, va="top")
    ax.text(0.02,0.86,"Běžecká data (vs. CZ benchmark)", fontsize=15, fontweight="bold", va="top")
    y=0.82; left=0.04; right=0.30
    for i,(_,label) in enumerate(RUN):
        pct = run_scores[RUN_KEY].get(label, np.nan); val = run_abs.get(label, np.nan)
        x = left if i%2==0 else right; c = color_for(pct)
        ax.add_patch(Rectangle((x,y-0.02),0.23,0.04,color=c,alpha=0.85,lw=0))
        txt_val = "n/a" if pd.isna(val) else (f"{val:.2f}" if isinstance(val,(int,float,np.number)) else str(val))
        txt_pct = "n/a" if pd.isna(pct) else f"{int(round(pct))}%"
        ax.text(x+0.005,y-0.002,f"{label}: {txt_val} ({txt_pct})",fontsize=10,va="center",ha="left")
        if i%2==1: y-=0.05
    ax.text(0.60,0.86,"Souhrn", fontsize=15, fontweight="bold", va="top")
    c_run = color_for(run_index)
    ax.add_patch(Rectangle((0.60,0.79),0.35,0.06,color=c_run,alpha=0.75,lw=0))
    ax.text(0.615,0.81,f"Běžecký index: {'n/a' if pd.isna(run_index) else str(int(round(run_index)))+'%'}",fontsize=14,va="center",ha="left")
    ax.add_patch(Rectangle((0.60,0.70),0.35,0.06,color='lightgrey',alpha=0.5,lw=0))
    ax.text(0.615,0.73,f"Verdikt: {verdict}", fontsize=13, va="center", ha="left")
    return fig

# -----------------------------------------------------------------------------
# POMOCNÍCI: overall + peers + run-index pro řádek
# -----------------------------------------------------------------------------
def compute_overall_for_row(row, cz_agg, sec_weights, metric_weights):
    scores, sec_idx = compute_section_scores(row, cz_agg, blocks, metric_weights)
    overall = weighted_role_index(sec_idx, sec_weights)
    return scores, sec_idx, overall

def avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights):
    vals=[]
    for nm in peers_for_pos_group(pos_group):
        if nm in cz_agg.index:
            r = cz_agg.loc[nm].copy()
            r["Player"]=nm; r["Position"]=pos_group
            _, _, overall = compute_overall_for_row(r, cz_agg, sec_weights, metric_weights)
            if not np.isnan(overall): vals.append(overall)
    return float(np.mean(vals)) if vals else np.nan

def run_index_for_row(row, cz_run_df_pos):
    if cz_run_df_pos is None or cz_run_df_pos.empty: return np.nan, {}, {}
    plc = get_player_col(cz_run_df_pos) or "Player"
    cz_tmp = cz_run_df_pos.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_run_df_pos.columns else cz_run_df_pos
    cz_run_agg = cz_tmp.groupby("Player").mean(numeric_only=True)
    return compute_run_scores(row, cz_run_agg)[2], compute_run_scores(row, cz_run_agg)[0], compute_run_scores(row, cz_run_agg)[1]

# -----------------------------------------------------------------------------
# SIDEBAR – VÁHY
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Váhy sekcí (herní)")
default_weights = {"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
sec_weights = {sec: st.sidebar.slider(sec, 0, 100, default_weights[sec], 1) for sec in default_weights}
tot = sum(sec_weights.values()) or 1
sec_weights = {k: 100.0*v/tot for k,v in sec_weights.items()}

with st.sidebar.expander("Váhy metrik (volitelné)"):
    metric_weights = {}
    for title, lst, key in blocks:
        st.markdown(f"**{title}**")
        tmp = {label: st.slider(f"– {label}", 0, 100, 10, 1, key=f"{key}_{label}") for _,label in lst}
        ssum = sum(tmp.values())
        metric_weights[key] = None if ssum==0 else {lab: w/ssum for lab,w in tmp.items()}

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_card, tab_search, tab_run = st.tabs(["Karta hráče (herní + běh)", "Vyhledávání hráčů", "Běžecká karta (jen běh)"])

# -----------------------------------------------------------------------------
# TAB 1: KARTA HRÁČE (herní + volitelně běh)
# -----------------------------------------------------------------------------
with tab_card:
    c1, c2 = st.columns(2)
    with c1:
        league_file = st.file_uploader("CZ liga – herní (xlsx)", type=["xlsx"], key="league_card")
        run_cz_file = st.file_uploader("CZ běžecká data (xlsx)", type=["xlsx"], key="run_cz_card")
    with c2:
        players_file = st.file_uploader("Hráč/hráči – herní (xlsx)", type=["xlsx"], key="players_card")
        run_players_file = st.file_uploader("Hráč/hráči – běžecká (xlsx) [volitelné]", type=["xlsx"], key="run_players_card")

    if not league_file or not players_file:
        st.info("➡️ Nahraj minimálně CZ herní dataset + hráčský herní export.")
        st.stop()

    league = pd.read_excel(league_file); players = pd.read_excel(players_file)
    run_cz_df = auto_fix_run_df(pd.read_excel(run_cz_file), league) if run_cz_file else None
    run_players_df = auto_fix_run_df(pd.read_excel(run_players_file), players) if run_players_file else None

    player_names = players["Player"].dropna().unique().tolist()
    sel_player = st.selectbox("Vyber hráče (herní export)", player_names)
    row = players.loc[players["Player"] == sel_player].iloc[0]
    player = row.get("Player",""); team = row.get("Team",""); pos = row.get("Position",""); age=row.get("Age","n/a")

    pos_group = resolve_pos_group(str(pos)); rgx = POS_REGEX[pos_group]
    group = league[league["Position"].astype(str).str.contains(rgx, na=False, regex=True)].copy()
    agg = group.groupby("Player").mean(numeric_only=True)

    scores, block_idx, sec_overall = compute_overall_for_row(row, agg, sec_weights, metric_weights)

    # běh (pokud je k dispozici)
    run_scores = None; run_abs=None; run_index=np.nan; final_index=None
    if (run_cz_df is not None) and (run_players_df is not None):
        posc = get_pos_col(run_cz_df)
        cz_run_pos = run_cz_df[run_cz_df[posc].astype(str).str.contains(rgx, na=False, regex=True)] if posc else pd.DataFrame()
        if not cz_run_pos.empty:
            plcol_run = get_player_col(run_players_df)
            row_run_candidates = run_players_df.loc[run_players_df[plcol_run] == player] if plcol_run else pd.DataFrame()
            if not row_run_candidates.empty:
                row_run = row_run_candidates.iloc[0]
                cz_tmp = cz_run_pos.copy()
                plc = get_player_col(cz_tmp) or "Player"
                if plc!="Player" and plc in cz_tmp.columns: cz_tmp = cz_tmp.rename(columns={plc:"Player"})
                cz_run_agg = cz_tmp.groupby("Player").mean(numeric_only=True)
                run_scores, run_abs, run_index = compute_run_scores(row_run, cz_run_agg)

    # verdikt vs. peers na herní straně
    peer_avg = avg_peer_index(agg, pos_group, sec_weights, metric_weights)
    base_for_verdict = sec_overall
    verdict = ("ANO – potenciální posila do Slavie"
               if (not np.isnan(peer_avg) and not np.isnan(base_for_verdict) and base_for_verdict >= peer_avg)
               else "NE – nedosahuje úrovně slávistických konkurentů")

    # vykreslení
    fig = render_card_visual(player, team, pos, age, scores, block_idx, sec_overall, verdict,
                             run_scores=run_scores, run_abs=run_abs, run_index=run_index, final_index=None)
    st.pyplot(fig)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    st.download_button("📥 Stáhnout kartu (PNG)", data=buf.getvalue(), file_name=f"{player}.png", mime="image/png")

# -----------------------------------------------------------------------------
# TAB 2: VYHLEDÁVÁNÍ KANDIDÁTŮ (herní + volitelně běh do final indexu)
# -----------------------------------------------------------------------------
with tab_search:
    cA, cB = st.columns(2)
    with cA:
        cz_file = st.file_uploader("CZ liga – herní (xlsx)", type=["xlsx"], key="cz_search")
        run_cz_file2 = st.file_uploader("CZ běžecká data (xlsx) [volitelné]", type=["xlsx"], key="cz_run_search")
    with cB:
        fr_file = st.file_uploader("Cizí liga – herní (xlsx)", type=["xlsx"], key="fr_search")
        run_fr_file = st.file_uploader("Cizí liga – běžecká (xlsx) [volitelné]", type=["xlsx"], key="fr_run_search")

    if cz_file: st.session_state["cz_bytes"] = cz_file.getvalue()
    if fr_file: st.session_state["fr_bytes"] = fr_file.getvalue()
    if run_cz_file2: st.session_state["cz_run_bytes"] = run_cz_file2.getvalue()
    if run_fr_file: st.session_state["fr_run_bytes"] = run_fr_file.getvalue()

    all_pos_opts = list(POS_REGEX.keys())
    positions_selected = st.multiselect("Pozice", all_pos_opts, default=all_pos_opts)

    c1,c2,c3 = st.columns(3)
    with c1: league_name = st.text_input("Název ligy (zobrazí se ve výstupu)", value="Cizí liga")
    with c2: min_minutes = st.number_input("Min. minut (pokud ve zdroji)", min_value=0, value=0, step=100)
    with c3: min_games = st.number_input("Min. zápasů (pokud ve zdroji)", min_value=0, value=0, step=1)
    w_run_pct = st.slider("Váha běžeckého indexu v celkovém hodnocení", 0, 50, 0, 5); w_run = w_run_pct/100.0

    if st.button("Spustit vyhledávání"):
        if "cz_bytes" not in st.session_state or "fr_bytes" not in st.session_state:
            st.error("Nahraj alespoň CZ herní + cizí liga herní."); st.stop()

        cz_df = load_xlsx(st.session_state["cz_bytes"])
        fr_df = load_xlsx(st.session_state["fr_bytes"])
        cz_run_df = auto_fix_run_df(load_xlsx(st.session_state["cz_run_bytes"]), cz_df) if "cz_run_bytes" in st.session_state else None
        fr_run_df = auto_fix_run_df(load_xlsx(st.session_state["fr_run_bytes"]), fr_df) if "fr_run_bytes" in st.session_state else None

        # filtr pozic + min minutes/games
        mask_pos = pd.Series(False, index=fr_df.index)
        for p in positions_selected:
            rgx = POS_REGEX[p]; mask_pos |= fr_df["Position"].astype(str).str.contains(rgx, na=False, regex=True)
        base = fr_df.loc[mask_pos].copy()
        def best_col(df, names): return next((n for n in names if n in df.columns), None)
        min_col = best_col(base, ["Minutes","Minutes played","Min"])
        games_col = best_col(base, ["Games","Matches"])
        if min_minutes and min_col: base = base[pd.to_numeric(base[min_col], errors="coerce").fillna(0) >= min_minutes]
        if min_games and games_col: base = base[pd.to_numeric(base[games_col], errors="coerce").fillna(0) >= min_games]

        rows, cards = [], []
        for _,r in base.iterrows():
            pos_group = resolve_pos_group(str(r.get("Position",""))); rgx = POS_REGEX[pos_group]
            cz_pos = cz_df[cz_df["Position"].astype(str).str.contains(rgx, na=False, regex=True)]
            if cz_pos.empty: continue
            cz_agg = cz_pos.groupby("Player").mean(numeric_only=True)

            scores, sec_idx, sec_overall = compute_overall_for_row(r, cz_agg, sec_weights, metric_weights)

            # běžecký index (pokud máme run data obou stran)
            run_idx = np.nan; run_scores=None; run_abs=None
            if (cz_run_df is not None) and (fr_run_df is not None):
                posc = get_pos_col(cz_run_df)
                cz_run_pos = cz_run_df[cz_run_df[posc].astype(str).str.contains(rgx, na=False, regex=True)] if posc else pd.DataFrame()
                if not cz_run_pos.empty:
                    plcol_run = get_player_col(fr_run_df)
                    r_run_cand = fr_run_df.loc[fr_run_df[plcol_run] == r.get("Player","")] if plcol_run else pd.DataFrame()
                    if not r_run_cand.empty:
                        r_run = r_run_cand.iloc[0]
                        cz_tmp = cz_run_pos.copy()
                        plc = get_player_col(cz_tmp) or "Player"
                        if plc!="Player" and plc in cz_tmp.columns: cz_tmp = cz_tmp.rename(columns={plc:"Player"})
                        cz_run_agg = cz_tmp.groupby("Player").mean(numeric_only=True)
                        run_scores, run_abs, run_idx = compute_run_scores(r_run, cz_run_agg)

            final_index = ((1.0 - w_run)*sec_overall + w_run*run_idx) if (w_run>0 and not pd.isna(run_idx)) else np.nan

            peer_avg = avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights)
            base_for_verdict = (final_index if (w_run>0 and not pd.isna(final_index)) else sec_overall)
            verdict = ("ANO – potenciální posila do Slavie"
                       if (not np.isnan(peer_avg) and not np.isnan(base_for_verdict) and base_for_verdict >= peer_avg)
                       else "NE – nedosahuje úrovně slávistických konkurentů")

            if verdict.startswith("ANO"):
                player = r.get("Player",""); team = r.get("Team",""); pos = r.get("Position",""); age = r.get("Age","n/a")
                rows.append({
                    "Hráč": player, "Věk": age, "Klub": team, "Pozice": pos, "Liga": league_name,
                    "Index Def": sec_idx.get("Defenziva", np.nan),
                    "Index Off": sec_idx.get("Ofenziva", np.nan),
                    "Index Pass": sec_idx.get("Přihrávky", np.nan),
                    "Index 1v1": sec_idx.get("1v1", np.nan),
                    "Role-index (vážený)": sec_overall,
                    "Run index": run_idx,
                    "Final index": final_index,
                    "Verdikt": verdict
                })
                # render karta PNG do ZIPu
                global sec_overall  # pro render_card_visual label (zjednodušení)
                fig = render_card_visual(player, team, pos, age, scores, sec_idx, sec_overall, verdict,
                                         run_scores=run_scores, run_abs=run_abs, run_index=run_idx,
                                         final_index=(final_index if (w_run>0 and not pd.isna(run_idx)) else None))
                bio = BytesIO(); fig.savefig(bio, format="png", dpi=180, bbox_inches="tight"); plt.close(fig)
                cards.append((str(player), bio.getvalue()))
        res_df = pd.DataFrame(rows)

        if res_df.empty:
            st.info("Nenalezeny žádné kandidáty podle filtrů.")
        else:
            st.success(f"Nalezeno kandidátů: {len(res_df)}")
            st.dataframe(res_df, use_container_width=True)

            st.download_button(
                "📥 Stáhnout CSV s kandidáty",
                data=res_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"kandidati_{league_name}.csv", mime="text/csv"
            )

            zbuf = BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, png_bytes in (cards or []):
                    safe = str(name).replace("/","_").replace("\\","_")
                    zf.writestr(f"{safe}.png", png_bytes)
            st.download_button("🗂️ Stáhnout všechny karty (ZIP)",
                               data=zbuf.getvalue(), file_name=f"karty_{league_name}.zip", mime="application/zip")

# -----------------------------------------------------------------------------
# TAB 3: BĚŽECKÁ KARTA (jen běh) + VERDIKT PRO SLAVII
# -----------------------------------------------------------------------------
with tab_run:
    colA, colB = st.columns(2)
    with colA:
        cz_run_file = st.file_uploader("CZ běžecká data (xlsx) – benchmark", type=["xlsx"], key="cz_run")
    with colB:
        any_run_file = st.file_uploader("Běžecká data – libovolná liga (xlsx)", type=["xlsx"], key="any_run")

    if not cz_run_file or not any_run_file:
        st.info("Nahraj prosím **oba** soubory: CZ benchmark + libovolná liga (běh).")
        st.stop()

    czb = auto_fix_run_df(pd.read_excel(cz_run_file))
    anyb = auto_fix_run_df(pd.read_excel(any_run_file))

    pl_col = get_player_col(anyb) or "Player"
    player_names = anyb[pl_col].dropna().astype(str).unique().tolist()
    sel_player = st.selectbox("Vyber hráče (běžecký export)", sorted(player_names))

    cz_tmp = czb.rename(columns={get_player_col(czb):"Player"}) if get_player_col(czb)!="Player" else czb
    cz_agg = cz_tmp.groupby("Player").mean(numeric_only=True)

    row_run = anyb.loc[anyb[pl_col]==sel_player]
    if row_run.empty: st.error("Hráč v běžeckých datech nenalezen."); st.stop()
    row_run = row_run.iloc[0]

    run_scores, run_abs, run_idx = compute_run_scores(row_run, cz_agg)

    # průměrný Run index „peerů“ Slavie
    peer_vals=[]
    for peer in SLAVIA_RUN_PEERS:
        if peer in cz_agg.index:
            r_peer = cz_agg.loc[peer]
            _, _, peer_idx = compute_run_scores(r_peer, cz_agg)
            if not pd.isna(peer_idx): peer_vals.append(peer_idx)
    peer_avg = float(np.mean(peer_vals)) if peer_vals else np.nan

    if not pd.isna(peer_avg) and not pd.isna(run_idx):
        verdict_run = "ANO – běžecky vhodný pro Slavii" if run_idx >= peer_avg else "NE – běžecky nestačí"
    else:
        verdict_run = "— (nelze vyhodnotit, chybí data)"

    # kontrola metrik
    with st.expander("Kontrola běžeckých dat", expanded=False):
        miss_cz = [lab for eng,lab in RUN if series_for_alias_run(cz_agg, eng) is None]
        miss_pl = [lab for eng,lab in RUN if pd.isna(value_with_alias_run(row_run, eng))]
        present = sum([0 if pd.isna(run_scores[RUN_KEY].get(lab,np.nan)) else 1 for _,lab in RUN])
        st.write(f"Chybějící metriky v CZ benchmarku: {', '.join(miss_cz) if miss_cz else '—'}")
        st.write(f"Chybějící metriky u hráče: {', '.join(miss_pl) if miss_pl else '—'}")
        st.write(f"Metrik započteno do Run indexu: {present}/8")
        if present <= 4:
            st.warning("Běžecké hodnocení je málo spolehlivé (≤ 4 metrik).")

    team = row_run.get("Team","—"); age = row_run.get("Age","n/a")
    fig = render_run_only_card(sel_player, team, age, run_scores, run_abs, run_idx, verdict_run)
    st.pyplot(fig)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    st.download_button("📥 Stáhnout běžeckou kartu (PNG)", data=buf.getvalue(),
                       file_name=f"{sel_player}_run.png", mime="image/png")

