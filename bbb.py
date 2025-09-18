# -*- coding: utf-8 -*-
# app.py — single-file verze (herní + běžecká data, vyčištěno a zkráceno)

import io, zipfile, unicodedata, re
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

# ========================= UI META =========================
st.set_page_config(page_title="Karty – Slavia (váhy + běžecká data)", layout="wide")
st.title("⚽ Generátor datových karet (váhový model + vyhledávání + běž. data)")

# ========================= I/O =============================
@st.cache_data
def load_xlsx(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))

# ========================= KONFIG ==========================
# Herní bloky/metriky
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

# Aliasy pro herní metriky
ALIASES = {
    "Cross accuracy, %": ["Accurate crosses, %","Cross accuracy, %"],
    "Progressive passes per 90": ["Progressive passes per 90","Progressive passes/90"],
    "Passes to final third per 90": ["Passes to final third per 90","Passes to final third/90"],
    "Dribbles per 90": ["Dribbles per 90","Dribbles/90"],
    "Progressive runs per 90": ["Progressive runs per 90","Progressive runs/90"],
    "Second assists per 90": ["Second assists per 90","Second assists/90"],
}

# Běžecké metriky + aliasy
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
    "Total distance per 90": ["Total distance per 90","Total distance/90","Distance per 90","Total distance (km) per 90","Distance P90"],
    "High-intensity runs per 90": ["High-intensity runs per 90","High intensity runs per 90","High intensity runs/90","HIR/90","HI Count P90"],
    "Sprints per 90": ["Sprints per 90","Sprints/90","Number of sprints per 90","Sprint Count P90"],
    "Max speed (km/h)": ["Max speed (km/h)","Top speed","Max velocity","Max speed","PSV-99","TOP 5 PSV-99"],
    "Average speed (km/h)": ["Average speed (km/h)","Avg speed","Average velocity","M/min P90"],
    "Accelerations per 90": ["Accelerations per 90","Accelerations/90","Accels per 90","High Acceleration Count P90 + Medium Acceleration Count P90"],
    "Decelerations per 90": ["Decelerations per 90","Decelerations/90","Decels per 90","High Deceleration Count P90 + Medium Deceleration Count P90"],
    "High-speed distance per 90": ["High-speed distance per 90","HS distance/90","High speed distance per 90","HSR Distance P90"],
}
CUSTOM_RUN_RENAME = {
    "Distance P90": "Total distance per 90",
    "HSR Distance P90": "High-speed distance per 90",
    "HI Count P90": "High-intensity runs per 90",
    "Sprint Count P90": "Sprints per 90",
    "High Acceleration Count P90": "High Acceleration Count P90",
    "Medium Acceleration Count P90": "Medium Acceleration Count P90",
    "High Deceleration Count P90": "High Deceleration Count P90",
    "Medium Deceleration Count P90": "Medium Deceleration Count P90",
    "PSV-99": "Max speed (km/h)",
    "Average speed (km/h)": "Average speed (km/h)",
}

# Pozice + peers
POS_REGEX = {
    "CB/DF": r"(CB|DF)", "RB": r"(RB)", "LB": r"(LB)",
    "WB/RWB/LWB": r"(WB|RWB|LWB)", "DM": r"(DM)", "CM": r"(CM)", "AM": r"(AM)",
    "RW": r"(RW)", "LW": r"(LW)", "CF/ST": r"(CF|ST|FW)",
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

# ========================= HELPERY =========================
def color_for(val):
    if pd.isna(val): return "lightgrey"
    if val <= 25: return "#FF4C4C"
    if val <= 50: return "#FF8C00"
    if val <= 75: return "#FFD700"
    return "#228B22"

def get_value_with_alias(row, key):
    if key in row.index: return row[key]
    for cand in ALIASES.get(key, []):
        if cand in row.index: return row[cand]
    if key == "Cross accuracy, %" and "Accurate crosses, %" in row.index:
        return row["Accurate crosses, %"]
    return np.nan

def get_pos_col(df: pd.DataFrame):
    if df is None: return None
    for c in ["Position", "Pos", "position", "Role", "Primary position"]:
        if c in df.columns: return c
    return None

def get_player_col(df: pd.DataFrame):
    if df is None: return None
    for c in ["Player", "Name", "player", "name", "Short Name"]:
        if c in df.columns: return c
    return None

def ensure_run_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    if "Metric" in df.columns and "Value" in df.columns:
        pcol = get_pos_col(df)
        plcol = get_player_col(df) or "Player"
        idx_cols = [c for c in [plcol, "Team", pcol, "Age"] if c and c in df.columns]
        wide = df.pivot_table(index=idx_cols, columns="Metric", values="Value", aggfunc="mean").reset_index()
        if plcol != "Player" and plcol in wide.columns: wide = wide.rename(columns={plcol:"Player"})
        if pcol and pcol != "Position" and pcol in wide.columns: wide = wide.rename(columns={pcol:"Position"})
        return wide
    pcol = get_pos_col(df)
    if pcol and pcol != "Position": df = df.rename(columns={pcol:"Position"})
    plcol = get_player_col(df)
    if plcol and plcol != "Player": df = df.rename(columns={plcol:"Player"})
    return df

def series_for_alias_run(df: pd.DataFrame, eng_key: str):
    if df is None or df.empty: return None
    if eng_key in df.columns: return df[eng_key]
    for cand in ALIASES_RUN.get(eng_key, []):
        if cand in df.columns: return df[cand]
    return None

def value_with_alias_run(row, key):
    if key in row.index: return row[key]
    for cand in ALIASES_RUN.get(key, []):
        if cand in row.index: return row[cand]
    return np.nan

def _strip_accents_lower(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join([c for c in s if not unicodedata.combining(c)])
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _best_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns: return n
    return None

def auto_fix_run_df(run_df: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
    if run_df is None or run_df.empty: return run_df
    # přejmenování běžeckých sloupců + ID
    run_df = run_df.rename(columns=CUSTOM_RUN_RENAME).copy()
    id_map = {}
    if "Player" not in run_df.columns:
        c = _best_col(run_df, ["Name","player","name","Short Name"])
        if c: id_map[c] = "Player"
    if "Team" not in run_df.columns:
        c = _best_col(run_df, ["Club","team","Team"])
        if c: id_map[c] = "Team"
    if "Position" not in run_df.columns:
        c = _best_col(run_df, ["Pos","Role","Primary position","position"])
        if c: id_map[c] = "Position"
    if "Age" not in run_df.columns:
        c = _best_col(run_df, ["Age (years)","age"])
        if c: id_map[c] = "Age"
    run_df = run_df.rename(columns=id_map)

    # long -> wide (pokud je)
    run_df = ensure_run_wide(run_df)

    # dopočty
    if "Average speed (km/h)" not in run_df.columns and "M/min P90" in run_df.columns:
        run_df["Average speed (km/h)"] = pd.to_numeric(run_df["M/min P90"], errors="coerce") * 0.06
    if "Accelerations per 90" not in run_df.columns:
        parts = []
        for c in ["High Acceleration Count P90","Medium Acceleration Count P90"]:
            if c in run_df.columns: parts.append(pd.to_numeric(run_df[c], errors="coerce"))
        if parts:
            s = parts[0]
            for a in parts[1:]: s = s.add(a, fill_value=0)
            run_df["Accelerations per 90"] = s
    if "Decelerations per 90" not in run_df.columns:
        parts = []
        for c in ["High Deceleration Count P90","Medium Deceleration Count P90"]:
            if c in run_df.columns: parts.append(pd.to_numeric(run_df[c], errors="coerce"))
        if parts:
            s = parts[0]
            for a in parts[1:]: s = s.add(a, fill_value=0)
            run_df["Decelerations per 90"] = s

    # doplnění Position z herních dat (pokud chybí)
    if "Position" not in run_df.columns and game_df is not None and not game_df.empty:
        g = game_df.copy()
        if "Player" not in g.columns:
            pcol = _best_col(g, ["Name","player","name"])
            if pcol: g = g.rename(columns={pcol:"Player"})
        if "Position" in g.columns and "Player" in g.columns:
            g_small = g[["Player","Position"]].dropna().groupby("Player", as_index=False).agg({"Position":"first"})
            run_df["_k"] = run_df["Player"].astype(str).map(_strip_accents_lower)
            g_small["_k"] = g_small["Player"].astype(str).map(_strip_accents_lower)
            run_df = run_df.merge(g_small[["_k","Position"]], on="_k", how="left", suffixes=("","_from_game")).drop(columns=["_k"])
            if "Position" not in run_df.columns and "Position_from_game" in run_df.columns:
                run_df = run_df.rename(columns={"Position_from_game":"Position"})

    for c in ["Player","Team","Position"]:
        if c in run_df.columns: run_df[c] = run_df[c].astype(str).str.strip()
    return run_df

def find_run_row_by_player(run_df: pd.DataFrame, player_name: str):
    if run_df is None or run_df.empty or not player_name: return None
    pcol = get_player_col(run_df) or "Player"
    hits = run_df.loc[run_df[pcol].astype(str) == str(player_name)]
    return None if hits.empty else hits.iloc[0]

# ========================= MODELY ==========================
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
    for _, lst, key in blocks:
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

def compute_overall_for_row(row, cz_agg, sec_weights, metric_weights, blocks=blocks):
    scores, sec_idx = compute_section_scores(row, cz_agg, blocks, metric_weights)
    overall = weighted_role_index(sec_idx, sec_weights)
    return scores, sec_idx, overall

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
    if cz_run_agg is None or cz_run_agg.empty: return {RUN_KEY:{}}, {}, np.nan
    run_scores, run_abs = {}, {}
    for eng, label in RUN:
        val_abs = value_with_alias_run(player_row, eng)
        if pd.isna(val_abs) and eng == "Average speed (km/h)" and "M/min P90" in player_row.index:
            val_abs = pd.to_numeric(player_row["M/min P90"], errors="coerce") * 0.06
        run_abs[label] = val_abs if not pd.isna(val_abs) else np.nan
        run_scores[label] = normalize_run_metric(cz_run_agg, eng, val_abs)
    vals = [v for v in run_scores.values() if not pd.isna(v)]
    run_index = float(np.mean(vals)) if vals else np.nan
    return {RUN_KEY: run_scores}, run_abs, run_index

def run_index_for_row(row, cz_run_df_pos):
    if cz_run_df_pos is None or cz_run_df_pos.empty: return np.nan, {}, {}
    plc = get_player_col(cz_run_df_pos) or "Player"
    cz_tmp = cz_run_df_pos.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_run_df_pos.columns else cz_run_df_pos
    cz_run_agg = cz_tmp.groupby("Player").mean(numeric_only=True)
    run_scores, run_abs, run_idx = compute_run_scores(row, cz_run_agg)
    return run_idx, run_scores, run_abs

def peers_for_pos_group(pos_group: str):
    return SLAVIA_PEERS.get(pos_group, [])

def avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights, blocks=blocks):
    peers = peers_for_pos_group(pos_group)
    vals=[]
    for nm in peers:
        if nm not in cz_agg.index: continue
        r = cz_agg.loc[nm]
        row_like = r.copy()
        row_like["Player"]=nm
        row_like["Team"]=row_like.get("Team","Slavia Praha")
        row_like["Position"]=pos_group
        row_like["Age"]=row_like.get("Age", np.nan)
        _, _, overall = compute_overall_for_row(row_like, cz_agg, sec_weights, metric_weights, blocks)
        if not np.isnan(overall): vals.append(overall)
    return float(np.mean(vals)) if vals else np.nan

# ========================= RENDER ==========================
def render_card_visual(player, team, pos, age,
                       scores, sec_index, overall, verdict,
                       run_scores=None, run_abs=None, run_index=np.nan, final_index=None, w_run=0.0):
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("off")
    ax.text(0.02,0.96, f"{player} (věk {age})", fontsize=20, fontweight="bold", va="top", color="black")
    ax.text(0.02,0.93, f"Klub: {team}   Pozice: {pos}", fontsize=13, va="top", color="black")

    # levý sloupec – 4 herní sekce
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
            ax.text(x+0.005,y-0.001,f"{label}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",
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
            txt_abs = "n/a" if pd.isna(val_abs) else (f"{float(val_abs):.2f}" if isinstance(val_abs,(int,float,np.number)) else str(val_abs))
            txt_pct = "n/a" if pd.isna(val_pct) else f"{int(round(val_pct))}%"
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
        ax.text(0.56,y-0.005,f"{key_disp}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",
                fontsize=13,va="center",ha="left",color="black")
        y -= 0.075

    if not pd.isna(run_index):
        c_run = color_for(run_index)
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c_run,alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"Běžecký index: {int(round(run_index))}%",fontsize=13,va="center",ha="left",color="black")
        y -= 0.075

    label_total = "Celkový role-index (vážený)" if (final_index is None) else "Celkový index (herní + běžecký)"
    value_total = overall if (final_index is None) else final_index
    c_over = color_for(value_total)
    ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c_over,alpha=0.7,lw=0))
    ax.text(0.56,y-0.005,f"{label_total}: {'n/a' if pd.isna(value_total) else str(int(round(value_total)))+'%'}",
            fontsize=14,fontweight="bold",va="center",ha="left",color="black")

    ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color='lightgrey',alpha=0.35,lw=0))
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center",color="black")
    return fig

# ========================= SIDEBAR =========================
st.sidebar.header("⚙️ Váhy sekcí")
default_weights = {"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
sec_weights = {k: st.sidebar.slider(k, 0, 100, default_weights[k], 1) for k in default_weights}
tot = sum(sec_weights.values()) or 1
for k in sec_weights: sec_weights[k] = 100.0 * sec_weights[k] / tot

with st.sidebar.expander("Váhy metrik v sekcích (volitelné)", expanded=False):
    metric_weights = {}
    for _, lst, key in blocks:
        tmp = {label: st.slider("– "+label, 0, 100, 10, 1, key=f"{key}_{label}") for _, label in lst}
        ssum = sum(tmp.values())
        metric_weights[key] = None if ssum==0 else {lab: w/ssum for lab, w in tmp.items()}

# ========================= TABS ============================
tab_card, tab_search = st.tabs(["Karta hráče", "Vyhledávání hráčů"])

# ---------------- TAB 1: KARTA -----------------------------
with tab_card:
    c1, c2 = st.columns(2)
    with c1:
        league_file = st.file_uploader("CZ liga – herní data (xlsx)", type=["xlsx"], key="league_card")
        run_cz_file = st.file_uploader("CZ běžecká data (xlsx)", type=["xlsx"], key="run_cz_card")
    with c2:
        players_file = st.file_uploader("Hráč/hráči – herní data (xlsx)", type=["xlsx"], key="players_card")
        run_players_file = st.file_uploader("Hráč/hráči – běžecká data (xlsx) [volitelné]", type=["xlsx"], key="run_players_card")

    if not league_file or not players_file:
        st.info("➡️ Nahraj minimálně CZ herní dataset + hráčský herní export."); st.stop()

    try:
        league = pd.read_excel(league_file)
        players = pd.read_excel(players_file)
    except Exception as e:
        st.error(f"Chyba při načítání herních souborů: {e}"); st.stop()

    run_cz_df = auto_fix_run_df(pd.read_excel(run_cz_file), league) if run_cz_file else None
    run_players_df = auto_fix_run_df(pd.read_excel(run_players_file), players) if run_players_file else None

    player_names = players["Player"].dropna().unique().tolist()
    sel_player = st.selectbox("Vyber hráče (herní export)", player_names)

    row = players.loc[players["Player"] == sel_player].iloc[0]
    player = row.get("Player",""); team = row.get("Team",""); pos = row.get("Position","")
    pos_group = resolve_pos_group(str(pos)); rgx = POS_REGEX[pos_group]
    group = league[league["Position"].astype(str).str.contains(rgx, na=False, regex=True)].copy()
    agg = group.groupby("Player").mean(numeric_only=True)

    scores, block_idx, overall = compute_overall_for_row(row, agg, sec_weights, metric_weights, blocks)

    # běžecký index
    run_scores = None; run_abs=None; run_index=np.nan; row_run=None
    if (run_cz_df is not None) and (run_players_df is not None):
        posc = get_pos_col(run_cz_df)
        cz_run_pos = run_cz_df[run_cz_df[posc].astype(str).str.contains(rgx, na=False, regex=True)] if posc else pd.DataFrame()
        if not cz_run_pos.empty:
            row_run = find_run_row_by_player(run_players_df, player)
            if row_run is not None:
                run_index, run_scores, run_abs = run_index_for_row(row_run, cz_run_pos)

        with st.expander("Kontrola běžeckých dat", expanded=False):
            cz_agg_tmp = None
            if not cz_run_pos.empty:
                plc = get_player_col(cz_run_pos) or "Player"
                cz_tmp = cz_run_pos.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_run_pos.columns else cz_run_pos
                cz_agg_tmp = cz_tmp.groupby("Player").mean(numeric_only=True)
            miss_cz = [lab for eng,lab in RUN if series_for_alias_run(cz_agg_tmp, eng) is None] if cz_agg_tmp is not None else ["—"]
            if row_run is not None:
                miss_pl = [lab for eng,lab in RUN if pd.isna(value_with_alias_run(row_run, eng))]
            else:
                miss_pl = ["— (hráč nenalezen v běžeckých datech)"]
            st.write(f"Chybějící metriky v CZ benchmarku: {', '.join(miss_cz) if miss_cz else '—'}")
            st.write(f"Chybějící metriky u hráče: {', '.join(miss_pl) if miss_pl else '—'}")
            present = 0
            if run_scores and RUN_KEY in run_scores:
                for _, lab in RUN:
                    v = run_scores[RUN_KEY].get(lab, np.nan)
                    if not pd.isna(v): present += 1
            st.write(f"Metrik započteno do Run indexu: {present}/8")
            if present <= 4: st.warning("Běžecké hodnocení je málo spolehlivé (≤ 4 metrik).")

    peer_avg = avg_peer_index(agg, pos_group, sec_weights, metric_weights, blocks)
    verdict = ("ANO – potenciální posila do Slavie"
               if (not np.isnan(peer_avg) and not np.isnan(overall) and overall >= peer_avg)
               else "NE – nedosahuje úrovně slávistických konkurentů")

    fig = render_card_visual(player, team, pos, row.get("Age","n/a"),
                             scores, block_idx, overall, verdict,
                             run_scores=run_scores, run_abs=run_abs, run_index=run_index)
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    st.download_button("📥 Stáhnout kartu jako PNG", data=buf.getvalue(), file_name=f"{player}.png", mime="image/png")

# ---------------- TAB 2: SEARCH ----------------------------
with tab_search:
    st.subheader("Vyhledávání kandidátů pro Slavii (benchmark = CZ liga)")

    colA, colB = st.columns(2)
    with colA:
        cz_file = st.file_uploader("CZ liga – herní (xlsx)", type=["xlsx"], key="cz_search")
        run_cz_file = st.file_uploader("CZ běžecká (xlsx) [volitelné]", type=["xlsx"], key="cz_run_search")
    with colB:
        fr_file = st.file_uploader("Cizí liga – herní (xlsx)", type=["xlsx"], key="fr_search")
        run_fr_file = st.file_uploader("Cizí liga – běžecká (xlsx) [volit.]", type=["xlsx"], key="fr_run_search")

    if cz_file: st.session_state["cz_bytes"] = cz_file.getvalue()
    if fr_file: st.session_state["fr_bytes"] = fr_file.getvalue()
    if run_cz_file: st.session_state["cz_run_bytes"] = run_cz_file.getvalue()
    if run_fr_file: st.session_state["fr_run_bytes"] = run_fr_file.getvalue()

    all_pos_opts = list(POS_REGEX.keys())
    positions_selected = st.multiselect("Pozice", all_pos_opts, default=all_pos_opts, key="search_positions")

    c1,c2,c3 = st.columns(3)
    with c1:
        league_name = st.text_input("Název ligy (zobrazí se ve výstupu)", value="Cizí liga", key="search_league")
    with c2:
        min_minutes = st.number_input("Min. minut (pokud ve zdroji)", min_value=0, value=0, step=100, key="search_min_minutes")
    with c3:
        min_games = st.number_input("Min. zápasů (pokud ve zdroji)", min_value=0, value=0, step=1, key="search_min_games")

    w_run_pct = st.slider("Váha běžeckého indexu v celkovém hodnocení", 0, 50, 0, 5, key="w_run")
    w_run = w_run_pct / 100.0

    if st.button("Spustit vyhledávání", key="search_run"):
        if "cz_bytes" not in st.session_state or "fr_bytes" not in st.session_state:
            st.error("Nahraj alespoň CZ herní + cizí liga herní."); st.stop()

        cz_df = load_xlsx(st.session_state["cz_bytes"])
        fr_df = load_xlsx(st.session_state["fr_bytes"])
        cz_run_df = load_xlsx(st.session_state["cz_run_bytes"]) if "cz_run_bytes" in st.session_state else None
        fr_run_df = load_xlsx(st.session_state["fr_run_bytes"]) if "fr_run_bytes" in st.session_state else None

        cz_run_df = auto_fix_run_df(cz_run_df, cz_df) if cz_run_df is not None else None
        fr_run_df = auto_fix_run_df(fr_run_df, fr_df) if fr_run_df is not None else None

        def search_candidates(cz_df, foreign_df, positions_selected, sec_weights, metric_weights,
                              min_minutes=None, min_games=None, league_name="",
                              cz_run_df=None, fr_run_df=None, w_run:float=0.0):
            # Pozice (herní)
            mask_pos = pd.Series([False]*len(foreign_df))
            for p in positions_selected:
                rgx = POS_REGEX[p]
                mask_pos |= foreign_df["Position"].astype(str).str.contains(rgx, na=False, regex=True)
            base = foreign_df.loc[mask_pos].copy()

            def best_col(df, names):
                for n in names:
                    if n in df.columns: return n
                return None
            min_col = best_col(base, ["Minutes","Minutes played","Min"])
            games_col = best_col(base, ["Games","Matches"])
            if min_minutes is not None and min_col:
                base = base[pd.to_numeric(base[min_col], errors="coerce").fillna(0) >= min_minutes]
            if min_games is not None and games_col:
                base = base[pd.to_numeric(base[games_col], errors="coerce").fillna(0) >= min_games]

            rows=[]; cards=[]
            for _,r in base.iterrows():
                pos_group = resolve_pos_group(str(r.get("Position",""))); rgx = POS_REGEX[pos_group]
                cz_pos = cz_df[cz_df["Position"].astype(str).str.contains(rgx, na=False, regex=True)]
                if cz_pos.empty: 
                    continue
                cz_agg = cz_pos.groupby("Player").mean(numeric_only=True)

                scores, sec_idx, overall = compute_overall_for_row(r, cz_agg, sec_weights, metric_weights, blocks)

                run_idx = np.nan; run_scores=None; run_abs=None
                if (cz_run_df is not None) and (fr_run_df is not None):
                    posc = get_pos_col(cz_run_df)
                    cz_run_pos = cz_run_df[cz_run_df[posc].astype(str).str.contains(rgx, na=False, regex=True)] if posc else pd.DataFrame()
                    r_run = find_run_row_by_player(fr_run_df, r.get("Player",""))
                    if (not cz_run_pos.empty) and (r_run is not None):
                        run_idx, run_scores, run_abs = run_index_for_row(r_run, cz_run_pos)

                final_index = overall if (np.isnan(run_idx) or w_run<=0.0) else (1.0 - w_run)*overall + w_run*run_idx
                peer_avg = avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights, blocks)
                base_for_verdict = final_index
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
                        "Role-index (vážený)": overall,
                        "Run index": run_idx,
                        "Final index": final_index if (not np.isnan(run_idx) and w_run>0.0) else np.nan,
                        "Verdikt": verdict
                    })
                    fig = render_card_visual(player, team, pos, age,
                                             scores, sec_idx, overall, verdict,
                                             run_scores=run_scores, run_abs=run_abs, run_index=run_idx,
                                             final_index=(final_index if (not np.isnan(run_idx) and w_run>0.0) else None),
                                             w_run=w_run)
                    bio = BytesIO()
                    fig.savefig(bio, format="png", dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    cards.append((str(player), bio.getvalue()))
            return pd.DataFrame(rows), cards

        res_df, cards = search_candidates(
            cz_df, fr_df, positions_selected,
            sec_weights=sec_weights, metric_weights=metric_weights,
            min_minutes=min_minutes if min_minutes>0 else None,
            min_games=min_games if min_games>0 else None,
            league_name=league_name,
            cz_run_df=cz_run_df, fr_run_df=fr_run_df, w_run=w_run
        )

        st.session_state["search_results"] = res_df
        st.session_state["search_cards"] = cards
        st.session_state["fr_df"] = fr_df
        st.session_state["cz_df"] = cz_df
        st.session_state["fr_run_df"] = fr_run_df
        st.session_state["cz_run_df"] = cz_run_df

    # Výstup
    res_df = st.session_state.get("search_results")
    cards = st.session_state.get("search_cards")
    fr_df_cached = st.session_state.get("fr_df")
    cz_df_cached = st.session_state.get("cz_df")
    fr_run_df_cached = st.session_state.get("fr_run_df")
    cz_run_df_cached = st.session_state.get("cz_run_df")

    if res_df is None or res_df.empty:
        st.info("Zatím žádné výsledky – nahraj soubory a klikni na *Spustit vyhledávání*.")
    else:
        st.success(f"Nalezeno kandidátů: {len(res_df)}")
        st.dataframe(res_df, use_container_width=True)

        st.download_button(
            "📥 Stáhnout CSV s kandidáty",
            data=res_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"kandidati_{st.session_state.get('search_league','liga')}.csv",
            mime="text/csv",
            key="dl_csv"
        )

        zbuf = BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, png_bytes in (cards or []):
                safe = str(name).replace("/","_").replace("\\","_")
                zf.writestr(f"{safe}.png", png_bytes)
        st.download_button(
            "🗂️ Stáhnout všechny karty (ZIP)",
            data=zbuf.getvalue(),
            file_name=f"karty_{st.session_state.get('search_league','liga')}.zip",
            mime="application/zip",
            key="dl_zip"
        )

        sel = st.selectbox("Zobraz kartu hráče", res_df["Hráč"].tolist(), key="preview_player")
        if sel and fr_df_cached is not None and cz_df_cached is not None:
            r = fr_df_cached.loc[fr_df_cached["Player"]==sel].iloc[0]
            pos_group = resolve_pos_group(str(r.get("Position",""))); rgx = POS_REGEX[pos_group]
            cz_pos = cz_df_cached[cz_df_cached["Position"].astype(str).str.contains(rgx, na=False, regex=True)]
            if not cz_pos.empty:
                cz_agg = cz_pos.groupby("Player").mean(numeric_only=True)
                scores, sec_idx, overall = compute_overall_for_row(r, cz_agg, sec_weights, metric_weights, blocks)

                run_idx=np.nan; run_scores=None; run_abs=None
                if (cz_run_df_cached is not None) and (fr_run_df_cached is not None):
                    posc = get_pos_col(cz_run_df_cached)
                    cz_run_pos = cz_run_df_cached[cz_run_df_cached[posc].astype(str).str.contains(rgx, na=False, regex=True)] if posc else pd.DataFrame()
                    r_run = find_run_row_by_player(fr_run_df_cached, sel)
                    if (not cz_run_pos.empty) and (r_run is not None):
                        run_idx, run_scores, run_abs = run_index_for_row(r_run, cz_run_pos)

                final_index = None
                if (not np.isnan(run_idx)) and (w_run>0.0):
                    final_index = (1.0 - w_run)*overall + w_run*run_idx
                base_for_verdict = final_index if (final_index is not None) else overall
                peer_avg = avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights, blocks)
                verdict = ("ANO – potenciální posila do Slavie"
                           if (not np.isnan(peer_avg) and not np.isnan(base_for_verdict) and base_for_verdict >= peer_avg)
                           else "NE – nedosahuje úrovně slávistických konkurentů")

                fig = render_card_visual(
                    r.get("Player",""), r.get("Team",""), r.get("Position",""), r.get("Age","n/a"),
                    scores, sec_idx, overall, verdict,
                    run_scores=run_scores, run_abs=run_abs, run_index=run_idx,
                    final_index=final_index, w_run=w_run
                )
                st.pyplot(fig)
