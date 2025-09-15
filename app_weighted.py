import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

st.set_page_config(page_title="Karty – Slavia standard (váhy podle pozic)", layout="wide")

st.title("⚽ Generátor datových karet (váhový model podle pozic)")

# --- Color bands
def color_for(val):
    if pd.isna(val): return "lightgrey"
    if val <= 25: return "#FF4C4C"   # červená
    if val <= 50: return "#FF8C00"   # oranžová
    if val <= 75: return "#FFD700"   # žlutá
    return "#228B22"                 # zelená

# --- Uploady
col_up1, col_up2 = st.columns(2)
with col_up1:
    league_file = st.file_uploader("Nahraj ligovou databázi (xlsx) – CZ liga", type=["xlsx"], key="league")
with col_up2:
    players_file = st.file_uploader("Nahraj soubor s hráčem/hráči (xlsx) – export Wyscout", type=["xlsx"], key="players")

if not league_file or not players_file:
    st.info("➡️ Nahraj oba soubory (ligový dataset + hráčský export).")
    st.stop()

try:
    league = pd.read_excel(league_file)
    players = pd.read_excel(players_file)
except Exception as e:
    st.error(f"Chyba při načítání souborů: {e}")
    st.stop()

# --- Sekce a metriky
DEF = [
    ("Defensive duels per 90","Defenzivní duely /90"),
    ("Defensive duels won, %","Úspěšnost obr. duelů %"),
    ("Interceptions per 90","Interceptions /90"),
    ("Sliding tackles per 90","Sliding tackles /90"),
    ("Aerial duels won, %","Úspěšnost vzdušných %"),
    ("Fouls per 90","Fauly /90")
]
OFF = [
    ("Goals per 90","Góly /90"),
    ("xG per 90","xG /90"),
    ("Shots on target, %","Střely na branku %"),
    ("Assists per 90","Asistence /90"),
    ("xA per 90","xA /90"),
    ("Shot assists per 90","Shot assists /90")
]
PAS = [
    ("Accurate passes, %","Přesnost přihrávek %"),
    ("Key passes per 90","Klíčové přihrávky /90"),
    ("Smart passes per 90","Smart passes /90"),
    ("Progressive passes per 90","Progresivní přihrávky /90"),
    ("Passes to final third per 90","Do finální třetiny /90"),
    ("Cross accuracy, %","Úspěšnost centrů %"),
    ("Second assists per 90","Second assists /90")
]
ONE = [
    ("Dribbles per 90","Driblingy /90"),
    ("Successful dribbles, %","Úspěšnost dribblingu %"),
    ("Offensive duels won, %","Úspěšnost of. duelů %"),
    ("Progressive runs per 90","Progresivní běhy /90")
]
blocks = [("Defenziva", DEF, "Defenziva"),
          ("Ofenziva", OFF, "Ofenziva"),
          ("Přihrávky", PAS, "Přihrávky"),
          ("1v1", ONE, "1v1")]

# --- Aliasy (pro různé názvy metrik)
ALIASES = {
    "Cross accuracy, %": ["Accurate crosses, %","Cross accuracy, %"],
    "Progressive passes per 90": ["Progressive passes per 90","Progressive passes/90"],
    "Passes to final third per 90": ["Passes to final third per 90","Passes to final third/90"],
    "Dribbles per 90": ["Dribbles per 90","Dribbles/90"],
    "Progressive runs per 90": ["Progressive runs per 90","Progressive runs/90"],
    "Second assists per 90": ["Second assists per 90","Second assists/90"]
}
def get_value_with_alias(row, key):
    if key in row.index:
        return row[key]
    for cand in ALIASES.get(key, []):
        if cand in row.index:
            return row[cand]
    if key == "Cross accuracy, %" and "Accurate crosses, %" in row.index:
        return row["Accurate crosses, %"]
    return np.nan

# --- Výběr hráče
player_names = players["Player"].dropna().unique().tolist()
sel_player = st.selectbox("Vyber hráče ze souboru hráčů", player_names)

row = players.loc[players["Player"] == sel_player].iloc[0]
player = row.get("Player","")
team = row.get("Team","")
pos = row.get("Position","")

# --- Výpočet skóre metrik
def norm_0_100(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna()
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn, mx = s.min(), s.max()
    if mx == mn: return 50.0
    score = (v - mn) / (mx - mn) * 100.0
    return float(np.clip(score, 0, 100))

mask = league["Position"].astype(str).str.contains(pos.split()[0], na=False)
group = league.loc[mask].copy()
agg = group.groupby("Player").mean(numeric_only=True)

scores = {}
for title, lst, key in blocks:
    part = {}
    for eng, label in lst:
        val = get_value_with_alias(row, eng)
        if pd.isna(val):
            part[label] = np.nan
            continue
        if eng in agg.columns:
            col = eng
        else:
            col = None
            for cand in ALIASES.get(eng, []):
                if cand in agg.columns:
                    col = cand; break
        if col is None:
            part[label] = np.nan
            continue
        part[label] = norm_0_100(agg[col], val)
    scores[key] = part

# --- Váhy podle pozice (sekce)
st.sidebar.header("⚙️ Váhy sekcí")
default_weights = {"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
weights = {}
for sec in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
    weights[sec] = st.sidebar.slider(f"{sec}", 0, 100, default_weights[sec], 1)

# Normalizace vah
total_w = sum(weights.values())
for k in weights:
    weights[k] = 100.0 * weights[k] / total_w if total_w>0 else 0

# --- Souhrnné indexy
block_idx = {k: np.nanmean(list(v.values())) for k, v in scores.items()}
overall = sum(block_idx[k]*weights[k]/100.0 for k in block_idx if not pd.isna(block_idx[k]))

# --- Render karty
fig, ax = plt.subplots(figsize=(18, 12))
ax.axis("off")
ax.text(0.02,0.96, player, fontsize=20, fontweight="bold", va="top", color="black")
ax.text(0.02,0.93, f"Klub: {team}   Pozice: {pos}", fontsize=13, va="top", color="black")

y0 = 0.88
for display_title, lst, key in blocks:
    ax.text(0.02, y0, display_title, fontsize=15, fontweight="bold", va="top", color="black")
    y = y0 - 0.04
    col_x_left = 0.04; col_x_right = 0.26
    for i, (eng, label) in enumerate(lst):
        val = scores[key].get(label, np.nan)
        c = color_for(val)
        x = col_x_left if i % 2 == 0 else col_x_right
        ax.add_patch(Rectangle((x, y-0.018), 0.18, 0.034, color=c, alpha=0.85, lw=0))
        ax.text(x+0.005, y-0.001, f"{label}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",
                fontsize=9, va="center", ha="left", color="black")
        if i % 2 == 1:
            y -= 0.038
    y0 = y - 0.025

# Pravý sloupec
ax.text(0.55,0.9,"Souhrnné indexy (0–100 %)",fontsize=16,fontweight="bold",va="top",color="black")
y = 0.85
for key_disp in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
    val = block_idx[key_disp]; c = color_for(val)
    ax.add_patch(Rectangle((0.55, y-0.03), 0.38, 0.05, color=c, alpha=0.7, lw=0))
    ax.text(0.56, y-0.005, f"{key_disp}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",
            fontsize=13, va="center", ha="left", color="black")
    y -= 0.075

c_over = color_for(overall)
ax.add_patch(Rectangle((0.55, y-0.03), 0.38, 0.05, color=c_over, alpha=0.7, lw=0))
ax.text(0.56, y-0.005, f"Celkový role-index (vážený): {'n/a' if pd.isna(overall) else str(int(round(overall)))+'%'}",
        fontsize=14, fontweight="bold", va="center", ha="left", color="black")

st.pyplot(fig)

# --- Download
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
st.download_button("📥 Stáhnout kartu jako PNG", data=buf.getvalue(), file_name=f"{player}.png", mime="image/png")
