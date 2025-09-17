
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

st.set_page_config(page_title="Data karty – Slavia standard", layout="wide")

st.title("⚽ Generátor datových karet (Slavia standard)")

# --- Color bands (final standard)
def color_for(val):
    if pd.isna(val): return "lightgrey"
    if val <= 25: return "#FF4C4C"   # red
    if val <= 50: return "#FF8C00"   # orange
    if val <= 75: return "#FFD700"   # yellow
    return "#228B22"                 # green

# --- Uploads
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

# --- Metric template (agreed standard)
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

def build_position_mask(df, pos):
    pos_str = str(pos)
    if any(k in pos_str for k in ["RB","RWB","WB","RW"]):
        return df["Position"].astype(str).str.contains("RB|RWB|WB|RW", na=False)
    if any(k in pos_str for k in ["LB","LWB"]):
        return df["Position"].astype(str).str.contains("LB|LWB|WB", na=False)
    if any(k in pos_str for k in ["CM","DM","AM","MF"]):
        return df["Position"].astype(str).str.contains("CM|DM|AM|MF", na=False)
    if any(k in pos_str for k in ["CB","DF"]):
        return df["Position"].astype(str).str.contains("CB|DF", na=False)
    if any(k in pos_str for k in ["CF","ST","FW"]):
        return df["Position"].astype(str).str.contains("CF|ST|FW", na=False)
    return df["Position"].notna()

def norm_0_100(series, value, lower_better=False):
    s = pd.to_numeric(series, errors="coerce").dropna()
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn, mx = s.min(), s.max()
    if mx == mn: return 50.0
    score = (v - mn) / (mx - mn) * 100.0
    if lower_better: score = 100.0 - score
    return float(np.clip(score, 0, 100))

# select player
player_names = players["Player"].dropna().unique().tolist()
sel_player = st.selectbox("Vyber hráče ze souboru hráčů", player_names)
row = players.loc[players["Player"] == sel_player].iloc[0]
player = row.get("Player",""); team = row.get("Team",""); pos = row.get("Position","")

# build league group
mask = build_position_mask(league, pos)
group = league.loc[mask].copy()
agg = group.groupby("Player").mean(numeric_only=True)

# compute normalized scores
missing = []
scores = {}
for title, lst, key in blocks:
    part = {}
    for eng, label in lst:
        val = get_value_with_alias(row, eng)
        if pd.isna(val):
            missing.append((key, label, "hodnota chybí ve vstupu hráče"))
            part[label] = np.nan
            continue
        # pick matching column in league dataset
        if eng in agg.columns:
            col = eng
        else:
            col = None
            for cand in ALIASES.get(eng, []):
                if cand in agg.columns:
                    col = cand; break
            if eng == "Cross accuracy, %" and "Accurate crosses, %" in agg.columns:
                col = "Accurate crosses, %"
        if col is None:
            missing.append((key, label, "kolona pro normalizaci v CZ lize nenalezena"))
            part[label] = np.nan
            continue
        part[label] = norm_0_100(agg[col], val)
    scores[key] = part

block_idx = {k: np.nanmean(list(v.values())) for k, v in scores.items()}
overall = float(np.nanmean(list(block_idx.values())))

# verdict vs Slavia
pos_str = str(pos)
if any(k in pos_str for k in ["RB","RWB","WB"]):
    slavia_peers = ["D. Douděra","D. Hashioka"]
elif "RW" in pos_str:
    slavia_peers = ["I. Schranz","Y. Sanyang","V. Kušej"]
elif any(k in pos_str for k in ["CM","DM","AM","MF"]):
    slavia_peers = ["C. Zafeiris","L. Provod","E. Prekop","O. Dorley","M. Sadílek","T. Holeš"]
elif any(k in pos_str for k in ["CB","DF"]):
    slavia_peers = ["I. Ogbu","D. Zima","T. Holeš","J. Bořil"]
elif any(k in pos_str for k in ["CF","ST","FW"]):
    slavia_peers = ["M. Chytil","T. Chorý"]
else:
    slavia_peers = []

def overall_for_name(name):
    if name not in agg.index: return None
    r = agg.loc[name]
    vals = {}
    for title, lst, key in blocks:
        part = {}
        for eng, label in lst:
            if eng in agg.columns:
                col = eng
            else:
                col = None
                for cand in ALIASES.get(eng, []):
                    if cand in agg.columns:
                        col = cand; break
                if eng == "Cross accuracy, %" and "Accurate crosses, %" in agg.columns:
                    col = "Accurate crosses, %"
            if (col is None) or (col not in r.index):
                part[label] = np.nan; continue
            part[label] = norm_0_100(agg[col], r[col])
        vals[key] = part
    bi = {k: np.nanmean(list(v.values())) for k, v in vals.items()}
    return float(np.nanmean(list(bi.values())))

peer_vals = [overall_for_name(nm) for nm in slavia_peers]
peer_vals = [v for v in peer_vals if v is not None and not np.isnan(v)]
avg_peer = float(np.mean(peer_vals)) if peer_vals else np.nan
verdict = ("ANO – potenciální posila do Slavie" if (not np.isnan(avg_peer) and overall >= avg_peer)
           else "NE – nedosahuje úrovně slávistických konkurentů")

# render card (two columns per section)
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

# right side
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
ax.text(0.56, y-0.005, f"Celkový role-index: {'n/a' if pd.isna(overall) else str(int(round(overall)))+'%'}",
        fontsize=14, fontweight="bold", va="center", ha="left", color="black")

ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color="lightgrey",alpha=0.35,lw=0))
ax.text(0.74,0.055, f"Verdikt: {verdict}", fontsize=12, ha="center", va="center", color="black")

st.pyplot(fig)

# warnings
if any(np.isnan(list(block_idx.values()))):
    st.warning("Některé metriky chybí či nemají shodu v lize – rozklikni 'Kontrola' níže.")

# show missing details
missing_rows = [m for m in missing]
if missing_rows:
    with st.expander("Kontrola: metriky k doplnění / aliasy", expanded=False):
        miss_df = pd.DataFrame(missing_rows, columns=["Sekce","Metrika","Stav"])
        st.dataframe(miss_df, use_container_width=True)

# download button
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
st.download_button("📥 Stáhnout kartu jako PNG", data=buf.getvalue(), file_name=f"{player}.png", mime="image/png")
