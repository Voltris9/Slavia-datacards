# Streamlit app: Typologie hráče – automatický scouting report (CZ)
# ---------------------------------------------------------------
# Požadavky: pip install -r requirements.txt
# Spuštění:  streamlit run app_typologie_hrace_streamlit.py
# Vstupy:    1) Liga (CZE1.xlsx)  2) Profil hráče (xlsx) NEBO výběr hráče z ligového souboru
# Výstup:    Vizuály + stažitelný .docx scouting report

import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from docx import Document
from docx.shared import Inches
from datetime import datetime

st.set_page_config(page_title="Typologie hráče – Scouting report", layout="wide")
st.title("⚽ Typologie hráče – generátor scouting reportu (CZ)")

# -------------------- Pomocné funkce --------------------

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

# Poměr hráč / ligový průměr v %
def pct(player_val, league_val):
    if league_val is None or pd.isna(league_val) or league_val == 0 or pd.isna(player_val):
        return np.nan
    return (player_val / league_val) * 100.0

# Slovní pásma (méně procent, víc přehlednosti)
def label_band(v, thr_low=70, thr_high=130):
    if pd.isna(v):
        return "bez dat"
    if v < thr_low:
        return "podprůměr"
    if v > thr_high:
        return "nadprůměr"
    return "ligový standard"

# Mapování pozic do referenčních skupin
POSITION_GROUPS = {
    "CF": ["CF"],
    "Křídla/AMF": ["LW", "RW", "LWF", "RWF", "AMF"],
    "CM/DM": ["CMF", "DMF", "RCMF", "LCMF", "CDM", "DM"],
    "CB/FB": ["CB", "RCB", "LCB", "RB", "LB", "RWB", "LWB"],
    "GK": ["GK"]
}

def infer_group_from_position(pos_text: str) -> str:
    pos_text = str(pos_text or "").upper()
    for group, tokens in POSITION_GROUPS.items():
        if any(tok in pos_text for tok in tokens):
            return group
    return "Křídla/AMF"  # default pro OF profily

# -------------------- UI – vstupy --------------------
colA, colB = st.columns([1,1])
with colA:
    liga_file = st.file_uploader("Nahraj ligový soubor (CZE1.xlsx)", type=["xlsx"]) 
with colB:
    mode = st.radio("Jak zadáš hráče?", ["Vyberu z ligového souboru", "Nahraju samostatný soubor hráče (xlsx)"])

player_df = None
league_df = None

if liga_file is not None:
    league_df = pd.read_excel(liga_file)

    if mode == "Vyberu z ligového souboru":
        jmena = league_df.get("Player", pd.Series(dtype=str)).astype(str).tolist()
        vyber_jmeno = st.selectbox("Vyber hráče ze souboru ligy:", sorted(jmena))
        player_df = league_df.loc[league_df["Player"].astype(str)==vyber_jmeno].copy()
    else:
        player_file = st.file_uploader("Nahraj soubor konkrétního hráče (xlsx)", type=["xlsx"]) 
        if player_file is not None:
            player_df = pd.read_excel(player_file)

# -------------------- Zpracování --------------------
if league_df is not None and player_df is not None and len(player_df) > 0:
    player_row = player_df.iloc[0]

    # Základní metadata
    jmeno = str(player_row.get("Player", "Neznámý"))
    klub = str(player_row.get("Team", ""))
    pozice_raw = str(player_row.get("Position", ""))
    vek = safe_float(player_row.get("Age", np.nan))
    minuty = safe_float(player_row.get("Minutes played", np.nan))

    # Zvolená referenční skupina pro ligový průměr
    default_group = infer_group_from_position(pozice_raw)
    zvolena_skup = st.selectbox(
        "Referenční skupina pro srovnání (pozice):",
        list(POSITION_GROUPS.keys()),
        index=list(POSITION_GROUPS.keys()).index(default_group)
    )

    # Filtrování ligového průměru podle skupiny
    tokens = POSITION_GROUPS[zvolena_skup]
    mask = league_df["Position"].astype(str).str.upper().apply(lambda s: any(tok in s for tok in tokens))
    league_pos = league_df.loc[mask].copy()
    league_means = league_pos.mean(numeric_only=True)

    # Metriky pro RADAR a HEATMAPU (v % vůči průměru na pozici)
    radar_map = {
        "Ofenzivní duely vyhrané %": "Offensive duels won, %",
        "Defenzivní duely vyhrané %": "Defensive duels won, %",
        "Hlavičkové souboje vyhrané %": "Aerial duels won, %",
        "Úspěšnost driblinků %": "Successful dribbles, %",
        "Úspěšnost centrů %": "Accurate crosses, %",
        "Góly /90": "Goals per 90",
        "Asistence /90": "Assists per 90",
    }

    heat_map = {
        "Ofenzivní duely vyhrané %": "Offensive duels won, %",
        "Defenzivní duely vyhrané %": "Defensive duels won, %",
        "Hlavičkové souboje vyhrané %": "Aerial duels won, %",
        "Úspěšnost přihrávek celkem %": "Accurate passes, %",
        "Úspěšnost driblinků %": "Successful dribbles, %",
        "Úspěšnost centrů %": "Accurate crosses, %",
        "Góly /90": "Goals per 90",
        "Asistence /90": "Assists per 90",
    }

    # Výpočty % vs liga
    radar_labels, radar_vals = [], []
    for lab, col in radar_map.items():
        radar_labels.append(lab)
        radar_vals.append(pct(safe_float(player_row.get(col, np.nan)), safe_float(league_means.get(col, np.nan))))

    heat_labels, heat_vals = [], []
    for lab, col in heat_map.items():
        heat_labels.append(lab)
        heat_vals.append(pct(safe_float(player_row.get(col, np.nan)), safe_float(league_means.get(col, np.nan))))

    # -------------------- Vizualizace: RADAR --------------------
    st.subheader("Radar – procenta vs. ligový průměr na pozici")
    vals = np.array([v if not pd.isna(v) else 0 for v in radar_vals])
    angles = np.linspace(0, 2*np.pi, len(vals), endpoint=False)
    vals_closed = np.concatenate([vals, vals[:1]])
    angles_closed = np.concatenate([angles, angles[:1]])

    fig_radar, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, vals_closed, linewidth=2, label=jmeno)
    ax.fill(angles_closed, vals_closed, alpha=0.2)
    ax.plot(angles_closed, np.ones_like(vals_closed)*100, linestyle="--", linewidth=1, label="Liga = 100%")
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels, fontsize=9)
    ax.set_yticks([50,100,150])
    ax.set_yticklabels(["50%","100%","150%"])
    ax.set_ylim(0,150)
    ax.set_title(f"{jmeno} – radar (% vs. liga)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    st.pyplot(fig_radar, use_container_width=True)

    # -------------------- Vizualizace: HEATMAP (0–150 %) --------------------
    st.subheader("Heatmapa – 0–150 % vůči ligovému průměru")
    cmap = LinearSegmentedColormap.from_list("r2g", ["#b30000", "#ff6b6b", "#ffd11a", "#b7e1a1", "#1e7a1e"])  # červená→žlutá→zelená

    hm = np.array([[v if not pd.isna(v) else np.nan] for v in heat_vals], dtype=float)
    hm_plot = np.nan_to_num(hm, nan=100.0)  # Na zobrazení NaN dáme 100, textem označíme "bez dat"

    fig_hm, ax2 = plt.subplots(figsize=(5.0, 0.45*len(heat_labels)+1))
    im = ax2.imshow(hm_plot, cmap=cmap, vmin=0, vmax=150)
    ax2.set_yticks(range(len(heat_labels)))
    ax2.set_yticklabels(heat_labels)
    ax2.set_xticks([])

    for i, v in enumerate(heat_vals):
        txt = "bez dat" if pd.isna(v) else f"{v:.1f}"
        ax2.text(0, i, txt, ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    ax2.set_title(f"{jmeno} – komplexní činnosti (% vs. liga, škála 0–150)")
    st.pyplot(fig_hm, use_container_width=True)

    # -------------------- Textový report (souvislý, přísný) --------------------
    st.subheader("Scouting report – souvislý text")

    slovnik = {lab: label_band(val) for lab, val in zip(heat_labels, heat_vals)}
    g90 = pct(safe_float(player_row.get("Goals per 90", np.nan)), safe_float(league_means.get("Goals per 90", np.nan)))
    a90 = pct(safe_float(player_row.get("Assists per 90", np.nan)), safe_float(league_means.get("Assists per 90", np.nan)))

    paragraphs = []
    paragraphs.append(
        f"{jmeno} ({int(vek) if not pd.isna(vek) else 'věk ?'}, {klub}) má v této sezoně odehráno {int(minuty) if not pd.isna(minuty) else '?'} minut, "
        f"vzorek je {'dostatečný' if (not pd.isna(minuty) and minuty>=900) else 'omezený'}. "
        f"Typově jde o hráče skupiny {zvolena_skup}."
    )
    paragraphs.append(
        "Soubojově: ofenzivní duely " + slovnik["Ofenzivní duely vyhrané %"] +
        ", defenzivní duely " + slovnik["Defenzivní duely vyhrané %"] +
        ", hlavičkové souboje " + slovnik["Hlavičkové souboje vyhrané %"] + ". "
        "Technicky: driblink " + slovnik["Úspěšnost driblinků %"] +
        ", centry " + slovnik["Úspěšnost centrů %"] +
        ", celková přesnost přihrávek " + slovnik["Úspěšnost přihrávek celkem %"] + "."
    )

    prod = "velmi slabá" if (not pd.isna(g90) and g90 < 70 and not pd.isna(a90) and a90 < 70) else ("slabší" if ((pd.isna(g90) or g90 < 100) or (pd.isna(a90) or a90 < 100)) else "dobrá")
    paragraphs.append("Produktivita je " + prod + ", rozhodující je výkon v poslední třetině (góly, poslední přihrávka).")

    styl = []
    # Nadprůměrné oblasti → doporučené herní prostředí
    if label_band(heat_vals[3]) == "nadprůměr":  # driblink
        styl.append("otevřený prostor a 1v1")
    if label_band(heat_vals[4]) == "nadprůměr":  # centry
        styl.append("centry ze strany")
    if label_band(heat_vals[0]) == "nadprůměr":
        styl.append("napadání a útočná agresivita")
    if label_band(heat_vals[1]) == "nadprůměr":
        styl.append("týmová defenziva/pressing")
    if styl:
        paragraphs.append("Herně sedí do prostředí: " + ", ".join(styl) + ".")

    if (not pd.isna(g90) and g90 < 70) and (not pd.isna(a90) and a90 < 70):
        doporuceni = "Nedoporučuji pro top kluby ligy; dává smysl v dolní/střední části tabulky s důrazem na přechodovou fázi a presink, bez tlaku na vysokou gólovou produkci."
    elif (not pd.isna(g90) and g90 >= 100) or (not pd.isna(a90) and a90 >= 100):
        doporuceni = "Vhodný pro ambiciózní kluby horní poloviny tabulky; přenos dovedností do dominantního prostředí je realistický."
    else:
        doporuceni = "Použitelný ve středním patře ligy; do top klubů pouze pod podmínkou jasného růstu ve finální fázi."

    paragraphs.append("Doporučení: " + doporuceni)

    report_text = "\n\n".join(paragraphs)
    st.write(report_text)

    # -------------------- Export do DOCX --------------------
    st.subheader("Export reportu (DOCX)")

    buf_radar = io.BytesIO()
    fig_radar.savefig(buf_radar, format='png', dpi=200, bbox_inches='tight')
    buf_radar.seek(0)

    buf_heat = io.BytesIO()
    fig_hm.savefig(buf_heat, format='png', dpi=200, bbox_inches='tight')
    buf_heat.seek(0)

    doc = Document()
    doc.add_heading(f"Typologie hráče – {jmeno}", level=0)
    meta = doc.add_paragraph()
    meta.add_run(f"Klub: {klub} | Pozice: {pozice_raw} | Skupina srovnání: {zvolena_skup}\n").bold = True
    meta.add_run(f"Minuty: {int(minuty) if not pd.isna(minuty) else '?'} | Věk: {int(vek) if not pd.isna(vek) else '?'}\n")
    doc.add_paragraph(datetime.now().strftime("Vygenerováno: %d.%m.%Y %H:%M"))

    doc.add_heading("Scouting report", level=1)
    for p in paragraphs:
        doc.add_paragraph(p)

    doc.add_heading("Radar (% vs. liga)", level=1)
    doc.add_picture(buf_radar, width=Inches(6.0))

    doc.add_heading("Heatmapa (0–150 %)", level=1)
    doc.add_picture(buf_heat, width=Inches(5.0))

    out = io.BytesIO()
    doc.save(out)
    out.seek(0)

    st.download_button(
        label="⬇️ Stáhnout scouting report (.docx)",
        data=out,
        file_name=f"Typologie_{jmeno.replace(' ','_')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

else:
    st.info("Nahraj ligový soubor a vyber/nahraj hráče – pak ti vygeneruju vizuály a stažitelný report.")
