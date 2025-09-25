# Streamlit app: Typologie hráče – automatický scouting report (CZ)
# ---------------------------------------------------------------
# Požadavky: pip install -r requirements.txt
# Spuštění:  streamlit run app_typologie_hrace_streamlit.py
# Vstupy:    1) Liga (CZE1.xlsx)  2) Profil hráče (xlsx) NEBO výběr hráče z ligového souboru
# Výstup:    Vizuály + stažitelný .docx scouting report

import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

# DOCX – bezpečný import (app poběží i bez knihovny)
try:
    from docx import Document
    from docx.shared import Inches
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

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
def band(v, low=70, high=130):
    if pd.isna(v):
        return "bez dat"
    if v < low:
        return "podprůměr"
    if v > high:
        return "nadprůměr"
    return "ligový standard"

def band_adj(word):
    return {
        "podprůměr": "podprůměrný",
        "nadprůměr": "nadprůměrný",
        "ligový standard": "ligový standard",
        "bez dat": "bez dat"
    }.get(word, word)

# Primární pozice – vytáhnu první dvou/třípísmenný token (LW, RW, CF, AMF, CMF, DMF, CB, RB, LB, RWB, LWB, GK…)
def extract_primary_position(pos_text: str) -> str:
    s = str(pos_text or "").upper()
    # vezmi první „písmenkový“ token (ignoruj čísla a oddělovače)
    m = re.findall(r"[A-Z]{2,3}", s)
    return m[0] if m else s.strip()[:3]

# Filtrovat ligu jen na stejnou pozici (porovnávací vzorek)
def filter_same_position(df: pd.DataFrame, primary_pos: str) -> pd.DataFrame:
    if "Position" not in df.columns:
        return df.iloc[0:0].copy()
    pat = rf"\b{re.escape(primary_pos)}\b"
    mask = df["Position"].astype(str).str.upper().str.contains(pat, regex=True, na=False)
    out = df.loc[mask].copy()
    # Kdyby náhodou nic – nech aspoň všechny s obsahem stejného prefixu (záchrana, ale stále stejná pozice-rodina)
    if out.empty:
        mask2 = df["Position"].astype(str).str.upper().str.contains(primary_pos[:2], na=False)
        out = df.loc[mask2].copy()
    return out

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

    # Stejná pozice – referenční vzorek
    primary_pos = extract_primary_position(pozice_raw)
    league_pos = filter_same_position(league_df, primary_pos)
    n_ref = int(len(league_pos))
    league_means = league_pos.mean(numeric_only=True)

    # Metriky pro RADAR a HEATMAPU (v % vůči průměru na stejné pozici)
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

    # Další metriky pro text (aktivita/kreativita)
    shots90   = safe_float(player_row.get("Shots per 90", np.nan))
    shots90_L = safe_float(league_means.get("Shots per 90", np.nan))
    touch90   = safe_float(player_row.get("Touches in box per 90", np.nan))
    touch90_L = safe_float(league_means.get("Touches in box per 90", np.nan))
    keyp90    = safe_float(player_row.get("Key passes per 90", np.nan))
    keyp90_L  = safe_float(league_means.get("Key passes per 90", np.nan))

    # Výpočty % vs liga (stejná pozice)
    radar_labels, radar_vals = [], []
    for lab, col in radar_map.items():
        radar_labels.append(lab)
        radar_vals.append(pct(safe_float(player_row.get(col, np.nan)), safe_float(league_means.get(col, np.nan))))

    heat_labels, heat_vals = [], []
    for lab, col in heat_map.items():
        heat_labels.append(lab)
        heat_vals.append(pct(safe_float(player_row.get(col, np.nan)), safe_float(league_means.get(col, np.nan))))

    # -------------------- Vizualizace: RADAR (malý, nepřetéká) --------------------
    st.subheader("Radar – procenta vs. ligový průměr (stejná pozice)")
    vals = np.array([v if not pd.isna(v) else 0 for v in radar_vals])
    angles = np.linspace(0, 2*np.pi, len(vals), endpoint=False)
    vals_closed = np.concatenate([vals, vals[:1]])
    angles_closed = np.concatenate([angles, angles[:1]])

    fig_radar, ax = plt.subplots(figsize=(3.6, 3.6), subplot_kw=dict(polar=True))  # kompaktní
    ax.plot(angles_closed, vals_closed, linewidth=1.5, label=jmeno)
    ax.fill(angles_closed, vals_closed, alpha=0.18)
    ax.plot(angles_closed, np.ones_like(vals_closed)*100, linestyle="--", linewidth=1, label=f"Liga = 100% ({primary_pos}, N={n_ref})")
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels, fontsize=7)
    ax.set_yticks([50,100,150])
    ax.set_yticklabels(["50%","100%","150%"], fontsize=7)
    ax.set_ylim(0,150)
    ax.grid(alpha=0.35, linewidth=0.6)
    ax.set_title(f"{jmeno} – radar (% vs. liga, stejná pozice: {primary_pos}, N={n_ref})", fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22,1.05), prop={'size':7})
    st.pyplot(fig_radar, use_container_width=False)

    # -------------------- Vizualizace: HEATMAP (kompakt, 0–150 %) --------------------
    st.subheader("Heatmapa – 0–150 % vůči ligovému průměru (stejná pozice)")
    cmap = LinearSegmentedColormap.from_list("r2g",
        ["#b30000", "#ff6b6b", "#ffd11a", "#b7e1a1", "#1e7a1e"])  # červená → žlutá → zelená

    hm = np.array([[v if not pd.isna(v) else np.nan] for v in heat_vals], dtype=float)
    hm_plot = np.nan_to_num(hm, nan=100.0)

    fig_hm, ax2 = plt.subplots(figsize=(3.2, 0.26*len(heat_labels)+0.6))
    ax2.imshow(hm_plot, cmap=cmap, vmin=0, vmax=150)
    ax2.set_yticks(range(len(heat_labels)))
    ax2.set_yticklabels(heat_labels, fontsize=8)
    ax2.set_xticks([])
    for i, v in enumerate(heat_vals):
        txt = "bez dat" if pd.isna(v) else f"{v:.1f}"
        ax2.text(0, i, txt, ha='center', va='center', fontsize=8, fontweight='bold', color='black')
    ax2.set_title(f"{jmeno} – komplexní činnosti (0–150 %, stejná pozice: {primary_pos}, N={n_ref})", fontsize=9)
    st.pyplot(fig_hm, use_container_width=False)

    # -------------------- Textový report (dlouhý, hovorový) --------------------
    st.subheader("Scouting report – souvislý text")

    # slovní pásma pro hlavní oblasti
    slovnik = {lab: band(val) for lab, val in zip(heat_labels, heat_vals)}
    g90 = pct(safe_float(player_row.get("Goals per 90", np.nan)), safe_float(league_means.get("Goals per 90", np.nan)))
    a90 = pct(safe_float(player_row.get("Assists per 90", np.nan)), safe_float(league_means.get("Assists per 90", np.nan)))
    shots_band  = band(pct(shots90, shots90_L)) if not pd.isna(shots90) and not pd.isna(shots90_L) else "bez dat"
    touch_band  = band(pct(touch90, touch90_L)) if not pd.isna(touch90) and not pd.isna(touch90_L) else "bez dat"
    keyp_band   = band(pct(keyp90,  keyp90_L))  if not pd.isna(keyp90) and not pd.isna(keyp90_L)  else "bez dat"

    paragraphs = []

    # 0) Informace o referenčním vzorku
    ref_info = f"Porovnáváno s ligovým průměrem na stejné pozici ({primary_pos}, N={n_ref})."

    # 1) Executive summary (hovorově)
    summary_bits = []
    # pokus o typologii podle pár pásem
    if slovnik.get("Úspěšnost driblinků %") == "nadprůměr":
        summary_bits.append("rychlé křídlo do přechodu" if primary_pos in ["LW","RW","LWF","RWF","AMF"] else "hráč s dobrou prací v 1v1")
    if slovnik.get("Úspěšnost přihrávek celkem %") == "nadprůměr" and primary_pos in ["CB","RB","LB","RWB","LWB"]:
        summary_bits.append("klid na míči v první rozehrávce")
    if slovnik.get("Hlavičkové souboje vyhrané %") == "nadprůměr" and primary_pos in ["CF","CB"]:
        summary_bits.append("silný ve vzduchu")
    if not summary_bits:
        summary_bits.append("ligový standard s pár silnými stránkami")

    summary = (
        f"{jmeno} ({int(vek) if not pd.isna(vek) else 'věk ?'}, {klub}) je {', '.join(summary_bits)}. "
        f"Z dat: silný v {('driblinku' if slovnik.get('Úspěšnost driblinků %')=='nadprůměr' else 'základních dovednostech')}, "
        f"defenzivně {slovnik.get('Defenzivní duely vyhrané %','?')}, ve vzduchu {slovnik.get('Hlavičkové souboje vyhrané %','?')}. "
        + ref_info
    )
    paragraphs.append(summary)

    # 2) Produkce & vliv
    if (not pd.isna(g90) and g90 < 70) and (not pd.isna(a90) and a90 < 70):
        prod_phrase = "finální výstup je teď slabý – chybí góly i poslední přihrávka."
    elif (not pd.isna(g90) and g90 >= 100) or (not pd.isna(a90) and a90 >= 100):
        prod_phrase = "finální výstup je přenositelný – drží ligový standard nebo nad ním."
    else:
        prod_phrase = "produktivitou kolísá kolem průměru a výstup je nepravidelný."

    act_bits = []
    if shots_band != "bez dat":
        act_bits.append(f"do střel se dostává {shots_band}")
    if touch_band != "bez dat":
        act_bits.append(f"v boxu je vidět {touch_band}")
    if keyp_band != "bez dat":
        act_bits.append(f"tvorba šancí (key passes) je {keyp_band}")
    if act_bits:
        paragraphs.append((", ".join(act_bits) + ". " + prod_phrase).capitalize())
    else:
        paragraphs.append(prod_phrase.capitalize())

    # 3) Souboje + technika (narrativně)
    paragraphs.append(
        "V soubojích působí "
        f"{band_adj(slovnik['Ofenzivní duely vyhrané %'])} směrem dopředu, "
        f"{band_adj(slovnik['Defenzivní duely vyhrané %'])} v defenzivě "
        f"a {band_adj(slovnik['Hlavičkové souboje vyhrané %'])} ve vzduchu. "
        "Technicky je "
        f"{band_adj(slovnik['Úspěšnost driblinků %'])} v driblinku a "
        f"{band_adj(slovnik['Úspěšnost centrů %'])} v centrech; "
        f"celková přesnost přihrávek je {slovnik['Úspěšnost přihrávek celkem %']}."
    )

    # 4) Herní styl & fit
    fit_parts = []
    if slovnik.get("Úspěšnost driblinků %") == "nadprůměr":
        fit_parts.append("otevřený prostor a 1v1")
    if slovnik.get("Úspěšnost centrů %") == "nadprůměr":
        fit_parts.append("koncovka po centrech ze strany")
    if slovnik.get("Ofenzivní duely vyhrané %") == "nadprůměr":
        fit_parts.append("aktivní presink a útočná agresivita")
    if slovnik.get("Defenzivní duely vyhrané %") == "nadprůměr":
        fit_parts.append("týmová defenziva/pressing")
    if fit_parts:
        paragraphs.append("Herně z něj dostaneme nejvíc v: " + ", ".join(fit_parts) + ".")
    else:
        paragraphs.append("Herně mu víc sedí přechodová fáze než dlouhé kombinace do bloku.")

    # 5) Rizika
    risks = []
    if slovnik.get("Hlavičkové souboje vyhrané %") == "podprůměr":
        risks.append("slabší vzduch (zadní tyč/standardky)")
    if (not pd.isna(g90) and g90 < 70):
        risks.append("podvážená gólová produkce k objemu akcí")
    if keyp_band == "podprůměr":
        risks.append("málo kvalitní poslední přihrávky")
    if risks:
        paragraphs.append("Rizika: " + ", ".join(risks) + ".")

    # 6) Přísné doporučení + KPI
    if (not pd.isna(g90) and g90 < 70) and (not pd.isna(a90) and a90 < 70):
        recomend = "Do top klubů teď NE; smysl dává dolní/střed ligy s důrazem na přechod a presink, bez tlaku na dvouciferná čísla."
    elif (not pd.isna(g90) and g90 >= 100) or (not pd.isna(a90) and a90 >= 100):
        recomend = "OK pro ambiciózní horní polovinu tabulky; přenos do dominantnějšího prostředí je realistický."
    else:
        recomend = "Použitelný střed ligy; do top pouze při jasném růstu ve finále."
    paragraphs.append("Doporučení: " + recomend)

    kpi = [
        "Finále v boxu – opakovatelné vzorce (pullback / 1–2 doteky do střely).",
        "Rychlejší volba v poslední třetině – méně „přání“ centrů, víc přihrávek do nohy pod sebou.",
        "Zadní tyč – načasování a zakončení hlavou (pokud je podprůměrný vzduch)."
    ]
    paragraphs.append("Krátký plán na 8–12 týdnů: " + " ".join([f"{i+1}) {t}" for i, t in enumerate(kpi)]) )

    report_text = "\n\n".join(paragraphs)
    st.write(report_text)

    # -------------------- Export reportu (DOCX) --------------------
    st.subheader("Export reportu (DOCX)")
    if HAVE_DOCX:
        buf_radar = io.BytesIO()
        fig_radar.savefig(buf_radar, format='png', dpi=200, bbox_inches='tight')
        buf_radar.seek(0)

        buf_heat = io.BytesIO()
        fig_hm.savefig(buf_heat, format='png', dpi=200, bbox_inches='tight')
        buf_heat.seek(0)

        doc = Document()
        doc.add_heading(f"Typologie hráče – {jmeno}", level=0)
        meta = doc.add_paragraph()
        meta.add_run(f"Klub: {klub} | Pozice: {pozice_raw}\n").bold = True
        meta.add_run(f"Referenční vzorek: stejná pozice {primary_pos}, N={n_ref}\n")
        meta.add_run(f"Minuty: {int(minuty) if not pd.isna(minuty) else '?'} | Věk: {int(vek) if not pd.isna(vek) else '?'}\n")
        doc.add_paragraph(datetime.now().strftime("Vygenerováno: %d.%m.%Y %H:%M"))

        doc.add_heading("Scouting report (souvislý)", level=1)
        for p in paragraphs:
            doc.add_paragraph(p)

        doc.add_heading("Radar (% vs. liga – stejná pozice)", level=1)
        doc.add_picture(buf_radar, width=Inches(3.2))  # malé do DOCX

        doc.add_heading("Heatmapa (0–150 % – stejná pozice)", level=1)
        doc.add_picture(buf_heat, width=Inches(3.0))  # malé do DOCX

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
        st.warning("DOCX export není dostupný – chybí python-docx. Přidej ho do requirements.txt a redeployni appku.")
else:
    st.info("Nahraj ligový soubor a vyber/nahraj hráče – pak ti vygeneruju vizuály a stažitelný report.")



