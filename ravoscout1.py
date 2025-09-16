import io
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

st.set_page_config(page_title="Karty – Slavia standard (váhy podle pozic)", layout="wide")

st.title("⚽ Generátor datových karet (váhový model + vyhledávání hráčů)")

# =============================================================================
# CACHE – načítání Excelu (zrychlí a stabilizuje reruny)
# =============================================================================
@st.cache_data
def load_xlsx(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))

# =============================================================================
# BARVY
# =============================================================================
def color_for(val):
    if pd.isna(val): return "lightgrey"
    if val <= 25: return "#FF4C4C"   # červená
    if val <= 50: return "#FF8C00"   # oranžová
    if val <= 75: return "#FFD700"   # žlutá
    return "#228B22"                 # zelená

# =============================================================================
# BLOKY A METRIKY (STANDARD)
# =============================================================================
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

# =============================================================================
# ALIASY
# =============================================================================
ALIASES = {
    "Cross accuracy, %": ["Accurate crosses, %","Cross accuracy, %"],
    "Progressive passes per 90": ["Progressive passes per 90","Progressive passes/90"],
    "Passes to final third per 90": ["Passes to final third per 90","Passes to final third/90"],
    "Dribbles per 90": ["Dribbles per 90","Dribbles/90"],
    "Progressive runs per 90": ["Progressive runs per 90","Progressive runs/90"],
    "Second assists per 90": ["Second assists per 90","Second assists/90"],
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

# =============================================================================
# POZICE + SLAVIA PEERS
# =============================================================================
POS_REGEX = {
    "CB/DF": r"(CB|DF)",
    "RB": r"(RB)",
    "LB": r"(LB)",
    "WB/RWB/LWB": r"(WB|RWB|LWB)",
    "DM": r"(DM)",
    "CM": r"(CM)",
    "AM": r"(AM)",
    "RW": r"(RW)",
    "LW": r"(LW)",
    "CF/ST": r"(CF|ST|FW)",
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
    return "CM"  # bezpečný default

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

# =============================================================================
# POMOCNÉ FUNKCE
# =============================================================================
def get_age(row):
    age = row.get("Age", None)
    if pd.isna(age): return "n/a"
    try:
        return int(age)
    except:
        return str(age)

def series_for_alias(agg: pd.DataFrame, eng_key: str):
    if eng_key in agg.columns:
        return agg[eng_key]
    for cand in ALIASES.get(eng_key, []):
        if cand in agg.columns:
            return agg[cand]
    if eng_key == "Cross accuracy, %" and "Accurate crosses, %" in agg.columns:
        return agg["Accurate crosses, %"]
    return None

def normalize_metric(agg: pd.DataFrame, eng_key: str, value):
    s = series_for_alias(agg, eng_key)
    if s is None or pd.isna(value):
        return np.nan
    s = pd.to_numeric(s, errors="coerce").dropna()
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if s.empty or pd.isna(v):
        return np.nan
    mn, mx = s.min(), s.max()
    if mx == mn:
        return 50.0
    score = (v - mn) / (mx - mn) * 100.0
    return float(np.clip(score, 0, 100))

def compute_section_scores(player_row: pd.Series, agg: pd.DataFrame, blocks, metric_weights=None):
    """Vrátí: dict sekce -> {label: score 0-100}, + dict sekce -> sekční index."""
    sec_scores = {}
    sec_index = {}
    for title, lst, key in blocks:
        part = {}
        for eng, label in lst:
            val = get_value_with_alias(player_row, eng)
            part[label] = normalize_metric(agg, eng, val)
        sec_scores[key] = part
        # sekční index
        if metric_weights and metric_weights.get(key):
            wsum = 0.0; acc = 0.0
            for label, w in metric_weights[key].items():
                v = part.get(label, np.nan)
                if not pd.isna(v):
                    acc += v * w
                    wsum += w
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
            acc += v * w
            totw += w
    return float(acc/totw) if totw>0 else np.nan

# =============================================================================
# VYKRESLENÍ KARTY (s věkem)
# =============================================================================
def render_card_visual(player, team, pos, age, scores, sec_index, overall, verdict):
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("off")
    # Hlava
    ax.text(0.02,0.96, f"{player} (věk {age})", fontsize=20, fontweight="bold", va="top", color="black")
    ax.text(0.02,0.93, f"Klub: {team}   Pozice: {pos}", fontsize=13, va="top", color="black")

    # Levá část – 2 sloupce/sekce
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

    # Pravá část – souhrny + verdikt
    ax.text(0.55,0.9,"Souhrnné indexy (0–100 %) – vážené",fontsize=16,fontweight="bold",va="top",color="black")
    y=0.85
    for key_disp in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        val = sec_index.get(key_disp, np.nan)
        c = color_for(val)
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c,alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"{key_disp}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",
                fontsize=13,va="center",ha="left",color="black")
        y -= 0.075
    c_over=color_for(overall)
    ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=c_over,alpha=0.7,lw=0))
    ax.text(0.56,y-0.005,f"Celkový role-index (vážený): {'n/a' if pd.isna(overall) else str(int(round(overall)))+'%'}",
            fontsize=14,fontweight="bold",va="center",ha="left",color="black")

    ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color='lightgrey',alpha=0.35,lw=0))
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center",color="black")
    return fig

# =============================================================================
# PEERS INDEX + HLEDÁNÍ KANDIDÁTŮ
# =============================================================================
def compute_overall_for_row(row, cz_agg, sec_weights, metric_weights, blocks=blocks):
    scores, sec_idx = compute_section_scores(row, cz_agg, blocks, metric_weights)
    overall = weighted_role_index(sec_idx, sec_weights)
    return scores, sec_idx, overall

def avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights):
    peers = peers_for_pos_group(pos_group)
    vals=[]
    for nm in peers:
        if nm not in cz_agg.index:
            continue
        r = cz_agg.loc[nm]
        row_like = r.copy()
        row_like["Player"]=nm
        row_like["Team"]=row_like.get("Team","Slavia Praha")
        row_like["Position"]=pos_group
        row_like["Age"]=row_like.get("Age", np.nan)
        _, sec_idx, overall = compute_overall_for_row(row_like, cz_agg, sec_weights, metric_weights)
        if not np.isnan(overall): vals.append(overall)
    return float(np.mean(vals)) if vals else np.nan

def search_candidates(cz_df, foreign_df, positions_selected, sec_weights, metric_weights,
                      min_minutes=None, min_games=None, league_name=""):
    # filtrovat cizí ligu dle pozic
    mask_pos = pd.Series([False]*len(foreign_df))
    for p in positions_selected:
        rgx = POS_REGEX[p]
        mask_pos |= foreign_df["Position"].astype(str).str.contains(rgx, na=False, regex=True)
    base = foreign_df.loc[mask_pos].copy()

    # volitelné filtry
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
        pos_group = resolve_pos_group(str(r.get("Position","")))
        rgx = POS_REGEX[pos_group]
        cz_pos = cz_df[cz_df["Position"].astype(str).str.contains(rgx, na=False, regex=True)]
        if cz_pos.empty:
            continue
        cz_agg = cz_pos.groupby("Player").mean(numeric_only=True)

        scores, sec_idx, overall = compute_overall_for_row(r, cz_agg, sec_weights, metric_weights)
        peer_avg = avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights)
        verdict = ("ANO – potenciální posila do Slavie"
                   if (not np.isnan(peer_avg) and not np.isnan(overall) and overall >= peer_avg)
                   else "NE – nedosahuje úrovně slávistických konkurentů")
        if verdict.startswith("ANO"):
            player = r.get("Player",""); team = r.get("Team","")
            age = get_age(r); pos = r.get("Position","")
            rows.append({
                "Hráč": player, "Věk": age, "Klub": team, "Pozice": pos, "Liga": league_name,
                "Index Def": sec_idx.get("Defenziva", np.nan),
                "Index Off": sec_idx.get("Ofenziva", np.nan),
                "Index Pass": sec_idx.get("Přihrávky", np.nan),
                "Index 1v1": sec_idx.get("1v1", np.nan),
                "Role-index (vážený)": overall,
                "Verdikt": verdict
            })
            # karta do ZIPu
            fig = render_card_visual(player, team, pos, age, scores, sec_idx, overall, verdict)
            bio = BytesIO()
            fig.savefig(bio, format="png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            cards.append((str(player), bio.getvalue()))
    res = pd.DataFrame(rows)
    return res, cards

# =============================================================================
# SIDEBAR – VÁHY SEKcí + (volitelné) váhy metrik
# =============================================================================
st.sidebar.header("⚙️ Váhy sekcí")
default_weights = {"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
sec_weights = {}
for sec in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
    sec_weights[sec] = st.sidebar.slider(f"{sec}", 0, 100, default_weights[sec], 1)
# normování na 100 %
tot = sum(sec_weights.values()) or 1
for k in sec_weights:
    sec_weights[k] = 100.0 * sec_weights[k] / tot

with st.sidebar.expander("Váhy metrik v sekcích (volitelné)", expanded=False):
    metric_weights = {}
    for title, lst, key in blocks:
        st.markdown(f"**{title}**")
        tmp = {}
        for _, label in lst:
            tmp[label] = st.slider(f"– {label}", 0, 100, 10, 1, key=f"{key}_{label}")
        ssum = sum(tmp.values())
        metric_weights[key] = None if ssum==0 else {lab: w/ssum for lab, w in tmp.items()}

# =============================================================================
# TABS
# =============================================================================
tab_card, tab_search = st.tabs(["Karta hráče", "Vyhledávání hráčů"])

# ---------------------------------------------------------------------
# TAB 1: KARTA HRÁČE (jako dřív, ale s věkem)
# ---------------------------------------------------------------------
with tab_card:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        league_file = st.file_uploader("Nahraj ligovou databázi (xlsx) – CZ liga", type=["xlsx"], key="league_card")
    with col_up2:
        players_file = st.file_uploader("Nahraj soubor s hráčem/hráči (xlsx) – export Wyscout", type=["xlsx"], key="players_card")

    if not league_file or not players_file:
        st.info("➡️ Nahraj oba soubory (ligový dataset + hráčský export).")
        st.stop()

    try:
        league = pd.read_excel(league_file)
        players = pd.read_excel(players_file)
    except Exception as e:
        st.error(f"Chyba při načítání souborů: {e}")
        st.stop()

    player_names = players["Player"].dropna().unique().tolist()
    sel_player = st.selectbox("Vyber hráče ze souboru hráčů", player_names)

    row = players.loc[players["Player"] == sel_player].iloc[0]
    player = row.get("Player","")
    team = row.get("Team","")
    pos = row.get("Position","")

    # srovnávací skupina v CZ lize
    pos_group = resolve_pos_group(str(pos))
    rgx = POS_REGEX[pos_group]
    group = league[league["Position"].astype(str).str.contains(rgx, na=False, regex=True)].copy()
    agg = group.groupby("Player").mean(numeric_only=True)

    # výpočet metrik -> sekce -> indexy -> overall
    scores, block_idx = compute_section_scores(row, agg, blocks, metric_weights)
    overall = weighted_role_index(block_idx, sec_weights)

    # Verdikt vs Slavia peers
    peer_avg = avg_peer_index(agg, pos_group, sec_weights, metric_weights)
    verdict = ("ANO – potenciální posila do Slavie"
               if (not np.isnan(peer_avg) and not np.isnan(overall) and overall >= peer_avg)
               else "NE – nedosahuje úrovně slávistických konkurentů")

    # karta
    fig = render_card_visual(player, team, pos, get_age(row), scores, block_idx, overall, verdict)
    st.pyplot(fig)

    # download PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    st.download_button("📥 Stáhnout kartu jako PNG", data=buf.getvalue(), file_name=f"{player}.png", mime="image/png")

# ---------------------------------------------------------------------
# TAB 2: VYHLEDÁVÁNÍ HRÁČŮ (STABILNÍ – session_state)
# ---------------------------------------------------------------------
with tab_search:
    st.subheader("Vyhledávání kandidátů pro Slavii (benchmark = CZ liga)")

    colA, colB = st.columns(2)
    with colA:
        cz_file = st.file_uploader("CZ liga (xlsx)", type=["xlsx"], key="cz_search")
    with colB:
        fr_file = st.file_uploader("Cizí liga (xlsx)", type=["xlsx"], key="fr_search")

    # Ulož uploady do session_state, aby nezmizely po rerunu
    if cz_file:
        st.session_state["cz_bytes"] = cz_file.getvalue()
    if fr_file:
        st.session_state["fr_bytes"] = fr_file.getvalue()

    all_pos_opts = list(POS_REGEX.keys())
    positions_selected = st.multiselect("Pozice", all_pos_opts, default=all_pos_opts, key="search_positions")

    c1,c2,c3 = st.columns(3)
    with c1:
        league_name = st.text_input("Název ligy (zobrazí se ve výstupu)", value="Cizí liga", key="search_league")
    with c2:
        min_minutes = st.number_input("Min. minut (pokud ve zdroji)", min_value=0, value=0, step=100, key="search_min_minutes")
    with c3:
        min_games = st.number_input("Min. zápasů (pokud ve zdroji)", min_value=0, value=0, step=1, key="search_min_games")

    run = st.button("Spustit vyhledávání", key="search_run")

    # Výsledky v session_state (zůstanou i po kliknutí „Zobraz kartu“)
    res_df = st.session_state.get("search_results")
    cards = st.session_state.get("search_cards")
    fr_df_cached = st.session_state.get("fr_df")
    cz_df_cached = st.session_state.get("cz_df")

    # Nový běh po kliknutí
    if run:
        if "cz_bytes" not in st.session_state or "fr_bytes" not in st.session_state:
            st.error("Nahraj oba soubory (CZ + cizí liga).")
            st.stop()

        cz_df = load_xlsx(st.session_state["cz_bytes"])
        fr_df = load_xlsx(st.session_state["fr_bytes"])

        res_df, cards = search_candidates(
            cz_df, fr_df, st.session_state["search_positions"],
            sec_weights=sec_weights, metric_weights=metric_weights,
            min_minutes=st.session_state["search_min_minutes"] or None,
            min_games=st.session_state["search_min_games"] or None,
            league_name=st.session_state["search_league"]
        )

        # ulož do session_state
        st.session_state["search_results"] = res_df
        st.session_state["search_cards"] = cards
        st.session_state["fr_df"] = fr_df
        st.session_state["cz_df"] = cz_df

        # a pro zbytek kódu použij právě spočítané
        fr_df_cached, cz_df_cached = fr_df, cz_df

    # Zobrazení (z session_state)
    res_df = st.session_state.get("search_results")
    cards = st.session_state.get("search_cards")
    fr_df_cached = st.session_state.get("fr_df")
    cz_df_cached = st.session_state.get("cz_df")

    if res_df is None or res_df.empty:
        st.info("Zatím žádné výsledky – nahraj soubory a klikni na *Spustit vyhledávání*.")
    else:
        st.success(f"Nalezeno kandidátů: {len(res_df)}")
        st.dataframe(res_df, use_container_width=True)

        # Export CSV
        st.download_button(
            "📥 Stáhnout CSV s kandidáty",
            data=res_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"kandidati_{st.session_state.get('search_league','liga')}.csv",
            mime="text/csv",
            key="dl_csv"
        )

        # Export ZIP karet
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

        # Náhled karty vybraného hráče (bez pádu vyhledávání)
        sel = st.selectbox("Zobraz kartu hráče", res_df["Hráč"].tolist(), key="preview_player")

        if sel and fr_df_cached is not None and cz_df_cached is not None:
            r = fr_df_cached.loc[fr_df_cached["Player"]==sel].iloc[0]
            pos_group = resolve_pos_group(str(r.get("Position","")))
            rgx = POS_REGEX[pos_group]
            cz_pos = cz_df_cached[cz_df_cached["Position"].astype(str).str.contains(rgx, na=False, regex=True)]
            if not cz_pos.empty:
                cz_agg = cz_pos.groupby("Player").mean(numeric_only=True)
                scores, sec_idx, overall = compute_overall_for_row(r, cz_agg, sec_weights, metric_weights)
                peer_avg = avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights)
                verdict = ("ANO – potenciální posila do Slavie"
                           if (not np.isnan(peer_avg) and not np.isnan(overall) and overall >= peer_avg)
                           else "NE – nedosahuje úrovně slávistických konkurentů")
                fig = render_card_visual(
                    r.get("Player",""), r.get("Team",""), r.get("Position",""), get_age(r),
                    scores, sec_idx, overall, verdict
                )
                st.pyplot(fig)
