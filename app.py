# -*- coding: utf-8 -*-
import io, zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from config import blocks, RUN, RUN_KEY, POS_REGEX, resolve_pos_group
from data_io import load_xlsx, auto_fix_run_df, get_pos_col, get_player_col, find_run_row_by_player
from models import compute_overall_for_row, avg_peer_index, run_index_for_row
from render_card import render_card_visual

# ============== UI meta ============
st.set_page_config(page_title="Karty – Slavia standard (váhy + běžecká data)", layout="wide")
st.title("⚽ Generátor datových karet (váhový model + vyhledávání hráčů + běžecká data)")

# ============== SIDEBAR ============
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

# ============== TABS ===============
tab_card, tab_search = st.tabs(["Karta hráče", "Vyhledávání hráčů"])

# -------- TAB 1: karta hráče --------
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

    run_scores = None; run_abs=None; run_index=np.nan
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
            miss_cz = [lab for eng,lab in RUN if models.series_for_alias_run(cz_agg_tmp, eng) is None]  # optional
            tmp_row = row_run if ('row_run' in locals() and row_run is not None) else pd.Series(dtype=object)
            from models import value_with_alias_run
            miss_pl = [lab for eng,lab in RUN if pd.isna(value_with_alias_run(tmp_row, eng))]
            st.write(f"Chybějící metriky v CZ benchmarku: {', '.join(miss_cz) if miss_cz else '—'}")
            st.write(f"Chybějící metriky u hráče: {', '.join(miss_pl) if miss_pl else '—'}")

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

# -------- TAB 2: vyhledávání --------
with tab_search:
    st.subheader("Vyhledávání kandidátů pro Slavii (benchmark = CZ liga)")

    colA, colB = st.columns(2)
    with colA:
        cz_file = st.file_uploader("CZ liga – herní (xlsx)", type=["xlsx"], key="cz_search")
        run_cz_file = st.file_uploader("CZ běžecká data (xlsx) [volitelné]", type=["xlsx"], key="cz_run_search")
    with colB:
        fr_file = st.file_uploader("Cizí liga – herní (xlsx)", type=["xlsx"], key="fr_search")
        run_fr_file = st.file_uploader("Cizí liga – běžecká (xlsx) [volitelné]", type=["xlsx"], key="fr_run_search")

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

                # herní index
                scores, sec_idx, overall = compute_overall_for_row(r, cz_agg, sec_weights, metric_weights, blocks)

                # běžecký index
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
                        "Role-index (vážený)": (overall if np.isnan(run_idx) or w_run<=0 else (1.0 - w_run)*overall + 0*w_run),  # jen pro kompatibilitu
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
                    import matplotlib.pyplot as plt
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

    # výstup
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
