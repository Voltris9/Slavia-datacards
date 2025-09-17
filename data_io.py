# -*- coding: utf-8 -*-
from io import BytesIO
import unicodedata, re
import numpy as np
import pandas as pd
import streamlit as st

from config import CUSTOM_RUN_RENAME

@st.cache_data
def load_xlsx(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))

def _best_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns: return n
    return None

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
    # long -> wide
    if "Metric" in df.columns and "Value" in df.columns:
        pcol = get_pos_col(df)
        plcol = get_player_col(df) or "Player"
        idx_cols = [c for c in [plcol, "Team", pcol, "Age"] if c and c in df.columns]
        wide = df.pivot_table(index=idx_cols, columns="Metric", values="Value", aggfunc="mean").reset_index()
        if plcol != "Player" and plcol in wide.columns: wide = wide.rename(columns={plcol:"Player"})
        if pcol and pcol != "Position" and pcol in wide.columns: wide = wide.rename(columns={pcol:"Position"})
        return wide
    # sjednocení názvů u wide
    pcol = get_pos_col(df)
    if pcol and pcol != "Position": df = df.rename(columns={pcol:"Position"})
    plcol = get_player_col(df)
    if plcol and plcol != "Player": df = df.rename(columns={plcol:"Player"})
    return df

def _strip_accents_lower(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join([c for c in s if not unicodedata.combining(c)])
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def std_player_series(sr: pd.Series) -> pd.Series:
    return sr.astype(str).map(_strip_accents_lower)

def apply_run_renames(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.rename(columns=CUSTOM_RUN_RENAME)
    id_map = {}
    if "Player" not in df.columns:
        c = _best_col(df, ["Name","player","name","Short Name"])
        if c: id_map[c] = "Player"
    if "Team" not in df.columns:
        c = _best_col(df, ["Club","team","Team"])
        if c: id_map[c] = "Team"
    if "Position" not in df.columns:
        c = _best_col(df, ["Pos","Role","Primary position","position"])
        if c: id_map[c] = "Position"
    if "Age" not in df.columns:
        c = _best_col(df, ["age","Age (years)"])
        if c: id_map[c] = "Age"
    return df.rename(columns=id_map)

def postcompute_run_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    # M/min P90 -> km/h
    if "Average speed (km/h)" not in df.columns and "M/min P90" in df.columns:
        df["Average speed (km/h)"] = pd.to_numeric(df["M/min P90"], errors="coerce") * 0.06
    # Accels = High + Medium
    if "Accelerations per 90" not in df.columns:
        acc = []
        for c in ["High Acceleration Count P90", "Medium Acceleration Count P90"]:
            if c in df.columns: acc.append(pd.to_numeric(df[c], errors="coerce"))
        if acc:
            s = acc[0]
            for a in acc[1:]: s = s.add(a, fill_value=0)
            df["Accelerations per 90"] = s
    # Decels = High + Medium
    if "Decelerations per 90" not in df.columns:
        dec = []
        for c in ["High Deceleration Count P90", "Medium Deceleration Count P90"]:
            if c in df.columns: dec.append(pd.to_numeric(df[c], errors="coerce"))
        if dec:
            s = dec[0]
            for a in dec[1:]: s = s.add(a, fill_value=0)
            df["Decelerations per 90"] = s
    return df

def auto_fix_run_df(run_df: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
    """aliasy + long->wide + dopočty + doplnění Position z herních dat"""
    if run_df is None or run_df.empty: return run_df
    run_df = apply_run_renames(run_df.copy())
    run_df = ensure_run_wide(run_df)
    run_df = postcompute_run_columns(run_df)

    if "Position" not in run_df.columns and game_df is not None and not game_df.empty:
        g = game_df.copy()
        if "Player" not in g.columns:
            pcol = _best_col(g, ["Name","player","name"])
            if pcol: g = g.rename(columns={pcol: "Player"})
        if "Position" in g.columns and "Player" in g.columns:
            g_small = g[["Player","Position"]].dropna().groupby("Player", as_index=False).agg({"Position":"first"})
            run_df["_key"] = std_player_series(run_df["Player"])
            g_small["_key"] = std_player_series(g_small["Player"])
            run_df = run_df.merge(g_small[["_key","Position"]], on="_key", how="left", suffixes=("","_from_game"))
            if "Position" not in run_df.columns:
                run_df = run_df.rename(columns={"Position_from_game": "Position"})
            run_df = run_df.drop(columns=["_key"], errors="ignore")

    for c in ["Player","Team","Position"]:
        if c in run_df.columns: run_df[c] = run_df[c].astype(str).str.strip()
    return run_df

def find_run_row_by_player(run_df: pd.DataFrame, player_name: str):
    """Robustní vyhledání řádku pro hráče (Player / Short Name)."""
    if run_df is None or run_df.empty or not player_name: return None
    key = _strip_accents_lower(str(player_name))
    pl = get_player_col(run_df) or "Player"

    if pl in run_df.columns:
        keys = run_df[pl].astype(str).map(_strip_accents_lower)
        idx = keys[keys==key].index
        if len(idx): return run_df.loc[idx[0]]

    if "Short Name" in run_df.columns:
        keys = run_df["Short Name"].astype(str).map(_strip_accents_lower)
        idx = keys[keys==key].index
        if len(idx): return run_df.loc[idx[0]]

    return None
