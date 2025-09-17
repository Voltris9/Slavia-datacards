# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from config import ALIASES, ALIASES_RUN, RUN_KEY, RUN, peers_for_pos_group

# ---------- Herní výpočty ----------
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

def get_value_with_alias(row, key):
    if key in row.index: return row[key]
    for cand in ALIASES.get(key, []):
        if cand in row.index: return row[cand]
    if key == "Cross accuracy, %" and "Accurate crosses, %" in row.index:
        return row["Accurate crosses, %"]
    return np.nan

def compute_section_scores(player_row: pd.Series, agg: pd.DataFrame, blocks, metric_weights=None):
    sec_scores, sec_index = {}, {}
    for _, lst, key in blocks:
        part = {}
        for eng, label in lst:
            part[label] = normalize_metric(agg, eng, get_value_with_alias(player_row, eng))
        sec_scores[key] = part
        if metric_weights and metric_weights.get(key):
            acc = 0.0; wsum = 0.0
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

def compute_overall_for_row(row, cz_agg, sec_weights, metric_weights, blocks):
    scores, sec_idx = compute_section_scores(row, cz_agg, blocks, metric_weights)
    overall = weighted_role_index(sec_idx, sec_weights)
    return scores, sec_idx, overall

def avg_peer_index(cz_agg, pos_group, sec_weights, metric_weights, blocks):
    peers = peers_for_pos_group(pos_group)
    vals=[]
    for nm in peers:
        if nm not in cz_agg.index: continue
        r = cz_agg.loc[nm].copy()
        r["Player"]=nm; r["Position"]=pos_group
        _, _, overall = compute_overall_for_row(r, cz_agg, sec_weights, metric_weights, blocks)
        if not np.isnan(overall): vals.append(overall)
    return float(np.mean(vals)) if vals else np.nan

# ---------- Běh ----------
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
        if pd.isna(val_abs) and eng == "Average speed (km/h)" and "M/min P90" in player_row.index:
            val_abs = pd.to_numeric(player_row["M/min P90"], errors="coerce") * 0.06
        run_abs[label] = val_abs if not pd.isna(val_abs) else np.nan
        run_scores[label] = normalize_run_metric(cz_run_agg, eng, val_abs)
    vals = [v for v in run_scores.values() if not pd.isna(v)]
    run_index = float(np.mean(vals)) if vals else np.nan
    return {RUN_KEY: run_scores}, run_abs, run_index

def run_index_for_row(row, cz_run_df_pos):
    """Vrací (run_index, run_scores, run_abs)."""
    if cz_run_df_pos is None or cz_run_df_pos.empty:
        return np.nan, {}, {}
    plc = cz_run_df_pos.columns[0]  # dummy, přejmenujeme níže bezpečně
    if "Player" in cz_run_df_pos.columns:
        cz_tmp = cz_run_df_pos
    else:
        # pokus o sjednocení na Player
        any_pl = [c for c in ["Name","player","name","Short Name"] if c in cz_run_df_pos.columns]
        cz_tmp = cz_run_df_pos.rename(columns={any_pl[0]:"Player"}) if any_pl else cz_run_df_pos
    cz_run_agg = cz_tmp.groupby("Player").mean(numeric_only=True)
    run_scores, run_abs, run_idx = compute_run_scores(row, cz_run_agg)
    return run_idx, run_scores, run_abs
