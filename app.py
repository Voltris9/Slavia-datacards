# app.py — Slavia datacards (herní + běžecká v jedné kartě, smart matching jmen)
import zipfile, unicodedata, re
from io import BytesIO
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

st.set_page_config(page_title="Karty – Slavia standard", layout="wide")
st.title("⚽ Generátor datových karet (herní + běžecká v jedné kartě)")

# ---------- Utils ----------
@st.cache_data
def load_xlsx(b: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(b))

def color_for(v):
    if pd.isna(v): return "lightgrey"
    if v <= 25: return "#FF4C4C"
    if v <= 50: return "#FF8C00"
    if v <= 75: return "#FFD700"
    return "#228B22"

def _best_col(df, names):
    for n in names:
        if n in df.columns: return n
    return None

def get_pos_col(df):
    cols = df.columns if df is not None else []
    return next((c for c in ["Position","Pos","position","Role","Primary position"] if c in cols), None)

def get_player_col(df):
    cols = df.columns if df is not None else []
    return next((c for c in ["Player","Name","player","name","Short Name"] if c in cols), None)

def ensure_run_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    if "Metric" in df.columns and "Value" in df.columns:
        idx=[c for c in [get_player_col(df) or "Player","Team",get_pos_col(df),"Age"] if c and c in df.columns]
        wide=df.pivot_table(index=idx, columns="Metric", values="Value", aggfunc="mean").reset_index()
        pc=get_player_col(df)
        if pc and pc!="Player" and pc in wide.columns: wide=wide.rename(columns={pc:"Player"})
        sc=get_pos_col(df)
        if sc and sc!="Position" and sc in wide.columns: wide=wide.rename(columns={sc:"Position"})
        return wide
    pc=get_player_col(df)
    if pc and pc!="Player": df=df.rename(columns={pc:"Player"})
    sc=get_pos_col(df)
    if sc and sc!="Position": df=df.rename(columns={sc:"Position"})
    return df

def _normtxt(s:str)->str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+"," ",s).strip().lower()

# --- SMART MATCHING: „Z. Iqbal“ ⇄ „Zidane Iqbal“ (+ tým) ---
def _split_name(s: str):
    """Vrátí (first_initial, surname) z libovolného formátu."""
    t = _normtxt(s).replace(".", " ")
    parts = [p for p in re.split(r"\s+", t) if p]
    if not parts: return "", ""
    surname = parts[-1]
    first = next((p for p in parts if p and p != surname), "")
    first_initial = first[0] if first else (parts[0][0] if parts else "")
    return first_initial, surname

def _norm_team(s: str) -> str:
    """Zjednoduší název klubu (odstraní FC/FK/SC/AC/… a whitespace)."""
    t = _normtxt(s)
    t = re.sub(r"\b(fk|fc|sc|ac|cf|afc|sv|us|cd|ud|bk|sk|ks)\b", " ", t)
    return re.sub(r"\s+"," ",t).strip()

def match_by_name(df: pd.DataFrame, name: str, team_hint: str = None) -> pd.DataFrame:
    """Pořadí matchů: 1) přesná očista, 2) příjmení+iniciála, 3) příjmení+tým, 4) jedinečné příjmení."""
    if df is None or df.empty or not name: return pd.DataFrame()
    pcol = get_player_col(df) or "Player"
    if pcol not in df.columns: return pd.DataFrame()

    # připrav cache klíčů
    if "_kname" not in df.columns:
        df["_kname"] = df[pcol].astype(str).map(_normtxt)
        fi, sn = zip(*df[pcol].astype(str).map(_split_name))
        df["_kfirst"] = list(fi)
        df["_ksurname"] = list(sn)
        tcol = _best_col(df, ["Team","Club","team","club"])
        df["_kteam"] = df[tcol].astype(str).map(_norm_team) if tcol else ""

    key_full = _normtxt(name)
    fi_key, sn_key = _split_name(name)
    team_key = _norm_team(team_hint) if team_hint else ""

    # 1) přesná očista
    exact = df.loc[df["_kname"] == key_full]
    if len(exact) == 1: return exact
    if len(exact) > 1 and team_key:
        pick = exact.loc[exact["_kteam"] == team_key]
        if not pick.empty: return pick

    # 2) příjmení + iniciála
    si = df.loc[(df["_ksurname"] == sn_key) & (df["_kfirst"] == fi_key)]
    if len(si) == 1: return si
    if len(si) > 1 and team_key:
        pick = si.loc[si["_kteam"] == team_key]
        if not pick.empty: return pick

    # 3) příjmení + tým
    if team_key:
        st = df.loc[(df["_ksurname"] == sn_key) & (df["_kteam"] == team_key)]
        if len(st) == 1: return st

    # 4) jedinečné příjmení
    sn = df.loc[df["_ksurname"] == sn_key]
    if len(sn) == 1: return sn

    return pd.DataFrame()

# ---------- Herní bloky ----------
DEF=[("Defensive duels per 90","Defenzivní duely /90"),("Defensive duels won, %","Úspěšnost obr. duelů %"),
     ("Interceptions per 90","Interceptions /90"),("Sliding tackles per 90","Sliding tackles /90"),
     ("Aerial duels won, %","Úspěšnost vzdušných %"),("Fouls per 90","Fauly /90")]
OFF=[("Goals per 90","Góly /90"),("xG per 90","xG /90"),("Shots on target, %","Střely na branku %"),
     ("Assists per 90","Asistence /90"),("xA per 90","xA /90"),("Shot assists per 90","Shot assists /90")]
PAS=[("Accurate passes, %","Přesnost přihrávek %"),("Key passes per 90","Klíčové přihrávky /90"),
     ("Smart passes per 90","Smart passes /90"),("Progressive passes per 90","Progresivní přihrávky /90"),
     ("Passes to final third per 90","Do finální třetiny /90"),("Cross accuracy, %","Úspěšnost centrů %"),
     ("Second assists per 90","Second assists /90")]
ONE=[("Dribbles per 90","Driblingy /90"),("Successful dribbles, %","Úspěšnost dribblingu %"),
     ("Offensive duels won, %","Úspěšnost of. duelů %"),("Progressive runs per 90","Progresivní běhy /90")]
blocks=[("Defenziva",DEF,"Defenziva"),("Ofenziva",OFF,"Ofenziva"),("Přihrávky",PAS,"Přihrávky"),("1v1",ONE,"1v1")]

ALIASES={"Cross accuracy, %":["Accurate crosses, %","Cross accuracy, %"],
         "Progressive passes per 90":["Progressive passes per 90","Progressive passes/90"],
         "Passes to final third per 90":["Passes to final third per 90","Passes to final third/90"],
         "Dribbles per 90":["Dribbles per 90","Dribbles/90"],
         "Progressive runs per 90":["Progressive runs per 90","Progressive runs/90"],
         "Second assists per 90":["Second assists per 90","Second assists/90"]}

def get_val_alias(row,key):
    if key in row.index: return row[key]
    for c in ALIASES.get(key,[]): 
        if c in row.index: return row[c]
    if key=="Cross accuracy, %" and "Accurate crosses, %" in row.index: return row["Accurate crosses, %"]
    return np.nan

def series_alias(df,key):
    if key in df.columns: return df[key]
    for c in ALIASES.get(key,[]): 
        if c in df.columns: return df[c]
    if key=="Cross accuracy, %" and "Accurate crosses, %" in df.columns: return df["Accurate crosses, %"]
    return None

def norm_metric(pop_df,key,val):
    s=series_alias(pop_df,key)
    if s is None or pd.isna(val): return np.nan
    s=pd.to_numeric(s,errors="coerce").dropna(); v=pd.to_numeric(pd.Series([val]),errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn,mx=s.min(),s.max()
    return 50.0 if mx==mn else float(np.clip((v-mn)/(mx-mn)*100,0,100))

def section_scores(row,agg,metric_weights=None):
    sec_scores,sec_idx={},{}
    for title,lst,key in blocks:
        vals={label:norm_metric(agg,eng,get_val_alias(row,eng)) for eng,label in lst}
        sec_scores[key]=vals
        if metric_weights and metric_weights.get(key):
            wsum=sum(metric_weights[key].values())
            sec_idx[key]=float(sum(v*metric_weights[key].get(lbl,0) for lbl,v in vals.items() if not pd.isna(v))/wsum) if wsum>0 else np.nan
        else:
            arr=[v for v in vals.values() if not pd.isna(v)]
            sec_idx[key]=float(np.mean(arr)) if arr else np.nan
    return sec_scores,sec_idx

def role_index(sec_idx,weights):
    acc=tot=0.0
    for k in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v=sec_idx.get(k,np.nan)
        if not pd.isna(v): 
            w=weights.get(k,0)/100.0; acc+=v*w; tot+=w
    return float(acc/tot) if tot>0 else np.nan

# ---------- Pozice + peers ----------
POS_REGEX={"CB/DF":r"(CB|DF)","RB":r"(RB)","LB":r"(LB)","WB/RWB/LWB":r"(WB|RWB|LWB)",
           "DM":r"(DM)","CM":r"(CM)","AM":r"(AM)","RW":r"(RW)","LW":r"(LW)","CF/ST":r"(CF|ST|FW)"}
def pos_group(p):
    P=(str(p) or "").upper()
    if any(x in P for x in ["CB","DF"]):return "CB/DF"
    for k in ["RB","LB","DM","CM","AM","RW","LW"]:
        if k in P: return k
    if any(x in P for x in ["RWB","LWB","WB"]):return "WB/RWB/LWB"
    if any(x in P for x in ["CF","ST","FW"]):return "CF/ST"
    return "CM"

SLAVIA_PEERS={"RB":["D. Douděra","D. Hashioka"],"LB":["O. Zmrzlý","J. Bořil"],
              "WB/RWB/LWB":["D. Douděra","D. Hashioka","O. Zmrzlý"],
              "CB/DF":["I. Ogbu","D. Zima","T. Holeš","J. Bořil"],
              "DM":["T. Holeš","O. Dorley","M. Sadílek"],
              "CM":["C. Zafeiris","L. Provod","E. Prekop","M. Sadílek"],
              "AM":["C. Zafeiris","L. Provod","E. Prekop"],
              "RW":["I. Schranz","Y. Sanyang","V. Kušej"],
              "LW":["I. Schranz","V. Kušej"],
              "CF/ST":["M. Chytil","T. Chorý"]}
def peers(pg): return SLAVIA_PEERS.get(pg,[])

def peer_avg(cz_agg, pg, weights, mweights):
    vals=[]
    for nm in peers(pg):
        if nm in cz_agg.index:
            r=cz_agg.loc[nm].copy(); r["Player"]=nm; r["Position"]=pg
            _,s=section_scores(r,cz_agg,mweights); v=role_index(s,weights)
            if not np.isnan(v): vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

# ---------- Běžecké ----------
RUN=[("Total distance per 90","Total distance /90"),
     ("High-intensity runs per 90","High-intensity runs /90"),
     ("Sprints per 90","Sprints /90"),
     ("Max speed (km/h)","Max speed (km/h)"),
     ("Average speed (km/h)","Average speed (km/h)"),
     ("Accelerations per 90","Accelerations /90"),
     ("Decelerations per 90","Decelerations /90"),
     ("High-speed distance per 90","High-speed distance /90")]
RUN_KEY="Běh"

ALIASES_RUN={
 "Total distance per 90":["Total distance per 90","Total distance/90","Distance per 90","Total distance (km) per 90","Distance P90"],
 "High-intensity runs per 90":["High-intensity runs per 90","High intensity runs per 90","High intensity runs/90","HIR/90","HI Count P90"],
 "Sprints per 90":["Sprints per 90","Sprints/90","Number of sprints per 90","Sprint Count P90"],
 "Max speed (km/h)":["Max speed (km/h)","Top speed","Max velocity","Max speed","PSV-99","TOP 5 PSV-99"],
 "Average speed (km/h)":["Average speed (km/h)","Avg speed","Average velocity","M/min P90"],
 "Accelerations per 90":["Accelerations per 90","Accelerations/90","Accels per 90","High Acceleration Count P90 + Medium Acceleration Count P90","High Acceleration Count P90","Medium Acceleration Count P90"],
 "Decelerations per 90":["Decelerations per 90","Decelerations/90","Decels per 90","High Deceleration Count P90 + Medium Deceleration Count P90","High Deceleration Count P90","Medium Deceleration Count P90"],
 "High-speed distance per 90":["High-speed distance per 90","HS distance/90","High speed distance per 90","HSR Distance P90"],
}

def run_val(row,key):
    if key in row.index: return row[key]
    for c in ALIASES_RUN.get(key,[]):
        if c in row.index: return row[c]
    return np.nan

def run_series(df,key):
    if df is None or df.empty: return None
    if key in df.columns: return df[key]
    for c in ALIASES_RUN.get(key,[]):
        if c in df.columns: return df[c]
    return None

def _post_run(df):
    if df is None or df.empty: return df
    if "Average speed (km/h)" not in df.columns and "M/min P90" in df.columns:
        df["Average speed (km/h)"]=pd.to_numeric(df["M/min P90"],errors="coerce")*0.06
    if "Accelerations per 90" not in df.columns:
        acc=[c for c in ["High Acceleration Count P90","Medium Acceleration Count P90"] if c in df.columns]
        if acc:
            s=pd.to_numeric(df[acc[0]],errors="coerce")
            for c in acc[1:]: s=s.add(pd.to_numeric(df[c],errors="coerce"),fill_value=0)
            df["Accelerations per 90"]=s
    if "Decelerations per 90" not in df.columns:
        dec=[c for c in ["High Deceleration Count P90","Medium Deceleration Count P90"] if c in df.columns]
        if dec:
            s=pd.to_numeric(df[dec[0]],errors="coerce")
            for c in dec[1:]: s=s.add(pd.to_numeric(df[c],errors="coerce"),fill_value=0)
            df["Decelerations per 90"]=s
    for c in ["Player","Team","Position"]:
        if c in df.columns: df[c]=df[c].astype(str).str.strip()
    return df

def auto_fix_run_df(run_df:pd.DataFrame, game_df:pd.DataFrame)->pd.DataFrame:
    if run_df is None or run_df.empty: return run_df
    id_map={}
    if "Player" not in run_df.columns:
        c=_best_col(run_df,["Name","player","name","Short Name"]);  id_map.update({c:"Player"} if c else {})
    if "Team" not in run_df.columns:
        c=_best_col(run_df,["Club","team","Team"]);                  id_map.update({c:"Team"} if c else {})
    if "Position" not in run_df.columns:
        c=_best_col(run_df,["Pos","Role","Primary position","position"]); id_map.update({c:"Position"} if c else {})
    if id_map: run_df=run_df.rename(columns=id_map)
    run_df=ensure_run_wide(run_df); run_df=_post_run(run_df)
    if "Position" not in run_df.columns and game_df is not None and not game_df.empty:
        g=game_df.copy()
        if "Player" not in g.columns:
            pc=_best_col(g,["Name","player","name"])
            if pc: g=g.rename(columns={pc:"Player"})
        if "Position" in g.columns and "Player" in g.columns:
            g=g[["Player","Position"]].dropna().groupby("Player",as_index=False).agg({"Position":"first"})
            run_df["_k"]=run_df["Player"].map(_normtxt); g["_k"]=g["Player"].map(_normtxt)
            run_df=run_df.merge(g[["_k","Position"]],on="_k",how="left").drop(columns=["_k"])
    return run_df

def norm_run_metric(pop,key,val):
    s=run_series(pop,key)
    if s is None or pd.isna(val): return np.nan
    s=pd.to_numeric(s,errors="coerce").dropna(); v=pd.to_numeric(pd.Series([val]),errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn,mx=s.min(),s.max()
    return 50.0 if mx==mn else float(np.clip((v-mn)/(mx-mn)*100,0,100))

def run_scores_for_row(row,pop_agg):
    if pop_agg is None or pop_agg.empty: return {RUN_KEY:{}},{},np.nan
    scores,absv={},{}
    for eng,label in RUN:
        val=run_val(row,eng)
        if pd.isna(val) and eng=="Average speed (km/h)" and "M/min P90" in row.index:
            val=pd.to_numeric(row["M/min P90"],errors="coerce")*0.06
        absv[label]=val if not pd.isna(val) else np.nan
        scores[label]=norm_run_metric(pop_agg,eng,val)
    arr=[v for v in scores.values() if not pd.isna(v)]
    return {RUN_KEY:scores},absv,(float(np.mean(arr)) if arr else np.nan)

# ---------- Render ----------
def render_card_visual(player,team,pos,age,scores,sec_index,overall_base,verdict,
                       run_scores=None,run_abs=None,run_index=np.nan,final_index=None):
    fig,ax=plt.subplots(figsize=(18,12)); ax.axis("off")
    ax.text(0.02,0.96,f"{player} (věk {age})",fontsize=20,fontweight="bold",va="top")
    ax.text(0.02,0.93,f"Klub: {team}   Pozice: {pos}",fontsize=13,va="top")
    y0=0.88
    for title,lst,key in blocks:
        ax.text(0.02,y0,title,fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
        for i,(_,lab) in enumerate(lst):
            val=scores[key].get(lab,np.nan); x=L if i%2==0 else R
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(val),alpha=0.85,lw=0))
            ax.text(x+0.005,y-0.001,f"{lab}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",fontsize=9,va="center",ha="left")
            if i%2==1: y-=0.038
        y0=y-0.025
    if run_scores is not None and RUN_KEY in run_scores and len(run_scores[RUN_KEY])>0:
        ax.text(0.02,y0,"Běžecká data",fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
        for i,(_,lab) in enumerate(RUN):
            v_pct=run_scores[RUN_KEY].get(lab,np.nan); v_abs=(run_abs or {}).get(lab,np.nan); x=L if i%2==0 else R
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(v_pct),alpha=0.85,lw=0))
            ta="n/a" if pd.isna(v_abs) else (f"{v_abs:.2f}" if isinstance(v_abs,(int,float,np.number)) else str(v_abs))
            tp="n/a" if pd.isna(v_pct) else f"{int(round(v_pct))}%"
            ax.text(x+0.005,y-0.001,f"{lab}: {ta} ({tp})",fontsize=9,va="center",ha="left")
            if i%2==1: y-=0.038
        y0=y-0.025
    ax.text(0.55,0.9,"Souhrnné indexy (0–100 %) – vážené",fontsize=16,fontweight="bold",va="top"); y=0.85
    for k in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v=sec_index.get(k,np.nan)
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(v),alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"{k}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=13,va="center",ha="left"); y-=0.075
    if not pd.isna(run_index):
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(run_index),alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"Běžecký index: {int(round(run_index))}%",fontsize=13,va="center",ha="left"); y-=0.075
    label="Celkový index (herní + běžecký)" if (final_index is not None) else "Celkový role-index (vážený)"
    v=overall_base if final_index is None else final_index
    ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(v),alpha=0.7,lw=0))
    ax.text(0.56,y-0.005,f"{label}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=14,fontweight="bold",va="center",ha="left")
    ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color='lightgrey',alpha=0.35,lw=0))
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center")
    return fig

# ---------- Sidebar váhy ----------
st.sidebar.header("⚙️ Váhy sekcí")
defaults={"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
sec_w={k:st.sidebar.slider(k,0,100,defaults[k],1) for k in defaults}
tot=sum(sec_w.values()) or 1
for k in sec_w: sec_w[k]=100.0*sec_w[k]/tot
w_run_pct=st.sidebar.slider("Váha běžeckého indexu v celkovém hodnocení",0,50,20,5)
metric_w={}
with st.sidebar.expander("Váhy metrik v sekcích (volitelné)",expanded=False):
    for title,lst,key in blocks:
        st.markdown(f"**{title}**")
        tmp={lab:st.slider(f"– {lab}",0,100,10,1,key=f"{key}_{lab}") for _,lab in lst}
        s=sum(tmp.values()); metric_w[key]=None if s==0 else {lab:w/s for lab,w in tmp.items()}

# ---------- Tabs ----------
tab_card, tab_search = st.tabs(["Karta hráče (herní + běžecká)", "Vyhledávání hráčů"])

# === TAB 1: kombinovaná karta ===
with tab_card:
    c1,c2=st.columns(2)
    with c1:
        league_file = st.file_uploader("CZ liga – herní (xlsx)", type=["xlsx"], key="league_card")
        run_cz_file = st.file_uploader("CZ běžecká data (xlsx) [pro běžeckou část]", type=["xlsx"], key="run_cz_card")
    with c2:
        players_file= st.file_uploader("Hráč/hráči – herní (xlsx)", type=["xlsx"], key="players_card")
        run_players_file=st.file_uploader("Hráč/hráči – běžecká (xlsx) [volitelné]", type=["xlsx"], key="run_players_card")

    if not league_file or not players_file:
        st.info("➡️ Nahraj minimálně CZ herní dataset + hráčský herní export."); st.stop()

    league=pd.read_excel(league_file)
    players=pd.read_excel(players_file)
    run_cz_df = auto_fix_run_df(pd.read_excel(run_cz_file), league) if run_cz_file else None
    run_pl_df = auto_fix_run_df(pd.read_excel(run_players_file), players) if run_players_file else None

    sel_player = st.selectbox("Vyber hráče (herní export)", players["Player"].dropna().unique().tolist())
    row = players.loc[players["Player"]==sel_player].iloc[0]
    player,team,pos,age = row.get("Player",""), row.get("Team",""), row.get("Position",""), row.get("Age","n/a")
    pg = pos_group(pos); rgx = POS_REGEX[pg]
    cz_group = league[league["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
    agg = cz_group.groupby("Player").mean(numeric_only=True)

    scores,sec_idx = section_scores(row, agg, metric_w)
    overall = role_index(sec_idx, sec_w)

    # ---- běžecká část (smart matching) ----
    run_scores = run_abs = None; run_index = np.nan
    if run_cz_df is not None and run_pl_df is not None:
        cand_df = match_by_name(run_pl_df, player, team_hint=team)
        # benchmark podle pozice, nebo celý když pozice chybí
        poscol = get_pos_col(run_cz_df)
        if poscol:
            cz_pos_df = run_cz_df[run_cz_df[poscol].astype(str).str.contains(rgx, na=False, regex=True)]
            cz_base = cz_pos_df if not cz_pos_df.empty else run_cz_df
        else:
            cz_base = run_cz_df
        plc = get_player_col(cz_base) or "Player"
        cz_agg = (cz_base.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_base.columns else cz_base).groupby("Player").mean(numeric_only=True)
        if not cand_df.empty and not cz_agg.empty:
            r_run = cand_df.iloc[0]
            run_scores, run_abs, run_index = run_scores_for_row(r_run, cz_agg)

    # finální index (herní + běh dle váhy)
    w_run = w_run_pct/100.0
    final_idx = (1.0-w_run)*overall + w_run*run_index if not pd.isna(run_index) else None

    peer = peer_avg(agg, pg, sec_w, metric_w)
    base = final_idx if (final_idx is not None) else overall
    verdict = "ANO – potenciální posila do Slavie" if (not np.isnan(peer) and not np.isnan(base) and base>=peer) else "NE – nedosahuje úrovně slávistických konkurentů"

    fig = render_card_visual(player,team,pos,age,scores,sec_idx,overall,verdict,run_scores,run_abs,run_index,final_index=final_idx)
    st.pyplot(fig)
    bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight")
    st.download_button("📥 Stáhnout kartu jako PNG", data=bio.getvalue(), file_name=f"{player}.png", mime="image/png")

# === TAB 2: vyhledávání (smart matching + respektuje váhu běhu) ===
with tab_search:
    st.subheader("Vyhledávání kandidátů pro Slavii (benchmark = CZ liga)")
    cA,cB=st.columns(2)
    with cA:
        cz_file = st.file_uploader("CZ liga – herní (xlsx)", type=["xlsx"], key="cz_search")
        run_cz_file = st.file_uploader("CZ běžecká data (xlsx) [volitelné]", type=["xlsx"], key="cz_run_search")
    with cB:
        fr_file = st.file_uploader("Cizí liga – herní (xlsx)", type=["xlsx"], key="fr_search")
        run_fr_file = st.file_uploader("Cizí liga – běžecká (xlsx) [volitelné]", type=["xlsx"], key="fr_run_search")
    if cz_file: st.session_state["cz_bytes"]=cz_file.getvalue()
    if fr_file: st.session_state["fr_bytes"]=fr_file.getvalue()
    if run_cz_file: st.session_state["cz_run_bytes"]=run_cz_file.getvalue()
    if run_fr_file: st.session_state["fr_run_bytes"]=run_fr_file.getvalue()

    pos_opts=list(POS_REGEX.keys())
    pos_sel=st.multiselect("Pozice",pos_opts,default=pos_opts,key="search_positions")
    c1,c2,c3=st.columns(3)
    with c1: league_name=st.text_input("Název ligy",value="Cizí liga",key="search_league")
    with c2: min_minutes=st.number_input("Min. minut (pokud ve zdroji)",0,step=100,key="search_min_minutes")
    with c3: min_games  =st.number_input("Min. zápasů (pokud ve zdroji)",0,step=1,key="search_min_games")
    run=st.button("Spustit vyhledávání",key="search_run")

    if run:
        if "cz_bytes" not in st.session_state or "fr_bytes" not in st.session_state:
            st.error("Nahraj alespoň CZ herní + cizí liga herní."); st.stop()
        cz_df=load_xlsx(st.session_state["cz_bytes"])
        fr_df=load_xlsx(st.session_state["fr_bytes"])
        cz_run_df=auto_fix_run_df(load_xlsx(st.session_state["cz_run_bytes"]),cz_df) if "cz_run_bytes" in st.session_state else None
        fr_run_df=auto_fix_run_df(load_xlsx(st.session_state["fr_run_bytes"]),fr_df) if "fr_run_bytes" in st.session_state else None
        w_run = w_run_pct/100.0

        def search_candidates():
            mask=pd.Series(False,index=fr_df.index)
            for p in pos_sel:
                rgx=POS_REGEX[p]; mask|=fr_df["Position"].astype(str).str.contains(rgx,na=False,regex=True)
            base=fr_df.loc[mask].copy()
            def pick(df,names): return next((n for n in names if n in df.columns),None)
            mc=pick(base,["Minutes","Minutes played","Min"]); gc=pick(base,["Games","Matches"])
            if min_minutes and mc: base=base[pd.to_numeric(base[mc],errors="coerce").fillna(0)>=min_minutes]
            if min_games   and gc: base=base[pd.to_numeric(base[gc],errors="coerce").fillna(0)>=min_games]
            rows=[]; cards=[]
            for _,r in base.iterrows():
                pg=pos_group(r.get("Position","")); rgx=POS_REGEX[pg]
                cz_pos=cz_df[cz_df["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
                if cz_pos.empty: continue
                cz_agg=cz_pos.groupby("Player").mean(numeric_only=True)
                scores,sec_idx=section_scores(r,cz_agg,metric_w); overall=role_index(sec_idx,sec_w)

                run_idx=np.nan; r_scores=None; r_abs=None
                if cz_run_df is not None and fr_run_df is not None:
                    cand_df=match_by_name(fr_run_df, r.get("Player",""), team_hint=r.get("Team",""))
                    poscol=get_pos_col(cz_run_df)
                    if poscol:
                        cz_run_pos=cz_run_df[cz_run_df[poscol].astype(str).str.contains(rgx,na=False,regex=True)]
                        cz_base=cz_run_pos if not cz_run_pos.empty else cz_run_df
                    else:
                        cz_base=cz_run_df
                    plc=get_player_col(cz_base) or "Player"
                    cz_run_agg=(cz_base.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_base.columns else cz_base).groupby("Player").mean(numeric_only=True)
                    if not cand_df.empty and not cz_run_agg.empty:
                        r_run=cand_df.iloc[0]
                        r_scores,r_abs,run_idx = run_scores_for_row(r_run,cz_run_agg)

                final_idx = (1.0-w_run)*overall + w_run*run_idx if (not pd.isna(run_idx) and w_run>0) else overall
                pavg=peer_avg(cz_agg,pg,sec_w,metric_w)
                verdict="ANO – potenciální posila do Slavie" if (not np.isnan(pavg) and not np.isnan(final_idx) and final_idx>=pavg) else "NE – nedosahuje úrovně slávistických konkurentů"

                if verdict.startswith("ANO"):
                    player=r.get("Player",""); team=r.get("Team",""); pos=r.get("Position",""); age=r.get("Age","n/a")
                    rows.append({"Hráč":player,"Věk":age,"Klub":team,"Pozice":pos,"Liga":league_name,
                                 "Index Def":sec_idx.get("Defenziva",np.nan),"Index Off":sec_idx.get("Ofenziva",np.nan),
                                 "Index Pass":sec_idx.get("Přihrávky",np.nan),"Index 1v1":sec_idx.get("1v1",np.nan),
                                 "Role-index (vážený)":overall,"Run index":run_idx,"Final index":final_idx,"Verdikt":verdict})
                    fig=render_card_visual(player,team,pos,age,scores,sec_idx,overall,verdict,r_scores,r_abs,run_idx,final_index=(final_idx if not pd.isna(run_idx) else None))
                    bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight"); plt.close(fig)
                    cards.append((str(player),bio.getvalue()))
            return pd.DataFrame(rows),cards

        res_df,cards=search_candidates()
        st.session_state.update(search_results=res_df,search_cards=cards,fr_df=fr_df,cz_df=cz_df,fr_run_df=fr_run_df,cz_run_df=cz_run_df)

    res_df=st.session_state.get("search_results")
    if res_df is None or res_df.empty:
        st.info("Zatím žádné výsledky – nahraj soubory a klikni na *Spustit vyhledávání*.")
    else:
        st.success(f"Nalezeno kandidátů: {len(res_df)}"); st.dataframe(res_df, use_container_width=True)
        st.download_button("📥 Stáhnout CSV s kandidáty",
            data=res_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"kandidati_{st.session_state.get('search_league','liga')}.csv", mime="text/csv")
        zbuf=BytesIO()
        with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
            for name,png in (st.session_state.get("search_cards") or []):
                safe=str(name).replace("/","_").replace("\\","_"); zf.writestr(f"{safe}.png", png)
        st.download_button("🗂️ Stáhnout všechny karty (ZIP)", data=zbuf.getvalue(),
                           file_name=f"karty_{st.session_state.get('search_league','liga')}.zip", mime="application/zip")
        sel=st.selectbox("Zobraz kartu hráče", res_df["Hráč"].tolist())
        if sel:
            fr_df=st.session_state.get("fr_df"); cz_df=st.session_state.get("cz_df")
            fr_run_df=st.session_state.get("fr_run_df"); cz_run_df=st.session_state.get("cz_run_df")
            r=fr_df.loc[fr_df["Player"]==sel].iloc[0]; pg=pos_group(r.get("Position","")); rgx=POS_REGEX[pg]
            cz_pos=cz_df[cz_df["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
            if not cz_pos.empty:
                cz_agg=cz_pos.groupby("Player").mean(numeric_only=True)
                scores,sec_idx=section_scores(r,cz_agg,metric_w); overall=role_index(sec_idx,sec_w)
                run_idx=np.nan; run_scores=None; run_abs=None
                if cz_run_df is not None and fr_run_df is not None:
                    cand_df=match_by_name(fr_run_df, sel, team_hint=r.get("Team",""))
                    poscol=get_pos_col(cz_run_df)
                    if poscol:
                        cz_run_pos=cz_run_df[cz_run_df[poscol].astype(str).str.contains(rgx,na=False,regex=True)]
                        cz_base=cz_run_pos if not cz_run_pos.empty else cz_run_df
                    else:
                        cz_base=cz_run_df
                    plc=get_player_col(cz_base) or "Player"
                    cz_run_agg=(cz_base.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_base.columns else cz_base).groupby("Player").mean(numeric_only=True)
                    if not cand_df.empty and not cz_run_agg.empty:
                        r_run=cand_df.iloc[0]; run_scores,run_abs,run_idx=run_scores_for_row(r_run,cz_run_agg)
                final_idx=(1.0-w_run_pct/100.0)*overall + (w_run_pct/100.0)*run_idx if not pd.isna(run_idx) else None
                peer=peer_avg(cz_agg,pg,sec_w,metric_w); base=final_idx if (final_idx is not None) else overall
                verdict="ANO – potenciální posila do Slavie" if (not np.isnan(peer) and not np.isnan(base) and base>=peer) else "NE – nedosahuje úrovně slávistických konkurentů"
                fig=render_card_visual(r.get("Player",""),r.get("Team",""),r.get("Position",""),r.get("Age","n/a"),
                                       scores,sec_idx,overall,verdict,run_scores,run_abs,run_idx,final_index=final_idx)
                st.pyplot(fig)



