# app.py — Slavia datacards (kratší verze, stejná funkčnost) + ANALYTICKÝ MODUL
import re, unicodedata, zipfile
from io import BytesIO
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

st.set_page_config(page_title="Karty – Slavia", layout="wide")
st.title("⚽ Generátor datových karet (herní + běžecká)")

# ---------- Utils ----------
@st.cache_data
def load_xlsx(b: bytes) -> pd.DataFrame: return pd.read_excel(BytesIO(b))

def color_for(v):
    if pd.isna(v): return "lightgrey"
    v=float(v);  return "#FF0000" if v<=20 else "#FF8C00" if v<=40 else "#FFD700" if v<=60 else "#90EE90" if v<=80 else "#006400"

def _best_col(df, names): return next((c for c in names if c in df.columns), None)
def _normtxt(s): return re.sub(r"\s+"," ","".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))).strip().lower()
def _split_name(s):
    t=_normtxt(s).replace("."," "); ps=[x for x in re.split(r"\s+",t) if x]
    if not ps: return "",""
    sur=ps[-1]; first=next((x for x in ps if x!=sur), "");  return (first[0] if first else (ps[0][0] if ps else "")), sur
def _norm_team(s): return re.sub(r"\s+"," ",re.sub(r"\b(fk|fc|sc|ac|cf|afc|sv|us|cd|ud|bk|sk|ks|ucl|ii|b)\b"," ",_normtxt(s))).strip()
def _norm_nat(s): return _normtxt(s)

def get_player_col(df): return _best_col(df,["Player","Name","player","name","Short Name"])
def get_team_col(df):   return _best_col(df,["Team","Club","team","club"])
def get_pos_col(df):    return _best_col(df,["Position","Pos","position","Role","Primary position"])
def get_age_col(df):    return _best_col(df,["Age","age","AGE"])
def get_nat_col(df):    return _best_col(df,["Nationality","Nationality 1","Nation","Country","Citizenship","Nat"])
def is_slavia(team:str) -> bool:
    t=_norm_team(team or ""); return ("slavia" in t) and ("praha" in t or "prague" in t)

def normalize_core_cols(df):
    if df is None or df.empty: return df
    m={}
    for src,dst in [(get_player_col(df),"Player"),(get_team_col(df),"Team"),(get_pos_col(df),"Position")]:
        if src and src!=dst: m[src]=dst
    return df.rename(columns=m) if m else df

def ensure_run_wide(df):
    if df is None or df.empty: return df
    if {"Metric","Value"}.issubset(df.columns):
        idx=[c for c in [get_player_col(df) or "Player",get_team_col(df),get_pos_col(df),"Age"] if c and c in df.columns]
        wide=df.pivot_table(index=idx,columns="Metric",values="Value",aggfunc="mean").reset_index()
        for c,d in [(get_player_col(df),"Player"),(get_pos_col(df),"Position")]:
            if c and c!=d and c in wide.columns: wide=wide.rename(columns={c:d})
        return wide
    return normalize_core_cols(df)

# ---------- Matching ----------
def match_by_name(df, name, team_hint=None, age_hint=None, nat_hint=None, min_score=8, require_surname=True):
    if df is None or df.empty or not name: return pd.DataFrame()
    pcol=get_player_col(df) or "Player"
    if pcol not in df.columns: return pd.DataFrame()
    if "_kname" not in df.columns:
        df["_kname"]=df[pcol].astype(str).map(_normtxt)
        fi,sn=zip(*df[pcol].astype(str).map(_split_name)); df["_kfirst"],df["_ksurname"]=list(fi),list(sn)
        tcol=get_team_col(df); df["_kteam"]=df[tcol].astype(str).map(_norm_team) if tcol else ""
        acol=get_age_col(df);  df["_kage"]=pd.to_numeric(df[acol],errors="coerce").astype("Int64") if acol else pd.Series([pd.NA]*len(df),dtype="Int64")
        ncol=get_nat_col(df);  df["_knat"]=df[ncol].astype(str).map(_norm_nat) if ncol else ""
    key_full=_normtxt(name); fi_key,sn_key=_split_name(name)
    team_key=_norm_team(team_hint) if team_hint else ""; nat_key=_norm_nat(nat_hint) if nat_hint else ""
    try: age_key=int(age_hint) if age_hint is not None else None
    except: age_key=None
    exact=df.loc[df["_kname"]==key_full]
    if len(exact)==1: return exact
    if len(exact)>1 and team_key:
        pick=exact.loc[exact["_kteam"]==team_key]
        if len(pick)==1: return pick
    pool=df.loc[df["_ksurname"]==sn_key].copy()
    if pool.empty: return pd.DataFrame()
    def score(r):
        s=4+(4 if fi_key and r["_kfirst"]==fi_key else 0)  # surname=+4
        if team_key: s+=4 if r["_kteam"]==team_key else (2 if team_key in r["_kteam"] or r["_kteam"] in team_key else 0)
        if age_key is not None and not pd.isna(r["_kage"]):
            d=abs(int(r["_kage"])-age_key); s+=3 if d==0 else 2 if d==1 else 1 if d==2 else 0
        if nat_key and r["_knat"]==nat_key: s+=2
        return s
    pool["_score"]=pool.apply(score,axis=1)
    best=pool.sort_values(["_score","_kage"],ascending=[False,True]).head(1)
    if best.empty or (require_surname and best["_ksurname"].iloc[0]!=sn_key) or best["_score"].iloc[0] < min_score: return pd.DataFrame()
    return best

# ---------- Bloky ----------
DEF=[("Defensive duels per 90","Defenzivní duely /90"),("Defensive duels won, %","Úspěšnost obr. duelů %"),
     ("Interceptions per 90","Interceptions /90"),("Sliding tackles per 90","Sliding tackles /90"),
     ("Aerial duels won, %","Úspěšnost vzdušných %"),("Fouls per 90","Fauly /90")]
OFF=[("Goals per 90","Góly /90"),("xG per 90","xG /90"),("Shots on target, %","Střely na branku %"),
     ("Assists per 90","Asistence /90"),("xA per 90","xA /90"),("Shot assists per 90","Shot assists /90")]
PAS=[("Accurate passes, %","Přesnost přihrávek %"),("Key passes per 90","Klíčové přihrávky /90"),
     ("Smart passes per 90","Smart passes /90"),("Progressive passes per 90","Progresivní přihrávek /90"),
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

def series_alias(df,key):
    if key in df.columns: return df[key]
    for c in ALIASES.get(key,[]): 
        if c in df.columns: return df[c]
    if key=="Cross accuracy, %" and "Accurate crosses, %" in df.columns: return df["Accurate crosses, %"]
    return None
def get_val_alias(row,key):
    if key in row.index: return row[key]
    for c in ALIASES.get(key,[]):
        if c in row.index: return row[c]
    if key=="Cross accuracy, %" and "Accurate crosses, %" in row.index: return row["Accurate crosses, %"]
    return np.nan

def norm_metric(pop,key,val):
    s=series_alias(pop,key)
    if s is None or pd.isna(val): return np.nan
    s=pd.to_numeric(s,errors="coerce").dropna(); v=pd.to_numeric(pd.Series([val]),errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn,mx=s.min(),s.max();  return 50.0 if mx==mn else float(np.clip((v-mn)/(mx-mn)*100,0,100))

def section_scores(row, agg, metric_w=None):
    # OPRAVENÁ VERZE (žádné sec_scores,sec_idx={})
    sec_scores = {}
    sec_idx = {}
    for _, lst, key in blocks:
        vals = {lab: norm_metric(agg, eng, get_val_alias(row, eng)) for eng, lab in lst}
        sec_scores[key] = vals
        if metric_w and metric_w.get(key):
            w = metric_w[key]; wsum = sum(w.values()) or 1
            sec_idx[key] = float(sum(v * w.get(l, 0) for l, v in vals.items() if not pd.isna(v)) / wsum)
        else:
            arr = [v for v in vals.values() if not pd.isna(v)]
            sec_idx[key] = float(np.mean(arr)) if arr else np.nan
    return sec_scores, sec_idx

def role_index(sec_idx,weights):
    acc=tot=0.0
    for k in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v=sec_idx.get(k,np.nan)
        if not pd.isna(v):
            w=weights.get(k,0)/100.0; acc+=v*w; tot+=w
    return float(acc/tot) if tot>0 else np.nan

# ---------- Pozice / Role5 ----------
POS_REGEX={"CB/DF":r"(CB|DF)","RB":r"(RB)","LB":r"(LB)","WB/RWB/LWB":r"(WB|RWB|LWB)","DM":r"(DM)","CM":r"(CM)","AM":r"(AM)","RW":r"(RW)","LW":r"(LW)","CF/ST":r"(CF|ST|FW)"}
def pos_group(p):
    P=(str(p) or "").upper()
    if any(x in P for x in ["CB","DF"]):return "CB/DF"
    for k in ["RB","LB","DM","CM","AM","RW","LW"]:
        if k in P: return k
    if any(x in P for x in ["RWB","LWB","WB"]):return "WB/RWB/LWB"
    if any(x in P for x in ["CF","ST","FW"]):return "CF/ST"
    return "CM"
WYS_TO_ROLE={"RCB":"CB","LCB":"CB","RCB3":"CB","LCB3":"CB","CB":"CB","RB":"RB","RB5":"RB","LB":"RB","LB5":"RB","RWB":"RB","LWB":"RB","WB":"RB",
             "DMF":"CM","RDMF":"CM","LDMF":"CM","RCMF":"CM","LCMF":"CM","RCMF3":"CM","LCMF3":"CM","AMF":"CM","DM":"CM","CM":"CM","AM":"CM",
             "RAMF":"RW","LAMF":"RW","RW":"RW","LW":"RW","AMFL":"RW","AMFR":"RW","LWF":"RW","RWF":"RW","W":"RW","WINGER":"RW",
             "CF":"CF","ST":"CF","FW":"CF","FORWARD":"CF","STRIKER":"CF"}
ROLE_PATTERNS=[("CB",r"(CB|CENTRE\s*BACK|CENTER\s*BACK|CENTRAL\s*DEF(ENDER)?|DEF(ENDER)?\b(?!.*MID))"),
               ("RB",r"(RB|LB|RWB|LWB|WB|FULL\s*BACK|WING\s*BACK)"),
               ("CM",r"(DMF|CMF|AMF|CM|AM|MIDFIELDER|MID)"),
               ("RW",r"(RW|LW|WINGER|W(?!B)\b|RIGHT\s*WING|LEFT\s*WING)"),
               ("CF",r"(CF|ST|FW|FORWARD|STRIKER|CENTRE\s*FORWARD|CENTER\s*FORWARD)")]
def _primary_wyscout_tag(pos_text:str) -> str: return "" if not pos_text else str(pos_text).split(",")[0].strip().upper()
def role5_from_pos_text(pos_text:str) -> str:
    if not pos_text: return ""
    first=_primary_wyscout_tag(pos_text)
    if first in WYS_TO_ROLE: return WYS_TO_ROLE[first]
    U=first.upper()
    for k in WYS_TO_ROLE:
        if k in U: return WYS_TO_ROLE[k]
    for role,pat in ROLE_PATTERNS:
        if re.search(pat, U, flags=re.IGNORECASE): return role
    return ""
def ensure_role5_column(df):
    if df is None or df.empty: return df
    if "Role5" not in df.columns: df["Role5"]=np.nan
    if "Position" in df.columns:
        mask=df["Role5"].isna() | (df["Role5"].astype(str).str.strip()=="")
        df.loc[mask,"Role5"]=df.loc[mask,"Position"].astype(str).map(role5_from_pos_text)
    return df
def _role5_or_none(x):
    if x is None: return None
    if isinstance(x,float) and np.isnan(x): return None
    s=str(x).strip().upper(); return s if s else None

def _attach_role5_from_game(run_df, game_df):
    if run_df is None or run_df.empty: return run_df
    run_df=ensure_role5_column(run_df)
    if game_df is None or game_df.empty: return ensure_role5_column(run_df)
    g=normalize_core_cols(game_df.copy())
    if "Player" not in g.columns or "Position" not in g.columns: return ensure_role5_column(run_df)
    tmp=g[["Player","Position"]].dropna().copy(); tmp["Role5"]=tmp["Position"].astype(str).map(role5_from_pos_text)
    fi,sur=zip(*tmp["Player"].map(_split_name)); tmp["_k"]=pd.Series(fi,index=tmp.index)+"|"+pd.Series(sur,index=tmp.index)
    tmp=tmp.dropna(subset=["Role5","_k"]).groupby("_k",as_index=False).agg({"Role5":"first"})
    fi2,sur2=zip(*run_df["Player"].astype(str).map(_split_name)); run_df["_k"]=pd.Series(fi2,index=run_df.index)+"|"+pd.Series(sur2,index=run_df.index)
    out=run_df.merge(tmp[["_k","Role5"]],on="_k",how="left",suffixes=("","_g"))
    need=out["Role5"].isna() & out["Role5_g"].notna(); out.loc[need,"Role5"]=out.loc[need,"Role5_g"]
    return ensure_role5_column(out.drop(columns=["_k","Role5_g"],errors="ignore"))

# ---------- Running ----------
RUN=[("Total distance per 90","Total distance /90"),("High-intensity runs per 90","High-intensity runs /90"),
     ("Sprints per 90","Sprints /90"),("Max speed (km/h)","Max speed (km/h)"),
     ("Average speed (km/h)","Average speed (km/h)"),("Accelerations per 90","Accelerations /90"),
     ("Decelerations per 90","Decelerations /90"),("High-speed distance per 90","High-speed distance /90")]
RUN_KEY="Běh"
ALIASES_RUN={"Total distance per 90":["Total distance per 90","Total distance/90","Distance per 90","Total distance (km) per 90","Distance P90"],
 "High-intensity runs per 90":["High-intensity runs per 90","High intensity runs per 90","High intensity runs/90","HIR/90","HI Count P90"],
 "Sprints per 90":["Sprints per 90","Sprints/90","Number of sprints per 90","Sprint Count P90"],
 "Max speed (km/h)":["Max speed (km/h)","Top speed","Max velocity","Max speed","PSV-99","TOP 5 PSV-99"],
 "Average speed (km/h)":["Average speed (km/h)","Avg speed","Average velocity","M/min P90"],
 "Accelerations per 90":["Accelerations per 90","Accelerations/90","Accels per 90","High Acceleration Count P90 + Medium Acceleration Count P90","High Acceleration Count P90","Medium Acceleration Count P90"],
 "Decelerations per 90":["Decelerations per 90","Decelerations/90","Decels per 90","High Deceleration Count P90 + Medium Deceleration Count P90","High Deceleration Count P90","Medium Deceleration Count P90"],
 "High-speed distance per 90":["High-speed distance per 90","HS distance/90","High speed distance per 90","HSR Distance P90"]}
def run_series(df,key):
    if df is None or df.empty: return None
    if key in df.columns: return df[key]
    for c in ALIASES_RUN.get(key,[]): 
        if c in df.columns: return df[c]
    return None
def run_val(row,key):
    if key in row.index: return row[key]
    for c in ALIASES_RUN.get(key,[]):
        if c in row.index: return row[c]
    return np.nan
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

def auto_fix_run_df(run_df, game_df):
    if run_df is None or run_df.empty: return run_df
    id_map={}
    if "Player" not in run_df.columns:   c=_best_col(run_df,["Name","player","name","Short Name"]);  id_map.update({c:"Player"} if c else {})
    if "Team"   not in run_df.columns:   c=_best_col(run_df,["Club","team","Team"]);                  id_map.update({c:"Team"} if c else {})
    if "Position" not in run_df.columns: c=_best_col(run_df,["Pos","Role","Primary position","position"]); id_map.update({c:"Position"} if c else {})
    if id_map: run_df=run_df.rename(columns=id_map)
    run_df=ensure_run_wide(run_df); run_df=_post_run(run_df)
    run_df=_attach_role5_from_game(run_df, game_df)
    return ensure_role5_column(run_df)

def norm_run_metric(pop,key,val):
    s=run_series(pop,key)
    if s is None or pd.isna(val): return np.nan
    s=pd.to_numeric(s,errors="coerce").dropna(); v=pd.to_numeric(pd.Series([val]),errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn,mx=s.min(),s.max();  return 50.0 if mx==mn else float(np.clip((v-mn)/(mx-mn)*100,0,100))

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
def render_card_visual(player,team,pos,age,scores,sec_index,overall_base,verdict,run_scores=None,run_abs=None,run_index=np.nan,final_index=None, role5=None):
    fig,ax=plt.subplots(figsize=(18,12)); ax.axis("off")
    ax.text(0.02,0.96,f"{player} (věk {age})",fontsize=20,fontweight="bold",va="top")
    ax.text(0.02,0.93,f"Klub: {team}   Pozice: {pos}{('   Role (běh): '+role5) if role5 else ''}",fontsize=13,va="top")
    y0=0.88
    for title,lst,key in blocks:
        ax.text(0.02,y0,title,fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
        for i,(_,lab) in enumerate(lst):
            val=scores[key].get(lab,np.nan); x=L if i%2==0 else R
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(val),alpha=0.85,lw=0))
            ax.text(x+0.005,y-0.001,f"{lab}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",fontsize=9,va="center",ha="left")
            if i%2==1: y-=0.038
        y0=y-0.025
    if run_scores and RUN_KEY in run_scores and len(run_scores[RUN_KEY])>0 and not pd.isna(run_index):
        ax.text(0.02,y0,f"Běžecká data (vs. CZ benchmark{f' – role {role5}' if role5 else ''})",fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
        for i,(_,lab) in enumerate(RUN):
            p=run_scores[RUN_KEY].get(lab,np.nan); a=(run_abs or {}).get(lab,np.nan); x=L if i%2==0 else R
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(p),alpha=0.85,lw=0))
            ta="n/a" if pd.isna(a) else (f"{a:.2f}" if isinstance(a,(int,float,np.number)) else str(a))
            tp="n/a" if pd.isna(p) else f"{int(round(p))}%"
            ax.text(x+0.005,y-0.001,f"{lab}: {ta} ({tp})",fontsize=9,va="center",ha="left")
            if i%2==1: y-=0.038
        y0=y-0.025
    ax.text(0.55,0.9,"Souhrnné indexy (0–100 %) – vážené",fontsize=16,fontweight="bold",va="top"); y=0.85
    for k in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v=sec_index.get(k,np.nan); ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(v),alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"{k}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=13,va="center",ha="left"); y-=0.075
    if not pd.isna(run_index):
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(run_index),alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"Běžecký index: {int(round(run_index))}%",fontsize=13,va="center",ha="left"); y-=0.075
    v=overall_base if final_index is None else final_index
    ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(v),alpha=0.7,lw=0))
    ax.text(0.56,y-0.005,f"{'Celkový index (herní + běžecký)' if final_index is not None else 'Celkový role-index (vážený)'}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=14,fontweight="bold",va="center",ha="left")
    ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color='lightgrey',alpha=0.35,lw=0))
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center")
    return fig

def render_run_card(player,team,pos,age,run_scores,run_abs,run_index,verdict,role5=None):
    fig,ax=plt.subplots(figsize=(14,8)); ax.axis("off")
    ax.text(0.02,0.95,f"{player} (věk {age})",fontsize=20,fontweight="bold",va="top")
    ax.text(0.02,0.92,f"Klub: {team or '—'}   Pozice: {pos or '—'}   Role (běh): {role5 or '—'}",fontsize=13,va="top")
    ax.text(0.02,0.86,f"Běžecká data (vs. CZ benchmark{f' – role {role5}' if role5 else ''})",fontsize=15,fontweight="bold",va="top")
    y=0.82; L,R=0.04,0.36
    for i,(_,lab) in enumerate(RUN):
        pct=run_scores[RUN_KEY].get(lab,np.nan); val=(run_abs or {}).get(lab,np.nan)
        x=L if i%2==0 else R
        ax.add_patch(Rectangle((x,y-0.03),0.28,0.05,color=color_for(pct),alpha=0.8,lw=0))
        ta="n/a" if pd.isna(val) else (f"{val:.2f}" if isinstance(val,(int,float,np.number)) else str(val))
        tp="n/a" if pd.isna(pct) else f"{int(round(pct))}%"
        ax.text(x+0.01,y-0.006,f"{lab}: {ta} ({tp})",fontsize=10,va="center",ha="left")
        if i%2==1: y-=0.07
    ax.text(0.7,0.86,"Souhrn",fontsize=15,fontweight="bold",va="top")
    ax.add_patch(Rectangle((0.7,0.79),0.26,0.06,color=color_for(run_index),alpha=0.8,lw=0))
    ax.text(0.71,0.81,f"Běžecký index: {'n/a' if pd.isna(run_index) else str(int(round(run_index)))+'%'}",fontsize=13,va="center",ha="left")
    ax.add_patch(Rectangle((0.7,0.12),0.26,0.06,color='lightgrey',alpha=0.5,lw=0))
    ax.text(0.83,0.15,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center")
    return fig

# ---------- Sidebar ----------
st.sidebar.header("⚙ Váhy sekcí")
base_w={"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
sec_w={k:st.sidebar.slider(k,0,100,base_w[k],1) for k in base_w}
tot=sum(sec_w.values()) or 1
for k in sec_w: sec_w[k]=100.0*sec_w[k]/tot
w_run_pct=st.sidebar.slider("Váha běžeckého indexu v celkovém hodnocení",0,50,20,5)
th_agg=st.sidebar.selectbox("Prahování vs Slavia – statistika",["Medián","Průměr"],index=0)

metric_w={}
with st.sidebar.expander("Váhy metrik v sekcích (volitelné)",False):
    for _,lst,key in blocks:
        st.markdown(f"{key}")
        tmp={lab:st.slider(f"– {lab}",0,100,10,1,key=f"{key}_{lab}") for _,lab in lst}
        s=sum(tmp.values()) or 1; metric_w[key]={lab:w/s for lab,w in tmp.items()} if s else None

# ---------- Výpočty ----------
def compute_overall_and_run(row, league_agg, run_cz_df, run_df_for_row, team_hint, age_hint, nat_hint, pos_text):
    scores,sec_idx=section_scores(row,league_agg,metric_w); overall=role_index(sec_idx,sec_w)
    role5=_role5_or_none(role5_from_pos_text(pos_text))
    run_scores=run_abs=None; run_idx=np.nan
    if (run_cz_df is not None) and (run_df_for_row is not None) and role5:
        cand=match_by_name(run_df_for_row, row.get("Player",""), team_hint=team_hint, age_hint=age_hint, nat_hint=nat_hint, min_score=8, require_surname=True)
        cz_base=run_cz_df[run_cz_df.get("Role5","").astype(str).str.upper()==role5]
        if not cand.empty and not cz_base.empty:
            plc=get_player_col(cz_base) or "Player"
            cz_agg=(cz_base.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_base.columns else cz_base).groupby("Player").mean(numeric_only=True)
            run_scores,run_abs,run_idx=run_scores_for_row(cand.iloc[0],cz_agg)
    return scores,sec_idx,overall,run_scores,run_abs,run_idx,role5

def final_from_overall_and_run(overall, run_idx, w_run): return (1.0-w_run)*overall + w_run*run_idx if not pd.isna(run_idx) else overall

def compute_slavia_role_thresholds(cz_game_df, cz_run_df, w_run, how="Medián"):
    thr={}
    if cz_game_df is None or cz_game_df.empty: return thr
    g=normalize_core_cols(cz_game_df.copy())
    if not {"Player","Team","Position"}.issubset(g.columns): return thr
    slv=g[g["Team"].astype(str).map(is_slavia)]
    if slv.empty: return thr
    vals=[]
    for _,r in slv.iterrows():
        pos=r.get("Position",""); pg=pos_group(pos); rgx=POS_REGEX[pg]
        cz_pos=g[g["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
        if cz_pos.empty: continue
        agg=cz_pos.groupby("Player").mean(numeric_only=True)
        scores,sec_idx=section_scores(r,agg,metric_w); overall=role_index(sec_idx,sec_w)
        role5=_role5_or_none(role5_from_pos_text(pos)); run_idx=np.nan
        if cz_run_df is not None and role5:
            base=cz_run_df[cz_run_df.get("Role5","").astype(str).str.upper()==role5]
            if not base.empty and get_player_col(base):
                pcol=get_player_col(base)
                rows=base[base[pcol].astype(str).map(_normtxt)==_normtxt(r.get("Player",""))]
                if rows.empty:
                    fi,sn=_split_name(r.get("Player","")); rows=base[base[pcol].astype(str).map(lambda x: _split_name(x)==(fi,sn))]
                if not rows.empty:
                    cz_agg=base.groupby("Player").mean(numeric_only=True)
                    run_scores,run_abs,run_idx=run_scores_for_row(rows.iloc[0],cz_agg)
        final=final_from_overall_and_run(overall,run_idx,w_run)
        if not pd.isna(final) and role5: vals.append((role5,float(final)))
    if not vals: return thr
    df=pd.DataFrame(vals,columns=["Role5","Final"])
    thr={k:float(v["Final"].mean() if how=="Průměr" else v["Final"].median()) for k,v in df.groupby("Role5")}
    return thr

# ---------- ANALYTICKÝ MODUL ----------
def _lvl(v):
    """Lidsky čitelná úroveň: '68 % (nadprůměr)' nebo 'n/a'."""
    return "n/a" if pd.isna(v) else f"{_pct(v)} % ({_band(v)})"

def _is_high(v, th=65):
    return (not pd.isna(v)) and (_pct(v) >= th)

def _band(p):
    if pd.isna(p): return "n/a"
    p=float(p)
    if p>=80: return "elitní"
    if p>=70: return "výborný"
    if p>=60: return "nadprůměr"
    if p>=50: return "průměr"
    if p>=40: return "podprůměr"
    return "slabý"

def _topk(d, k=3, reverse=True):
    items=[(k_, v) for k_, v in d.items() if not pd.isna(v)]
    items.sort(key=lambda x: x[1], reverse=reverse)
    return items[:k]

def _fmt_metric_list(pairs):
    """pairs = [(label, value), ...] -> 'Label 75%, Label 62%, ...' nebo 'n/a'"""
    vals = [f"{k} {int(round(v))}%" for k, v in pairs if not pd.isna(v)]
    return ", ".join(vals) if vals else "n/a"

# stručné vysvětlení metrik pro report
_METRIC_EXPL = {
    "Góly /90":"gólová produkce",
    "xG /90":"kvalita šancí (xG)",
    "Asistence /90":"připravené góly",
    "xA /90":"tvorba šancí (xA)",
    "Shot assists /90":"přihrávky do střel",
    "Klíčové přihrávky /90":"klíčová finalita",
    "Smart passes /90":"kreativní průniky",
    "Progresivní přihrávek /90":"posouvání hry dopředu",
    "Do finální třetiny /90":"zásobování finální třetiny",
    "Úspěšnost centrů %":"kvalita centrů",
    "Second assists /90":"před-asistence",
    "Driblingy /90":"objem 1v1",
    "Úspěšnost dribblingu %":"efektivita 1v1",
    "Úspěšnost of. duelů %":"síla v ofenzivních duelech",
    "Progresivní běhy /90":"nesení míče vpřed",
    "Defenzivní duely /90":"objem def. soubojů",
    "Úspěšnost obr. duelů %":"úspěšnost def. soubojů",
    "Interceptions /90":"čtení hry / zachycení",
    "Sliding tackles /90":"slide zákroky",
    "Úspěšnost vzdušných %":"vzdušné souboje",
    "Fauly /90":"faulovost",
    # běh:
    "Total distance /90":"celková zátěž",
    "High-intensity runs /90":"počet HI běhů",
    "Sprints /90":"sprintový objem",
    "Max speed (km/h)":"maximální rychlost",
    "Average speed (km/h)":"průměrná rychlost",
    "Accelerations /90":"akcelerace",
    "Decelerations /90":"decelerace",
    "High-speed distance /90":"vzdál. ve vysoké rychlosti",
}

def _build_long_scout_report(player, team, pos, role5,
                             sec_scores, sec_idx, overall,
                             run_scores, run_idx, final_idx, thr_slavia, archetype):
    # zkratky pro čtení metrik (v procentech 0–100 z našeho srovnání)
    def g(sec, lab): 
        return sec_scores.get(sec, {}).get(lab, np.nan)

    # sekční indexy
    def_i = sec_idx.get("Defenziva", np.nan)
    off_i = sec_idx.get("Ofenziva", np.nan)
    pas_i = sec_idx.get("Přihrávky", np.nan)
    one_i = sec_idx.get("1v1", np.nan)

    # ofenziva / tvorba
    G   = g("Ofenziva", "Góly /90")
    xG  = g("Ofenziva", "xG /90")
    xA  = g("Ofenziva", "xA /90")
    SA  = g("Ofenziva", "Shot assists /90")

    # passing / progres
    KP  = g("Přihrávky","Klíčové přihrávky /90")
    SP  = g("Přihrávky","Smart passes /90")
    PP  = g("Přihrávky","Progresivní přihrávek /90")
    CR  = g("Přihrávky","Úspěšnost centrů %")

    # 1v1
    DR  = g("1v1","Driblingy /90")
    DRp = g("1v1","Úspěšnost dribblingu %")
    ODp = g("1v1","Úspěšnost of. duelů %")
    PR  = g("1v1","Progresivní běhy /90")

    # defenziva
    DDp = g("Defenziva","Úspěšnost obr. duelů %")
    INT = g("Defenziva","Interceptions /90")
    ADp = g("Defenziva","Úspěšnost vzdušných %")
    FOU = g("Defenziva","Fauly /90")

    # běh (role-benchmark CZ)
    rs  = run_scores.get(RUN_KEY, {}) if run_scores else {}
    HIR = rs.get("High-intensity runs /90", np.nan)
    SPR = rs.get("Sprints /90", np.nan)
    HSD = rs.get("High-speed distance /90", np.nan)
    TOP = rs.get("Max speed (km/h)", np.nan)
    AVG = rs.get("Average speed (km/h)", np.nan)
    ACC = rs.get("Accelerations /90", np.nan)
    DEC = rs.get("Decelerations /90", np.nan)
    TDL = rs.get("Total distance /90", np.nan)

    p = []  # odstavce

    # 1) Úvod
    p.append(
        f"{player} ({team}, {pos}) působí v datech jako **{archetype}**"
        f"{(' v roli ' + role5) if role5 else ''}. V našich indexech (škála 0–100) "
        f"dosahuje herního souhrnu **{_lvl(overall)}**, běžeckého profilu **{_lvl(run_idx)}** "
        f"a kombinovaného výsledku **{_lvl(final_idx)}**. "
        + (
            f"V porovnání s interním prahem Slavie pro danou roli (**{_pct(thr_slavia)} %**) "
            f"je hráč **{'nad prahem' if (not pd.isna(final_idx) and not pd.isna(thr_slavia) and final_idx>=thr_slavia) else 'pod prahem'}**, "
            f"což je důležitá informace pro rozhodnutí o přestupu."
            if not pd.isna(thr_slavia) else
            "Pro přímé srovnání se slávistickým standardem chybí role-prahová hodnota; hodnotíme tedy čistě relativně k lize."
        )
    )

    # 2) Herní charakteristika – co je na očích
    best_sec = max(
        [("Ofenziva", off_i), ("Přihrávky", pas_i), ("1v1", one_i), ("Defenziva", def_i)],
        key=lambda kv: -999 if pd.isna(kv[1]) else kv[1]
    )[0]
    sec_notes = []
    if _is_high(off_i): sec_notes.append(f"ofenziva **{_lvl(off_i)}** (produkce, tlak na bránu)")
    if _is_high(pas_i): sec_notes.append(f"přihrávky **{_lvl(pas_i)}** (posouvání hry dopředu a tvorba)")
    if _is_high(one_i): sec_notes.append(f"1v1 **{_lvl(one_i)}** (individuální průniky a souboje)")
    if _is_high(def_i): sec_notes.append(f"defenziva **{_lvl(def_i)}** (zisky a obranné souboje)")
    if sec_notes:
        p.append(
            "Z hlediska sekcí působí nejsilněji **" + best_sec + "**; "
            "konkrétně platí, že " + ", ".join(sec_notes) + ". "
            "Pro laické čtení: čím blíže je hodnota ke 100 %, tím více hráč dominuje v ligovém kontextu."
        )

    # 3) On-ball přínos – co dělá s míčem
    ob = []
    if any(_is_high(v) for v in [KP, SP, PP]):
        ob.append(
            f"Ve výstavbě je patrná schopnost **progresivní přihrávky** "
            f"(PP {_lvl(PP)}; smart/SP {_lvl(SP)}; klíčové/KP {_lvl(KP)}), "
            "což zjednodušeně znamená, že často a kvalitně posouvá míč do nebezpečných zón "
            "a vytváří spoluhráčům lepší pozice k zakončení."
        )
    if _is_high(CR):
        ob.append(f"Z křídla dokáže doručit **kvalitní centry** (úspěšnost {_lvl(CR)}), což pomáhá proti zavřeným blokům.")
    if any(_is_high(v) for v in [G, xG]):
        ob.append(
            f"Směrem k brance představuje **gólovou hrozbu** (G {_lvl(G)}, xG {_lvl(xG)}); "
            "nevytváří jen objem střel, ale také se dostává do šancí dobré kvality."
        )
    if any(_is_high(v) for v in [DR, DRp, PR]):
        ob.append(
            f"V individuálním řešení situací působí **sebevědomě** (dribling {_lvl(DR)} / úspěšnost {_lvl(DRp)}; "
            f"progresivní nesení míče {_lvl(PR)}), "
            "což je užitečné pro odemykání obranných bloků jeden na jednoho."
        )
    if ob:
        p.append("S míčem: " + " ".join(ob))

    # 4) Off-ball & defenziva – jak pracuje bez míče
    df = []
    if any(_is_high(v) for v in [DDp, INT]):
        df.append(
            f"V obraně je silný v **souborech a zachycování** (úsp. obr. duelů {_lvl(DDp)}, intercepts {_lvl(INT)}), "
            "což z něj dělá hráče, který dokáže zastavit akce ještě před rozvinutím."
        )
    if _is_high(ADp):
        df.append(f"Ve **vzdušných soubojích** drží solidní úroveň ({_lvl(ADp)}), což pomáhá při standardkách i v obranném boxu.")
    if _is_high(FOU, th=60):
        df.append(
            f"Současně ale vykazuje vyšší **faulovost** ({_lvl(FOU)}); "
            "v prostředí evropských utkání může být potřebná práce na timingu a postavení těla."
        )
    if df:
        p.append("Bez míče: " + " ".join(df))

    # 5) Běžecký kontext – tempo, rychlost, intenzita
    rb = []
    if not pd.isna(run_idx):
        if _is_high(run_idx, th=60):
            rb.append(
                f"Běžecky působí **nadprůměrně** ({_lvl(run_idx)}), "
                "což je dobrý signál pro náročnější presink a rychlé přechody."
            )
        elif _pct(run_idx) <= 45:
            rb.append(
                f"Běžecký index **{_lvl(run_idx)}** naznačuje, že v utkáních s dlouhodobě vysokým tempem "
                "může výkon kolísat; vhodné je řízení vytížení a role."
            )
    if any(_is_high(v) for v in [HIR, SPR, HSD]):
        rb.append(
            f"Ukazatele **HI běhů a sprintů** (HIR {_lvl(HIR)}, sprinty {_lvl(SPR)}, HSD {_lvl(HSD)}) "
            "podporují ochotu a schopnost opakovaně zrychlit a vytvářet tlak na obranu."
        )
    if _is_high(TOP, th=70):
        rb.append(f"Maximální rychlost vychází **nad standardem** ({_lvl(TOP)}), což je cenné pro náběhy i obranné krytí prostoru.")
    if any(_is_high(v) for v in [ACC, DEC]):
        rb.append(f"Pozitivní je i **explozivita** (akcelerace {_lvl(ACC)}, decelerace {_lvl(DEC)}); pomáhá v náhlých změnách směru.")
    if rb:
        p.append("Běžecký profil: " + " ".join(rb))

    # 6) Rizika a rozvojové priority
    dev = []
    if role5 in ("RW","LW") and not _is_high(CR) and not pd.isna(CR):
        dev.append("U křídel stojí za pozornost **kvalita centrů** – technika a výběr momentu mohou posunout finální výstup.")
    if role5 in ("CM","RB","CB") and not _is_high(PP) and not pd.isna(PP):
        dev.append("Pro posun na vyšší úroveň bude důležitý **rychlejší první dotek a orientace těla**, aby rostl objem progresivních přihrávek.")
    if role5 in ("RW","LW","CF") and not _is_high(DRp) and not pd.isna(DRp):
        dev.append("V **1v1** by pomohla lepší změna rytmu a práce tělem; zvýší se úspěšnost průniků.")
    if role5 in ("CB","CF") and not _is_high(ADp) and not pd.isna(ADp):
        dev.append("Ve **vzduchu** doporučujeme zaměřit se na timing odrazu a práci lokty v pravidlech.")
    if dev:
        p.append("Rozvoj: " + " ".join(dev))

    # 7) Taktický fit – kde bude prospívat / kde ne
    fit = []
    unfit = []
    roleU = (role5 or "").upper()
    if roleU in ("RW","LW","RB") and _is_high(HIR):
        fit.append("rychlý, vertikální herní plán s **vysokým presinkem** a přepínáním po zisku")
    if roleU in ("CM","CB") and _is_high(PP):
        fit.append("tým s **pozicní výstavbou**, kde se cení progres míčem a rozdělování hry")
    if roleU == "CF" and ( _is_high(G) or _is_high(xG) ):
        fit.append("sestavy, které dostávají míč do **boxu** a hledají finále v prostoru mezi stopery")
    if not fit: 
        fit.append("kontextové využití dle match-planu; profil je vyvážený a přizpůsobitelný")

    if roleU in ("RW","LW") and (not _is_high(run_idx) and not _is_high(DRp)):
        unfit.append("dlouhé pásmo **poslední třetiny 1v1** bez kvalitní podpory spoluhráčů")
    if roleU == "CM" and (not _is_high(PP) and not _is_high(SP)):
        unfit.append("ústřední role **hlavního distributora** v pomalé poziční hře")
    if roleU == "CB" and (not _is_high(DDp) and not _is_high(ADp)):
        unfit.append("systém se **spoustou izolovaných 1v1** v hlubokém bloku proti silným soupeřům")

    p.append(
        "Vhodné herní prostředí: " + "; ".join(fit) + ". " +
        ("Hůře vhodné: " + "; ".join(unfit) + "." if unfit else "")
    )

    # 8) Závěr
    p.append(
        "Souhrnně řečeno: profil hráče je z dat čitelný a přenositelný. "
        "Pokud nastavíme roli v souladu s jeho silnými stránkami a současně ošetříme slabší oblasti "
        "(zejména ty výše zmíněné v rozvojových prioritách), dostaneme stabilní výkon i v zápasech s vyšší intenzitou. "
        "Z hlediska rozhodnutí o posílení kádru je klíčové porovnat, zda očekávaná role v systému Slavie využije jeho přednosti naplno."
    )

    return "\n\n".join(p)


def _explain_metric(label: str) -> str:
    base = label.replace(" (běh)", "")
    return _METRIC_EXPL.get(base, base.lower())

def _format_strength_item(label: str, val: float) -> str:
    return f"- {label}: {int(round(val))}% ({_band(val)}) – {_explain_metric(label)}"

def _pct(p):
    return None if pd.isna(p) else int(round(float(p)))


def _fmt_metric_list(pairs):
    """pairs = [(label, value), ...] -> 'Label 75%, Label 62%, ...' nebo 'n/a'"""
    vals = [f"{k} {int(round(v))}%" for k, v in pairs if not pd.isna(v)]
    return ", ".join(vals) if vals else "n/a"

def _collect_strengths_weaknesses(sec_scores, run_scores):
    strengths=[]; weaknesses=[]
    for _, lst, key in blocks:
        for _, lab in lst:
            v = sec_scores.get(key, {}).get(lab, np.nan)
            if not pd.isna(v):
                if v>=70: strengths.append((lab, v))
                if v<=40: weaknesses.append((lab, v))
    if run_scores and RUN_KEY in run_scores:
        for lab, v in run_scores[RUN_KEY].items():
            if not pd.isna(v):
                if v>=70: strengths.append((f"{lab} (běh)", v))
                if v<=40: weaknesses.append((f"{lab} (běh)", v))
    strengths.sort(key=lambda x: x[1], reverse=True)
    weaknesses.sort(key=lambda x: x[1])
    return strengths[:6], weaknesses[:6]

# stručné vysvětlení metrik pro report
_METRIC_EXPL = {
    "Góly /90":"gólová produkce",
    "xG /90":"kvalita šancí (xG)",
    "Asistence /90":"připravené góly",
    "xA /90":"tvorba šancí (xA)",
    "Shot assists /90":"přihrávky do střel",
    "Klíčové přihrávky /90":"klíčová finalita",
    "Smart passes /90":"kreativní průniky",
    "Progresivní přihrávek /90":"posouvání hry dopředu",
    "Do finální třetiny /90":"zásobování finální třetiny",
    "Úspěšnost centrů %":"kvalita centrů",
    "Second assists /90":"před-asistence",
    "Driblingy /90":"objem 1v1",
    "Úspěšnost dribblingu %":"efektivita 1v1",
    "Úspěšnost of. duelů %":"síla v ofenzivních duelech",
    "Progresivní běhy /90":"nesení míče vpřed",
    "Defenzivní duely /90":"objem def. soubojů",
    "Úspěšnost obr. duelů %":"úspěšnost def. soubojů",
    "Interceptions /90":"čtení hry / zachycení",
    "Sliding tackles /90":"slide zákroky",
    "Úspěšnost vzdušných %":"vzdušné souboje",
    "Fauly /90":"faulovost",
    # běh:
    "Total distance /90":"celková zátěž",
    "High-intensity runs /90":"počet HI běhů",
    "Sprints /90":"sprintový objem",
    "Max speed (km/h)":"maximální rychlost",
    "Average speed (km/h)":"průměrná rychlost",
    "Accelerations /90":"akcelerace",
    "Decelerations /90":"decelerace",
    "High-speed distance /90":"vzdál. ve vysoké rychlosti",
}

def _explain_metric(label: str) -> str:
    return _METRIC_EXPL.get(label, label.lower())

def _format_strength_item(label: str, val: float) -> str:
    return f"- {label}: {int(round(val))}% ({_band(val)}) – {_explain_metric(label)}"


def _infer_archetype(role5, sec_scores, run_scores):
    G   = sec_scores.get("Ofenziva", {}).get("Góly /90", np.nan)
    xG  = sec_scores.get("Ofenziva", {}).get("xG /90", np.nan)
    xA  = sec_scores.get("Ofenziva", {}).get("xA /90", np.nan)
    SA  = sec_scores.get("Ofenziva", {}).get("Shot assists /90", np.nan)
    KP  = sec_scores.get("Přihrávky", {}).get("Klíčové přihrávky /90", np.nan)
    SP  = sec_scores.get("Přihrávky", {}).get("Smart passes /90", np.nan)
    PP  = sec_scores.get("Přihrávky", {}).get("Progresivní přihrávek /90", np.nan)
    CR  = sec_scores.get("Přihrávky", {}).get("Úspěšnost centrů %", np.nan)
    DR  = sec_scores.get("1v1", {}).get("Driblingy /90", np.nan)
    DRp = sec_scores.get("1v1", {}).get("Úspěšnost dribblingu %", np.nan)
    ODp = sec_scores.get("1v1", {}).get("Úspěšnost of. duelů %", np.nan)
    DDp = sec_scores.get("Defenziva", {}).get("Úspěšnost obr. duelů %", np.nan)
    INT = sec_scores.get("Defenziva", {}).get("Interceptions /90", np.nan)
    ADp = sec_scores.get("Defenziva", {}).get("Úspěšnost vzdušných %", np.nan)
    PR  = sec_scores.get("1v1", {}).get("Progresivní běhy /90", np.nan)

    HIR = run_scores.get(RUN_KEY, {}).get("High-intensity runs /90", np.nan) if run_scores else np.nan
    SPR = run_scores.get(RUN_KEY, {}).get("Sprints /90", np.nan) if run_scores else np.nan
    HSD = run_scores.get(RUN_KEY, {}).get("High-speed distance /90", np.nan) if run_scores else np.nan
    TOP = run_scores.get(RUN_KEY, {}).get("Max speed (km/h)", np.nan) if run_scores else np.nan

    role = (role5 or "").upper()
    tags=[]

    if role=="CB":
        if max(DDp, INT) >= 70 and (ADp>=60 or not pd.isna(ADp)): tags.append("Stopper / duelistický stoper")
        if max(PP, SP) >= 65: tags.append("Rozehrávající stoper (ball-playing)")
        if max(HIR, SPR, HSD) >= 65: tags.append("Cover stoper (rychlostní krytí)")
        if not tags: tags.append("Univerzální stoper")

    elif role=="RB":
        if max(PP, CR, PR) >= 65: tags.append("Útočný (overlap/wing-back)")
        if max(DDp, INT, ADp) >= 65: tags.append("Defenzivní fullback")
        if max(HIR, SPR, TOP) >= 70: tags.append("Rychlostní profil – náběhy, pressing")
        if not tags: tags.append("Vyvážený krajní obránce")

    elif role=="CM":
        if max(PP, KP, SP) >= 70: tags.append("Playmaker / progresivní passer")
        if max(DDp, INT) >= 65: tags.append("Ball-winner / disruptor")
        if max(DR, PR, HIR) >= 65: tags.append("Box-to-box runner")
        if not tags: tags.append("Univerzální středový záložník")

    elif role=="RW":
        if max(DR, DRp, PR) >= 70: tags.append("Driblující křídelník (1v1)")
        if max(KP, SA, xA) >= 70: tags.append("Kreator / final 3rd passer")
        if max(G, xG) >= 70: tags.append("Inside forward / gólový křídelník")
        if not tags: tags.append("Vyvážené křídlo")

    elif role=="CF":
        if max(G, xG) >= 70: tags.append("Gólový hrot (poacher/finisher)")
        if max(KP, SA) >= 65: tags.append("Spojka / target-link")
        if max(HIR, SPR) >= 65: tags.append("Pressing / běžecký hrot")
        if not tags: tags.append("Komplexní útočník")

    else:
        if max(DR, DRp, PR) >= 70: tags.append("Dynamický 1v1 profíl")
        if max(PP, KP, SP) >= 70: tags.append("Progresivní tvůrce")
        if max(DDp, INT) >= 65: tags.append("Pracovitý bez míče / def. přínos")
        if not tags: tags.append("Neutrální profil")

    return ", ".join(tags)

def build_player_analysis_md(player, team, age, pos, role5, league_name,
                             sec_scores, sec_idx, overall, 
                             run_scores, run_idx, final_idx, thr_slavia):
    # TOP/LOW metriky
    top_game = _topk({lab:sec_scores[k].get(lab,np.nan) 
                      for _, lst, k in blocks for _, lab in lst}, k=5, reverse=True)
    low_game = _topk({lab:sec_scores[k].get(lab,np.nan) 
                      for _, lst, k in blocks for _, lab in lst}, k=5, reverse=False)

    strengths_all, weaknesses = _collect_strengths_weaknesses(sec_scores, run_scores)
    top5_strengths = strengths_all[:5]
    archetype = _infer_archetype(role5, sec_scores, run_scores)

    # vhodnost
    vhodnost = []
    if not pd.isna(final_idx):
        if not pd.isna(thr_slavia):
            vhodnost.append(f"**Slavia (role {role5 or '—'})**: "
                            f"{'ANO – nad prahem' if final_idx>=thr_slavia else 'NE – pod prahem'} "
                            f"(hráč {int(round(final_idx))}% vs práh {int(round(thr_slavia))}%).")
        else:
            vhodnost.append("**Slavia**: nelze vyhodnotit (chybí práh).")
        if final_idx>=55:
            vhodnost.append("**Fortuna:Liga**: silná vhodnost (55%+).")
        elif final_idx>=45:
            vhodnost.append("**Fortuna:Liga**: hraniční/kontextová (45–55%).")
        else:
            vhodnost.append("**Fortuna:Liga**: spíše nevhodný (<45%).")

    # --- Markdown hlavička (stručné fakty) ---
    md=[]
    md.append(f"### 🧠 Analýza typologie: {player} ({team}, {pos}, věk {age})")
    md.append(f"- **Role5**: `{role5 or '—'}`  •  **Archetyp**: **{archetype}**")
    md.append(f"- **Role-index (herní, vážený)**: {int(round(overall)) if not pd.isna(overall) else 'n/a'}%")
    md.append(f"- **Běžecký index**: {int(round(run_idx)) if not pd.isna(run_idx) else 'n/a'}%")
    md.append(f"- **Celkový index (herní + běžecký)**: {int(round(final_idx)) if not pd.isna(final_idx) else 'n/a'}%")
    if vhodnost:
        md.append("- " + "  \n- ".join(vhodnost))

    # TOP 5 silných stránek (kontextové popisy)
    if top5_strengths:
        md.append("\n**TOP 5 silných stránek:**")
        for lab, v in top5_strengths:
            md.append(_format_strength_item(lab, v))

    # slabiny
    if weaknesses:
        md.append("\n**Slabé stránky:**")
        for lab, v in weaknesses[:5]:
            md.append(f"- {lab}: {int(round(v))}% ({_band(v)}) – {_explain_metric(lab)}")

    # rychlý přehled top/low metrik
    md.append("\n**TOP metriky (herní):** " + _fmt_metric_list(top_game))
    md.append("**NEJSLABŠÍ metriky (herní):** " + _fmt_metric_list(low_game))

    # --- DLOUHÝ SKAUTSKÝ REPORT (multi-paragraph) ---
    long_txt = _build_long_scout_report(
        player=player, team=team, pos=pos, role5=role5,
        sec_scores=sec_scores, sec_idx=sec_idx, overall=overall,
        run_scores=run_scores, run_idx=run_idx, final_idx=final_idx,
        thr_slavia=thr_slavia, archetype=archetype
    )
    md.append("\n---\n**📝 Skautský report (dlouhý):**\n\n" + long_txt)

    return "\n".join(md)

def build_run_only_md(player, team, age, pos, role5, run_idx, run_scores):
    strengths, weaknesses = _collect_strengths_weaknesses({}, run_scores)
    md = []
    md.append(f"### 🧠 Běžecká analýza: {player} ({team}, {pos}, věk {age})")
    md.append(f"- **Role5**: `{role5 or '—'}`")
    md.append(f"- **Běžecký index**: {int(round(run_idx)) if not pd.isna(run_idx) else 'n/a'}%")
    if strengths:
        md.append("\n**Silné stránky (běh):**")
        for lab,v in strengths: md.append(f"- {lab}: {int(round(v))}% ({_band(v)})")
    if weaknesses:
        md.append("\n**Slabé stránky (běh):**")
        for lab,v in weaknesses: md.append(f"- {lab}: {int(round(v))}% ({_band(v)})")
    tips=[]
    if not pd.isna(run_idx) and run_idx>=55: tips.append("vhodný do vysoké intenzity a pressingu")
    elif not pd.isna(run_idx) and run_idx<45: tips.append("šetřit vysoké běžecké nároky, řídit vytížení")
    if not tips: tips.append("běžecký profil neutrální/kontextový")
    md.append("\n**Doporučení:**\n- " + "\n- ".join(tips))
    return "\n".join(md)

# ---------- UI: Tabs ----------
tab_card, tab_search = st.tabs(["Karta hráče (herní + běžecká)", "Vyhledávání hráčů"])

# === TAB 1 ===
with tab_card:
    c1,c2=st.columns(2)
    with c1:
        league_file=st.file_uploader("CZ liga – herní (xlsx)",["xlsx"],key="league_card")
        run_cz_file=st.file_uploader("CZ běžecká data – benchmark (xlsx)",["xlsx"],key="run_cz_card")
    with c2:
        players_file=st.file_uploader("Hráč/hráči – herní (xlsx)",["xlsx"],key="players_card")
        run_players_file=st.file_uploader("Hráč/hráči – běžecká (xlsx)",["xlsx"],key="run_players_card")

    have_game=bool(league_file and players_file); have_run=bool(run_cz_file and run_players_file)
    if not have_game and not have_run:
        st.info("➡ Nahraj buď (a) CZ herní + hráčský herní export, nebo (b) CZ běžecký benchmark + běžecký export."); st.stop()

    # JEN BĚŽECKÁ
    if (not have_game) and have_run:
        cz_run=auto_fix_run_df(pd.read_excel(run_cz_file), None)
        any_run=auto_fix_run_df(pd.read_excel(run_players_file), None)
        pcol=get_player_col(any_run) or "Player"
        sel=st.selectbox("Vyber hráče (běžecký export)", any_run[pcol].dropna().unique().tolist())
        row=any_run.loc[any_run[pcol]==sel].iloc[0]
        role5=_role5_or_none(row.get("Role5","") or role5_from_pos_text(row.get("Position","")))
        cz_base=cz_run[cz_run.get("Role5","").astype(str).str.upper()==role5] if role5 else pd.DataFrame()
        if cz_base is None or cz_base.empty:
            st.warning("Chybí CZ benchmark pro danou roli (běžecká). Běžecký index nebude vypočten.")
            r_scores,r_abs,run_idx={RUN_KEY:{}},{},np.nan
        else:
            plc=get_player_col(cz_base) or "Player"
            cz_agg=(cz_base.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_base.columns else cz_base).groupby("Player").mean(numeric_only=True)
            r_scores,r_abs,run_idx=run_scores_for_row(row,cz_agg)
        verdict="ANO – běžecky vhodný (55%+)" if (not pd.isna(run_idx) and run_idx>=55) else ("OK – šedá zóna (45–55%)" if (not pd.isna(run_idx) and run_idx>=45) else "NE – běžecky pod úrovní")
        fig=render_run_card(row.get("Player",""),row.get("Team",""),row.get("Position","—"),row.get("Age","n/a"),r_scores,r_abs,run_idx,verdict,role5=role5 or None)
        st.pyplot(fig); bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight"); st.download_button("📥 Stáhnout běžeckou kartu",data=bio.getvalue(),file_name=f"{sel}_run.png",mime="image/png"); plt.close(fig)

        # Analytický text pro běžeckou větev
        with st.expander("🧠 Analýza typologie hráče (auto report)"):
            report_md = build_run_only_md(
                player=row.get("Player",""), team=row.get("Team",""), age=row.get("Age","n/a"),
                pos=row.get("Position","—"), role5=role5, run_idx=run_idx, run_scores=r_scores
            )
            st.markdown(report_md)
            st.download_button("📥 Stáhnout analýzu (MD)", data=report_md.encode("utf-8"),
                               file_name=f"{sel}_analyza_run.md", mime="text/markdown")
        st.stop()

    # HERNÍ / KOMBINOVANÁ
    league=normalize_core_cols(pd.read_excel(league_file)); players=normalize_core_cols(pd.read_excel(players_file))
    run_cz_df=auto_fix_run_df(pd.read_excel(run_cz_file), league) if run_cz_file else None
    run_pl_df=auto_fix_run_df(pd.read_excel(run_players_file), players) if run_players_file else None
    w_run=w_run_pct/100.0
    slavia_thr=compute_slavia_role_thresholds(league, run_cz_df, w_run, how=th_agg)

    sel=st.selectbox("Vyber hráče (herní export)", players["Player"].dropna().unique().tolist())
    row=players.loc[players["Player"]==sel].iloc[0]
    player,team,pos,age,nat=row.get("Player",""),row.get("Team",""),row.get("Position",""),row.get("Age","n/a"),row.get("Nationality","")
    pg=pos_group(pos); rgx=POS_REGEX[pg]; cz_pos=league[league["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
    agg=cz_pos.groupby("Player").mean(numeric_only=True)

    scores,sec_idx,overall,run_scores,run_abs,run_idx,role5=compute_overall_and_run(row, agg, run_cz_df, run_pl_df, team, age, nat, pos)
    final_idx=final_from_overall_and_run(overall, run_idx, w_run)
    thr=slavia_thr.get(role5, np.nan)
    verdict="ANO – potenciální posila do Slavie" if (not pd.isna(final_idx) and not pd.isna(thr) and final_idx>=thr) else "NE – nedosahuje úrovně Slavie (role)"

    fig=render_card_visual(player,team,pos,age,scores,sec_idx,overall,verdict,run_scores,run_abs,run_idx,final_index=final_idx, role5=role5)
    st.pyplot(fig); bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight"); st.download_button("📥 Stáhnout kartu (PNG)",data=bio.getvalue(),file_name=f"{player}.png",mime="image/png"); plt.close(fig)

    # Analytický text – kombinovaná větev
    with st.expander("🧠 Analýza typologie hráče (auto report)"):
        league_name_disp = "CZ liga"
        thr_role = slavia_thr.get(role5, np.nan) if 'slavia_thr' in locals() else np.nan
        report_md = build_player_analysis_md(
            player=player, team=team, age=age, pos=pos, role5=role5,
            league_name=league_name_disp,
            sec_scores=scores, sec_idx=sec_idx, overall=overall,
            run_scores=run_scores, run_idx=run_idx,
            final_idx=final_idx, thr_slavia=thr_role
        )
        st.markdown(report_md)
        st.download_button("📥 Stáhnout analýzu (MD)", data=report_md.encode("utf-8"),
                           file_name=f"{player}_analyza.md", mime="text/markdown")

# === TAB 2 ===
with tab_search:
    st.subheader("Vyhledávání kandidátů (benchmark = CZ liga, prahy = Slavia)")
    cA,cB=st.columns(2)
    with cA:
        cz_file=st.file_uploader("CZ liga – herní (xlsx)",["xlsx"],key="cz_search")
        run_cz_file=st.file_uploader("CZ běžecká (xlsx) [volitelné]",["xlsx"],key="cz_run_search")
    with cB:
        fr_file=st.file_uploader("Cizí liga – herní (xlsx)",["xlsx"],key="fr_search")
        run_fr_file=st.file_uploader("Cizí liga – běžecká (xlsx) [volitelné]",["xlsx"],key="fr_run_search")

    if cz_file: st.session_state["cz_bytes"]=cz_file.getvalue()
    if fr_file: st.session_state["fr_bytes"]=fr_file.getvalue()
    if run_cz_file: st.session_state["cz_run_bytes"]=run_cz_file.getvalue()
    if run_fr_file: st.session_state["fr_run_bytes"]=run_fr_file.getvalue()

    pos_opts=list(POS_REGEX.keys())
    pos_sel=st.multiselect("Pozice (herní filtr pro benchmark CZ)",pos_opts,default=pos_opts)
    c1,c2,c3=st.columns(3)
    with c1: league_name=st.text_input("Název ligy",value="Cizí liga")
    with c2: min_minutes=st.number_input("Min. minut (pokud ve zdroji)",0,step=100)
    with c3: min_games=st.number_input("Min. zápasů (pokud ve zdroji)",0,step=1)
    run_btn=st.button("Spustit vyhledávání")

    if run_btn:
        if "cz_bytes" not in st.session_state or "fr_bytes" not in st.session_state:
            st.error("Nahraj CZ herní + cizí liga herní."); st.stop()
        cz_df=normalize_core_cols(load_xlsx(st.session_state["cz_bytes"]))
        fr_df=normalize_core_cols(load_xlsx(st.session_state["fr_bytes"]))
        if "Position" not in cz_df.columns or "Position" not in fr_df.columns:
            st.error("V jednom ze souborů chybí sloupec s pozicí."); st.stop()
        cz_run_df=auto_fix_run_df(load_xlsx(st.session_state.get("cz_run_bytes")),cz_df) if "cz_run_bytes" in st.session_state else None
        fr_run_df=auto_fix_run_df(load_xlsx(st.session_state.get("fr_run_bytes")),fr_df) if "fr_run_bytes" in st.session_state else None
        w_run=w_run_pct/100.0
        slavia_thr=compute_slavia_role_thresholds(cz_df, cz_run_df, w_run, how=th_agg)
        if not slavia_thr: st.warning("Nepodařilo se spočítat prahy Slavie (zkontroluj, že v CZ herních datech jsou hráči Slavie).")

        def search_candidates():
            mask=pd.Series(False,index=fr_df.index)
            for p in pos_sel: mask|=fr_df["Position"].astype(str).str.contains(POS_REGEX[p],na=False,regex=True)
            base=fr_df.loc[mask].copy()
            def pick(df,names): return next((n for n in names if n in df.columns),None)
            mc=pick(base,["Minutes","Minutes played","Min"]); gc=pick(base,["Games","Matches"])
            if min_minutes and mc: base=base[pd.to_numeric(base[mc],errors="coerce").fillna(0)>=min_minutes]
            if min_games and gc: base=base[pd.to_numeric(base[gc],errors="coerce").fillna(0)>=min_games]
            rows,cards=[],[]
            for _,r in base.iterrows():
                pos_txt=r.get("Position",""); pg=pos_group(pos_txt); rgx=POS_REGEX[pg]
                cz_pos=cz_df[cz_df["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
                if cz_pos.empty: continue
                cz_agg=cz_pos.groupby("Player").mean(numeric_only=True)
                scores,sec_idx,overall, r_scores, r_abs, run_idx, role5 = compute_overall_and_run(r, cz_agg, cz_run_df, fr_run_df, r.get("Team",""), r.get("Age",None), r.get("Nationality",""), pos_txt)
                final_idx=final_from_overall_and_run(overall, run_idx, w_run); thr=slavia_thr.get(role5, np.nan)
                if not pd.isna(final_idx) and not pd.isna(thr) and final_idx>=thr:
                    verdict="ANO – potenciální posila do Slavie"
                    player=r.get("Player",""); team=r.get("Team",""); age=r.get("Age","n/a")
                    rows.append({"Hráč":player,"Věk":age,"Klub":team,"Pozice":pos_txt,"Liga":league_name,"Role5":role5,
                                 "Index Def":sec_idx.get("Defenziva",np.nan),"Index Off":sec_idx.get("Ofenziva",np.nan),
                                 "Index Pass":sec_idx.get("Přihrávky",np.nan),"Index 1v1":sec_idx.get("1v1",np.nan),
                                 "Role-index (vážený)":overall,"Run index":run_idx,"Final index":final_idx,
                                 "Prahová hodnota Slavia (role)":thr,"Verdikt":verdict})
                    fig=render_card_visual(player,team,pos_txt,age,scores,sec_idx,overall,verdict,r_scores,r_abs,run_idx,final_index=final_idx, role5=role5)
                    bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight"); plt.close(fig)
                    cards.append((str(player),bio.getvalue()))
            return pd.DataFrame(rows),cards

        res_df,cards=search_candidates()
        st.session_state.update(search_results=res_df,search_cards=cards,fr_df=fr_df,cz_df=cz_df,fr_run_df=fr_run_df,cz_run_df=cz_run_df,slavia_thr=slavia_thr,w_run=w_run,league_name=league_name)

    res_df=st.session_state.get("search_results")
    if res_df is None or res_df.empty:
        st.info("Zatím žádné výsledky – nahraj soubory a klikni na Spustit vyhledávání.")
    else:
        st.success(f"Nalezeno kandidátů (Verdikt = ANO): {len(res_df)}")
        st.dataframe(res_df.sort_values(["Role5","Final index"],ascending=[True,False]), use_container_width=True)
        st.download_button("📥 Stáhnout CSV s kandidáty (ANO)", data=res_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"kandidati_{st.session_state.get('league_name','liga')}_ANO.csv", mime="text/csv")
        zbuf=BytesIO()
        with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
            for name,png in (st.session_state.get("search_cards") or []):
                safe=str(name).replace("/","").replace("\\",""); zf.writestr(f"{safe}.png", png)
        st.download_button("🗂 Stáhnout všechny karty (ZIP)", data=zbuf.getvalue(),
                           file_name=f"karty_{st.session_state.get('league_name','liga')}_ANO.zip", mime="application/zip")
        with st.expander("🖼 Online karty (všichni s verdiktem ANO)"):
            for name,png in (st.session_state.get("search_cards") or []): st.image(png, caption=name, use_column_width=True)
