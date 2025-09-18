# app.py — Slavia datacards (herní + běžecká, smart matching, role-aware running; umí i "jen běžeckou")
import re, unicodedata, zipfile
from io import BytesIO
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

st.set_page_config(page_title="Karty – Slavia", layout="wide")
st.title("⚽ Generátor datových karet (herní + běžecká)")

# ---------- Utils ----------
@st.cache_data
def load_xlsx(b: bytes) -> pd.DataFrame: 
    return pd.read_excel(BytesIO(b))

def color_for(v):
    if pd.isna(v): return "lightgrey"
    return "#FF4C4C" if v<=25 else "#FF8C00" if v<=50 else "#FFD700" if v<=75 else "#228B22"

def _best_col(df, names): 
    return next((c for c in names if c in df.columns), None)

def _normtxt(s):
    s=unicodedata.normalize("NFKD", str(s))
    return re.sub(r"\s+"," ","".join(c for c in s if not unicodedata.combining(c))).strip().lower()

def get_player_col(df): return _best_col(df,["Player","Name","player","name","Short Name"])
def get_team_col(df):   return _best_col(df,["Team","Club","team","club"])
def get_pos_col(df):    return _best_col(df,["Position","Pos","position","Role","Primary position"])
def get_age_col(df):    return _best_col(df,["Age","age","AGE"])
def get_nat_col(df):    return _best_col(df,["Nationality","Nationality 1","Nation","Country","Citizenship","Nat"])

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

def _split_name(s):
    t=_normtxt(s).replace("."," "); ps=[x for x in re.split(r"\s+",t) if x]
    if not ps: return "",""
    sur=ps[-1]; first=next((x for x in ps if x!=sur), "")
    return (first[0] if first else (ps[0][0] if ps else "")), sur

def _norm_team(s):
    t=_normtxt(s); t=re.sub(r"\b(fk|fc|sc|ac|cf|afc|sv|us|cd|ud|bk|sk|ks|ucl|ii|b)\b"," ",t)
    return re.sub(r"\s+"," ",t).strip()

def _norm_nat(s): return _normtxt(s)

def match_by_name(df, name, team_hint=None, age_hint=None, nat_hint=None):
    """Match jméno + tým + věk + národnost (scoring). Vrací 0–1 řádek."""
    if df is None or df.empty or not name: return pd.DataFrame()
    pcol=get_player_col(df) or "Player"
    if pcol not in df.columns: return pd.DataFrame()
    if "_kname" not in df.columns:
        df["_kname"]=df[pcol].astype(str).map(_normtxt)
        fi,sn=zip(*df[pcol].astype(str).map(_split_name))
        df["_kfirst"],df["_ksurname"]=list(fi),list(sn)
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

    si=df.loc[(df["_ksurname"]==sn_key)&(df["_kfirst"]==fi_key)]
    if len(si)==1: return si
    if len(si)>1 and team_key:
        pick=si.loc[si["_kteam"]==team_key]
        if len(pick)==1: return pick

    if team_key:
        st_df=df.loc[(df["_ksurname"]==sn_key)&(df["_kteam"]==team_key)]
        if len(st_df)==1: return st_df

    pool=df.loc[(df["_ksurname"]==sn_key) | ((team_key!="") & (df["_kteam"]==team_key))].copy()
    if pool.empty: return pd.DataFrame()

    def score(r):
        s=0
        if r["_ksurname"]==sn_key: s+=4
        if fi_key and r["_kfirst"]==fi_key: s+=4
        if team_key:
            if r["_kteam"]==team_key: s+=4
            elif team_key in r["_kteam"] or r["_kteam"] in team_key: s+=2
        if age_key is not None and not pd.isna(r["_kage"]):
            d=abs(int(r["_kage"])-age_key); s+=3 if d==0 else 2 if d==1 else 1 if d==2 else 0
        if nat_key and r["_knat"]==nat_key: s+=2
        return s
    pool["_score"]=pool.apply(score,axis=1)
    best=pool.sort_values(["_score","_kage"],ascending=[False,True]).head(1)
    if not best.empty and best["_score"].iloc[0]>0: return best
    sn=df.loc[df["_ksurname"]==sn_key]
    return sn if len(sn)==1 else pd.DataFrame()

# ---------- Herní bloky & scoring ----------
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

def section_scores(row,agg,metric_w=None):
    sec_scores,sec_idx={},{}
    for _,lst,key in blocks:
        vals={lab:norm_metric(agg,eng,get_val_alias(row,eng)) for eng,lab in lst}
        sec_scores[key]=vals
        if metric_w and metric_w.get(key):
            w=metric_w[key]; wsum=sum(w.values()) or 1
            sec_idx[key]=float(sum(v*w.get(l,0) for l,v in vals.items() if not pd.isna(v))/wsum)
        else:
            arr=[v for v in vals.values() if not pd.isna(v)]
            sec_idx[key]=float(np.mean(arr)) if arr else np.nan
    return sec_scores,sec_idx

def role_index(sec_idx,weights):
    acc=tot=0.0
    for k in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v=sec_idx.get(k,np.nan)
        if not pd.isna(v):
            w=weights.get(k,0)/100.0
            acc+=v*w; tot+=w
    return float(acc/tot) if tot>0 else np.nan

# ---------- Pozice: herní benchmark (původní) ----------
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

# ---------- Pozice: kanonické role pro běžecká data (CB/RB/CM/RW/CF) ----------
# Wyscout -> Role5
WYS_TO_ROLE = {
    # Střední obránci
    "RCB":"CB","LCB":"CB","RCB3":"CB","LCB3":"CB","CB":"CB",
    # Krajní obránci
    "RB":"RB","RB5":"RB","LB":"RB","LB5":"RB","RWB":"RB","LWB":"RB",
    # Střední záložníci
    "DMF":"CM","RDMF":"CM","LDMF":"CM","RCMF":"CM","LCMF":"CM","RCMF3":"CM","LCMF3":"CM","AMF":"CM",
    # Křídla
    "RAMF":"RW","LAMF":"RW","RW":"RW","LW":"RW","AMFL":"RW","AMFR":"RW","LWF":"RW","RWF":"RW",
    # Útočníci
    "CF":"CF",
}

def _primary_wyscout_tag(pos_text:str) -> str:
    """Vezme první tag z 'RWB,RB,RB5' -> 'RWB' (upper, strip)."""
    if not pos_text: return ""
    first = str(pos_text).split(",")[0].strip().upper()
    # někdy může být 'Right Back' apod., zkus vyzobnout z textu známé kódy
    if first not in WYS_TO_ROLE:
        for k in WYS_TO_ROLE:
            if k in first:
                return k
    return first

def role5_from_pos_text(pos_text:str) -> str:
    """Kanonická role (CB/RB/CM/RW/CF) z Wyscout tagu (primárního)."""
    tag=_primary_wyscout_tag(pos_text)
    return WYS_TO_ROLE.get(tag,"")

# ---------- Running ----------
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

def _attach_role5_from_game(run_df, game_df):
    """Z herních dat vezmi primární pozici, přemapuj na Role5 a přidej do run_df (podle hráče)."""
    if run_df is None or run_df.empty: return run_df
    if game_df is None or game_df.empty: 
        # zkus aspoň odvodit z vlastního run_df["Position"]
        if "Position" in run_df.columns and "Role5" not in run_df.columns:
            run_df["Role5"]=run_df["Position"].map(role5_from_pos_text)
        return run_df
    g=normalize_core_cols(game_df.copy())
    if "Player" not in g.columns or "Position" not in g.columns: 
        if "Position" in run_df.columns and "Role5" not in run_df.columns:
            run_df["Role5"]=run_df["Position"].map(role5_from_pos_text)
        return run_df
    # vyrob mapu Player -> primární tag -> Role5
    tmp=g[["Player","Position"]].dropna().copy()
    tmp["Role5"]=tmp["Position"].astype(str).map(role5_from_pos_text)
    tmp=tmp.dropna(subset=["Role5"]).groupby("Player",as_index=False).agg({"Role5":"first"})
    run_df["_k"]=run_df["Player"].astype(str).map(_normtxt)
    tmp["_k"]=tmp["Player"].astype(str).map(_normtxt)
    out=run_df.merge(tmp[["_k","Role5"]],on="_k",how="left")
    out=out.drop(columns=["_k"])
    # fallback z vlastního Position, pokud pořád chybí
    if "Role5" in out.columns:
        mask=out["Role5"].isna() & out.get("Position",pd.Series(index=out.index)).astype(str).ne("")
        out.loc[mask,"Role5"]=out.loc[mask,"Position"].map(role5_from_pos_text)
    else:
        if "Position" in out.columns:
            out["Role5"]=out["Position"].map(role5_from_pos_text)
    return out

def auto_fix_run_df(run_df, game_df):
    """Normalizace běžečných dat + přidání Role5 (z herních, případně z vlastního Position)."""
    if run_df is None or run_df.empty: return run_df
    id_map={}
    if "Player" not in run_df.columns:   c=_best_col(run_df,["Name","player","name","Short Name"]);  id_map.update({c:"Player"} if c else {})
    if "Team"   not in run_df.columns:   c=_best_col(run_df,["Club","team","Team"]);                  id_map.update({c:"Team"} if c else {})
    if "Position" not in run_df.columns: c=_best_col(run_df,["Pos","Role","Primary position","position"]); id_map.update({c:"Position"} if c else {})
    if id_map: run_df=run_df.rename(columns=id_map)
    run_df=ensure_run_wide(run_df); run_df=_post_run(run_df)
    # přidej Role5
    run_df=_attach_role5_from_game(run_df, game_df)
    # pokud chybí Position a máme herní data, zkus doplnit Position (ne nutné pro Role5, ale hodí se)
    if "Position" not in run_df.columns and game_df is not None and not game_df.empty:
        g=normalize_core_cols(game_df.copy())
        if {"Player","Position"}.issubset(g.columns):
            g=g[["Player","Position"]].dropna().groupby("Player",as_index=False).agg({"Position":"first"})
            run_df["_k"]=run_df["Player"].map(_normtxt); g["_k"]=g["Player"].map(_normtxt)
            run_df=run_df.merge(g[["_k","Position"]],on="_k",how="left").drop(columns=["_k"])
    return run_df

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
def render_card_visual(player,team,pos,age,scores,sec_index,overall_base,verdict,
                       run_scores=None,run_abs=None,run_index=np.nan,final_index=None, role5=None):
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
    if run_scores and RUN_KEY in run_scores and len(run_scores[RUN_KEY])>0:
        sub=f"Běžecká data (vs. CZ benchmark{f' – role {role5}' if role5 else ''})"
        ax.text(0.02,y0,sub,fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
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
st.sidebar.header("⚙️ Váhy sekcí")
base_w={"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
sec_w={k:st.sidebar.slider(k,0,100,base_w[k],1) for k in base_w}
tot=sum(sec_w.values()) or 1
for k in sec_w: sec_w[k]=100.0*sec_w[k]/tot
w_run_pct=st.sidebar.slider("Váha běžeckého indexu v celkovém hodnocení",0,50,20,5)
metric_w={}
with st.sidebar.expander("Váhy metrik v sekcích (volitelné)",False):
    for _,lst,key in blocks:
        st.markdown(f"**{key}**")
        tmp={lab:st.slider(f"– {lab}",0,100,10,1,key=f"{key}_{lab}") for _,lab in lst}
        s=sum(tmp.values()) or 1; metric_w[key]={lab:w/s for lab,w in tmp.items()} if s else None

# ---------- Tabs ----------
tab_card, tab_search = st.tabs(["Karta hráče (herní + běžecká)", "Vyhledávání hráčů"])

# === TAB 1: umí herní / kombinovanou / jen běžeckou ===
with tab_card:
    c1,c2=st.columns(2)
    with c1:
        league_file=st.file_uploader("CZ liga – herní (xlsx)",["xlsx"],key="league_card")
        run_cz_file=st.file_uploader("CZ běžecká data – benchmark (xlsx)",["xlsx"],key="run_cz_card")
    with c2:
        players_file=st.file_uploader("Hráč/hráči – herní (xlsx)",["xlsx"],key="players_card")
        run_players_file=st.file_uploader("Hráč/hráči – běžecká (xlsx)",["xlsx"],key="run_players_card")

    # ---- REŽIMY ----
    have_game = bool(league_file and players_file)
    have_run  = bool(run_cz_file and run_players_file)

    if not have_game and not have_run:
        st.info("➡️ Nahraj buď (a) CZ herní + hráčský herní export, nebo (b) CZ běžecký benchmark + běžecký export."); st.stop()

    # ---------- JEN BĚŽECKÁ ----------
    if (not have_game) and have_run:
        cz_run=auto_fix_run_df(pd.read_excel(run_cz_file), None)
        any_run=auto_fix_run_df(pd.read_excel(run_players_file), None)
        pcol=get_player_col(any_run) or "Player"
        sel=st.selectbox("Vyber hráče (běžecký export)", any_run[pcol].dropna().unique().tolist())
        row=any_run.loc[any_run[pcol]==sel].iloc[0]
        # role pro běh z vlastního Position (když nemáme herní data)
        role5=row.get("Role5","") or role5_from_pos_text(row.get("Position",""))
        # benchmark CZ: filtrovat Role5, když možno
        cz_base = cz_run if not role5 else cz_run[cz_run.get("Role5","").astype(str)==role5]
        if cz_base is None or cz_base.empty: cz_base = cz_run
        plc=get_player_col(cz_base) or "Player"
        cz_agg=(cz_base.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_base.columns else cz_base).groupby("Player").mean(numeric_only=True)
        r_scores,r_abs,run_idx=run_scores_for_row(row,cz_agg)
        verdict="ANO – běžecky vhodný (55%+)" if (not pd.isna(run_idx) and run_idx>=55) else ("OK – šedá zóna (45–55%)" if (not pd.isna(run_idx) and run_idx>=45) else "NE – běžecky pod úrovní")
        fig=render_run_card(row.get("Player",""),row.get("Team",""),row.get("Position","—"),row.get("Age","n/a"),
                            r_scores,r_abs,run_idx,verdict,role5=role5 or None)
        st.pyplot(fig)
        bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight")
        st.download_button("📥 Stáhnout běžeckou kartu",data=bio.getvalue(),file_name=f"{sel}_run.png",mime="image/png")
        plt.close(fig)
        st.stop()

    # ---------- HERNÍ / KOMBINOVANÁ ----------
    league=normalize_core_cols(pd.read_excel(league_file))
    players=normalize_core_cols(pd.read_excel(players_file))
    run_cz_df=auto_fix_run_df(pd.read_excel(run_cz_file), league) if run_cz_file else None
    run_pl_df=auto_fix_run_df(pd.read_excel(run_players_file), players) if run_players_file else None

    sel=st.selectbox("Vyber hráče (herní export)", players["Player"].dropna().unique().tolist())
    row=players.loc[players["Player"]==sel].iloc[0]
    player,team,pos,age,nat=row.get("Player",""),row.get("Team",""),row.get("Position",""),row.get("Age","n/a"),row.get("Nationality","")

    # herní benchmark (původní logika po „pozicích“)
    pg=pos_group(pos); rgx=POS_REGEX[pg]
    cz_pos=league[league["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
    agg=cz_pos.groupby("Player").mean(numeric_only=True)
    scores,sec_idx=section_scores(row,agg,metric_w); overall=role_index(sec_idx,sec_w)

    # ---- běžecká část: role-aware ----
    run_scores=run_abs=None; run_idx=np.nan; role5=None
    if run_cz_df is not None and run_pl_df is not None:
        # 1) primární role z HERNÍ pozice vybraného hráče
        role5 = role5_from_pos_text(pos) or None
        # 2) najdi kandidáta v běžečných datech
        cand=match_by_name(run_pl_df, player, team_hint=team, age_hint=age, nat_hint=nat)
        # 3) benchmark = CZ běžečná data filtrovaná Role5
        cz_base=run_cz_df
        if role5 and "Role5" in run_cz_df.columns:
            cz_fil=run_cz_df[run_cz_df["Role5"]==role5]
            if not cz_fil.empty: cz_base=cz_fil
        plc=get_player_col(cz_base) or "Player"
        cz_agg=(cz_base.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_base.columns else cz_base).groupby("Player").mean(numeric_only=True)
        if not cand.empty and not cz_agg.empty:
            r_run=cand.iloc[0]
            # připiš role5 i na kandidáta (když chybí)
            if "Role5" not in r_run.index or pd.isna(r_run["Role5"]):
                r_run=r_run.copy(); r_run["Role5"]=role5
            run_scores,run_abs,run_idx=run_scores_for_row(r_run,cz_agg)

    w_run=w_run_pct/100.0
    final_idx=(1.0-w_run)*overall + w_run*run_idx if not pd.isna(run_idx) else None
    peer=np.nan  # peers můžeš doplnit dle potřeby
    base=final_idx if (final_idx is not None) else overall
    verdict="ANO – potenciální posila do Slavie" if (not np.isnan(base) and (np.isnan(peer) or base>=peer)) else "NE – nedosahuje úrovně"

    fig=render_card_visual(player,team,pos,age,scores,sec_idx,overall,verdict,
                           run_scores,run_abs,run_idx,final_index=final_idx, role5=role5)
    st.pyplot(fig)
    bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight")
    st.download_button("📥 Stáhnout kartu (PNG)",data=bio.getvalue(),file_name=f"{player}.png",mime="image/png")
    plt.close(fig)

# === TAB 2: vyhledávání (beze změny logiky herních metrik; běžecký index teď taky role-aware) ===
with tab_search:
    st.subheader("Vyhledávání kandidátů (benchmark = CZ liga)")
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
    pos_sel=st.multiselect("Pozice",pos_opts,default=pos_opts)
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
        cz_run_df=auto_fix_run_df(load_xlsx(st.session_state["cz_run_bytes"]),cz_df) if "cz_run_bytes" in st.session_state else None
        fr_run_df=auto_fix_run_df(load_xlsx(st.session_state["fr_run_bytes"]),fr_df) if "fr_run_bytes" in st.session_state else None
        w_run=w_run_pct/100.0

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
                pg=pos_group(r.get("Position","")); rgx=POS_REGEX[pg]
                cz_pos=cz_df[cz_df["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
                if cz_pos.empty: continue
                cz_agg=cz_pos.groupby("Player").mean(numeric_only=True)
                scores,sec_idx=section_scores(r,cz_agg,metric_w); overall=role_index(sec_idx,sec_w)

                run_idx=np.nan; r_scores=None; r_abs=None; role5=None
                if cz_run_df is not None and fr_run_df is not None:
                    # role z HERNÍ pozice kandidáta
                    role5 = role5_from_pos_text(r.get("Position","")) or None
                    cand=match_by_name(fr_run_df, r.get("Player",""), team_hint=r.get("Team",""),
                                       age_hint=r.get("Age",None), nat_hint=r.get("Nationality",""))
                    poscol=get_pos_col(cz_run_df)
                    cz_base=cz_run_df
                    if role5 and "Role5" in cz_run_df.columns:
                        cz_fil=cz_run_df[cz_run_df["Role5"]==role5]
                        if not cz_fil.empty: cz_base=cz_fil
                    plc=get_player_col(cz_base) or "Player"
                    cz_run_agg=(cz_base.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_base.columns else cz_base).groupby("Player").mean(numeric_only=True)
                    if not cand.empty and not cz_run_agg.empty:
                        rr=cand.iloc[0]
                        if "Role5" not in rr.index or pd.isna(rr["Role5"]):
                            rr=rr.copy(); rr["Role5"]=role5
                        r_scores,r_abs,run_idx=run_scores_for_row(rr,cz_run_agg)

                final_idx=(1.0-w_run)*overall + w_run*run_idx if (not pd.isna(run_idx) and w_run>0) else overall
                verdict="ANO – potenciální posila do Slavie" if (not np.isnan(final_idx)) else "NE"
                if verdict.startswith("ANO"):
                    player=r.get("Player",""); team=r.get("Team",""); pos=r.get("Position",""); age=r.get("Age","n/a")
                    rows.append({"Hráč":player,"Věk":age,"Klub":team,"Pozice":pos,"Liga":league_name,
                                 "Index Def":sec_idx.get("Defenziva",np.nan),"Index Off":sec_idx.get("Ofenziva",np.nan),
                                 "Index Pass":sec_idx.get("Přihrávky",np.nan),"Index 1v1":sec_idx.get("1v1",np.nan),
                                 "Role-index (vážený)":overall,"Run index":run_idx,"Final index":final_idx,"Verdikt":verdict,"Role5":role5})
                    fig=render_card_visual(player,team,pos,age,scores,sec_idx,overall,verdict,
                                           r_scores,r_abs,run_idx,final_index=(final_idx if not pd.isna(run_idx) else None),
                                           role5=role5)
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
        st.download_button("📥 Stáhnout CSV s kandidáty", data=res_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"kandidati_{st.session_state.get('search_league','liga')}.csv", mime="text/csv")
        zbuf=BytesIO()
        with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
            for name,png in (st.session_state.get("search_cards") or []):
                safe=str(name).replace("/","_").replace("\\","_"); zf.writestr(f"{safe}.png", png)
        st.download_button("🗂️ Stáhnout všechny karty (ZIP)", data=zbuf.getvalue(),
                           file_name=f"karty_{st.session_state.get('search_league','liga')}.zip", mime="application/zip")

