# app.py — Slavia datacards (shortened, same features)
import re, unicodedata, zipfile
from io import BytesIO
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

# ============ UI ============
st.set_page_config(page_title="Karty – Slavia", layout="wide")
st.title("⚽ Generátor datových karet (herní + běžecká)")

# ============ Helpers ============
_norm = lambda s: re.sub(r"\s+"," ","".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))).strip().lower()
def _best(df, names): return next((c for c in names if c in df.columns), None)
def col_player(df): return _best(df,["Player","Name","player","name","Short Name"]) or "Player"
def col_team(df):   return _best(df,["Team","Club","team","club"])
def col_pos(df):    return _best(df,["Position","Pos","position","Role","Primary position"])
def col_age(df):    return _best(df,["Age","age","AGE"])
def col_nat(df):    return _best(df,["Nationality","Nationality 1","Nation","Country","Citizenship","Nat"])

def normalize(df):
    if df is None or df.empty: return df
    ren={}
    for src,dst in [(col_player(df),"Player"),(col_team(df),"Team"),(col_pos(df),"Position")]:
        if src and src!=dst: ren[src]=dst
    return df.rename(columns=ren) if ren else df

def _split_name(s):
    t=_norm(s).replace("."," "); ps=[x for x in re.split(r"\s+",t) if x]
    if not ps: return "",""
    sur=ps[-1]; first=next((x for x in ps if x!=sur), "")
    return (first[0] if first else (ps[0][0] if ps else "")), sur

def _norm_team(s): return re.sub(r"\s+"," ",re.sub(r"\b(fk|fc|sc|ac|cf|afc|sv|us|cd|ud|bk|sk|ks|ucl|ii|b)\b"," ",_norm(s))).strip()
def is_slavia(t): t=_norm_team(t or ""); return "slavia" in t and ("praha" in t or "prague" in t)

def color_for(v): 
    if pd.isna(v): return "lightgrey"
    return "#FF4C4C" if v<=25 else "#FF8C00" if v<=50 else "#FFD700" if v<=75 else "#228B22"

# caching
@st.cache_data
def load_xlsx(b: bytes) -> pd.DataFrame: return pd.read_excel(BytesIO(b))

# ============ Pos/Role mapping ============
# 5 kanonických rolí pro běh
WYS_TO_ROLE = {
    # CB
    "RCB":"CB","LCB":"CB","RCB3":"CB","LCB3":"CB","CB":"CB",
    # RB/WB/LB
    "RB":"RB","RB5":"RB","LB":"RB","LB5":"RB","RWB":"RB","LWB":"RB","WB":"RB",
    # CM (DM/CM/AM)
    "DMF":"CM","RDMF":"CM","LDMF":"CM","RCMF":"CM","LCMF":"CM","RCMF3":"CM","LCMF3":"CM","AMF":"CM","DM":"CM","CM":"CM","AM":"CM",
    # Wings
    "RAMF":"RW","LAMF":"RW","RW":"RW","LW":"RW","AMFL":"RW","AMFR":"RW","LWF":"RW","RWF":"RW","W":"RW","WINGER":"RW",
    # Strikers
    "CF":"CF","ST":"CF","FW":"CF","FORWARD":"CF","STRIKER":"CF",
}
ROLE_PATTERNS = [
    ("CB", r"(CB|CENTRE\s*BACK|CENTER\s*BACK|CENTRAL\s*DEF(ENDER)?\b)"),
    ("RB", r"(RB|LB|RWB|LWB|WB|FULL\s*BACK|WING\s*BACK)"),
    ("CM", r"(DMF|CMF|AMF|DM|CM|AM|MIDFIELDER|MID\b)"),
    ("RW", r"(RW|LW|WINGER|RIGHT\s*WING|LEFT\s*WING|\bW(?!B)\b)"),
    ("CF", r"(CF|ST|FW|FORWARD|STRIKER|CENT(RE|ER)\s*FORWARD)"),
]
_primary = lambda pos: str(pos).split(",")[0].strip().upper() if pos else ""
def role5_from_pos(pos):
    if not pos: return ""
    tag=_primary(pos)
    if tag in WYS_TO_ROLE: return WYS_TO_ROLE[tag]
    for k,v in WYS_TO_ROLE.items():
        if k in tag: return v
    for role,pat in ROLE_PATTERNS:
        if re.search(pat, tag, flags=re.IGNORECASE): return role
    return ""
def role5_safe(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return None
    s=str(x).strip().upper()
    return s if s else None

# herní skupiny pro výběr CZ benchmarku (herní)
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

# ============ Matching (strict) ============
def match_by_name(df, name, team_hint=None, age_hint=None, nat_hint=None, min_score=8):
    """povinně stejné příjmení + skóre >= min_score; žádné párování jen přes klub."""
    if df is None or df.empty or not name: return pd.DataFrame()
    pcol=col_player(df)
    if pcol not in df.columns: return pd.DataFrame()

    if "_ksurname" not in df.columns:
        df["_kname"]=df[pcol].astype(str).map(_norm)
        fi,sn=zip(*df[pcol].astype(str).map(_split_name))
        df["_kfirst"],df["_ksurname"]=list(fi),list(sn)
        df["_kteam"]=df[col_team(df)].astype(str).map(_norm_team) if col_team(df) else ""
        df["_kage"]=pd.to_numeric(df[col_age(df)],errors="coerce").astype("Int64") if col_age(df) else pd.Series([pd.NA]*len(df),dtype="Int64")
        df["_knat"]=df[col_nat(df)].astype(str).map(_norm) if col_nat(df) else ""

    key_full=_norm(name); fi_key,sn_key=_split_name(name)
    team_key=_norm_team(team_hint) if team_hint else ""; nat_key=_norm(nat_hint) if nat_hint else ""
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
        s=4 # surname
        if fi_key and r["_kfirst"]==fi_key: s+=4
        if team_key:
            s+=4 if r["_kteam"]==team_key else (2 if (team_key in r["_kteam"] or r["_kteam"] in team_key) else 0)
        if age_key is not None and not pd.isna(r["_kage"]):
            d=abs(int(r["_kage"])-age_key); s+=3 if d==0 else 2 if d==1 else 1 if d==2 else 0
        if nat_key and r["_knat"]==nat_key: s+=2
        return s

    pool["_score"]=pool.apply(score,axis=1)
    best=pool.sort_values(["_score","_kage"],ascending=[False,True]).head(1)
    return best if (not best.empty and best["_score"].iloc[0]>=min_score) else pd.DataFrame()

# ============ Metrics & scoring ============
# Herní bloky
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
BLOCKS=[("Defenziva",DEF,"Defenziva"),("Ofenziva",OFF,"Ofenziva"),("Přihrávky",PAS,"Přihrávky"),("1v1",ONE,"1v1")]

ALIASES_GAME={
 "Cross accuracy, %":["Accurate crosses, %","Cross accuracy, %"],
 "Progressive passes per 90":["Progressive passes per 90","Progressive passes/90"],
 "Passes to final third per 90":["Passes to final third per 90","Passes to final third/90"],
 "Dribbles per 90":["Dribbles per 90","Dribbles/90"],
 "Progressive runs per 90":["Progressive runs per 90","Progressive runs/90"],
 "Second assists per 90":["Second assists per 90","Second assists/90"],
}
def s_alias(df,key,aliases): 
    if key in df.columns: return df[key]
    for c in aliases.get(key,[]): 
        if c in df.columns: return df[c]
    return None
def v_alias(row,key,aliases):
    if key in row.index: return row[key]
    for c in aliases.get(key,[]): 
        if c in row.index: return row[c]
    return np.nan

def ensure_run_wide(df):
    if df is None or df.empty: return df
    if {"Metric","Value"}.issubset(df.columns):
        idx=[c for c in [col_player(df),col_team(df),col_pos(df),"Age"] if c and c in df.columns]
        wide=df.pivot_table(index=idx,columns="Metric",values="Value",aggfunc="mean").reset_index()
        for c,d in [(col_player(df),"Player"),(col_pos(df),"Position")]:
            if c and c!=d and c in wide.columns: wide=wide.rename(columns={c:d})
        return wide
    return normalize(df)

def norm_pct(pop_series, val):
    s=pd.to_numeric(pop_series,errors="coerce").dropna()
    v=pd.to_numeric(pd.Series([val]),errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn,mx=s.min(),s.max()
    return 50.0 if mx==mn else float(np.clip((v-mn)/(mx-mn)*100,0,100))

def section_scores(row, agg, metric_w=None):
    sec_scores,sec_idx={},{}
    for _,lst,key in BLOCKS:
        vals={lab:norm_pct(s_alias(agg,eng,ALIASES_GAME), v_alias(row,eng,ALIASES_GAME)) for eng,lab in lst}
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

# ============ Running ============
RUN=[
 ("Total distance per 90","Total distance per 90",["Total distance per 90","Total distance/90","Distance per 90","Total distance (km) per 90","Distance P90"]),
 ("High-intensity runs per 90","High-intensity runs per 90",["High-intensity runs per 90","High intensity runs per 90","High intensity runs/90","HIR/90","HI Count P90"]),
 ("Sprints per 90","Sprints per 90",["Sprints per 90","Sprints/90","Number of sprints per 90","Sprint Count P90"]),
 ("Max speed (km/h)","Max speed (km/h)",["Max speed (km/h)","Top speed","Max velocity","Max speed","PSV-99","TOP 5 PSV-99"]),
 ("Average speed (km/h)","Average speed (km/h)",["Average speed (km/h)","Avg speed","Average velocity","M/min P90"]),
 ("Accelerations per 90","Accelerations per 90",["Accelerations per 90","Accelerations/90","Accels per 90","High Acceleration Count P90 + Medium Acceleration Count P90","High Acceleration Count P90","Medium Acceleration Count P90"]),
 ("Decelerations per 90","Decelerations per 90",["Decelerations per 90","Decelerations/90","Decels per 90","High Deceleration Count P90 + Medium Deceleration Count P90","High Deceleration Count P90","Medium Deceleration Count P90"]),
 ("High-speed distance per 90","High-speed distance per 90",["High-speed distance per 90","HS distance/90","High speed distance per 90","HSR Distance P90"]),
]
RUN_KEY="Běh"

def post_run(df):
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
    ren={}
    if "Player" not in run_df.columns:   ren[_best(run_df,["Name","player","name","Short Name"])]= "Player"
    if "Team"   not in run_df.columns:   ren[_best(run_df,["Club","team","Team"])]= "Team"
    if "Position" not in run_df.columns: ren[_best(run_df,["Pos","Role","Primary position","position"])]= "Position"
    ren={k:v for k,v in ren.items() if k}
    if ren: run_df=run_df.rename(columns=ren)
    run_df=ensure_run_wide(run_df); run_df=post_run(run_df)
    # role5 z herních
    if game_df is not None and not game_df.empty:
        g=normalize(game_df.copy())
        if {"Player","Position"}.issubset(g.columns):
            g=g[["Player","Position"]].dropna().copy()
            g["Role5"]=g["Position"].map(role5_from_pos)
            fi,sn=zip(*g["Player"].map(_split_name)); g["_k"]=pd.Series(fi,index=g.index)+"|"+pd.Series(sn,index=g.index)
            g=g.dropna(subset=["Role5","_k"]).groupby("_k",as_index=False)["Role5"].first()
            fi2,sn2=zip(*run_df["Player"].astype(str).map(_split_name)); run_df["_k"]=pd.Series(fi2,index=run_df.index)+"|"+pd.Series(sn2,index=run_df.index)
            run_df=run_df.merge(g[["_k","Role5"]],on="_k",how="left").drop(columns=["_k"])
    # fallback role5 z vlastního Position
    if "Role5" not in run_df.columns: run_df["Role5"]=np.nan
    mask=run_df["Role5"].isna() | (run_df["Role5"].astype(str).str.strip()=="")
    if "Position" in run_df.columns: run_df.loc[mask,"Role5"]=run_df.loc[mask,"Position"].map(role5_from_pos)
    return run_df

def run_scores_for_row(row, pop_agg):
    if pop_agg is None or pop_agg.empty: return {RUN_KEY:{}},{},np.nan
    scores,absv={},{}
    for col_key,label,opts in RUN:
        ser=next((pop_agg[c] for c in [col_key]+opts if c in pop_agg.columns), None)
        val=next((row[c] for c in [col_key]+opts if c in row.index), np.nan)
        if label=="Average speed (km/h)" and pd.isna(val) and "M/min P90" in row.index:
            val=pd.to_numeric(row["M/min P90"],errors="coerce")*0.06
        absv[label]=val if not pd.isna(val) else np.nan
        scores[label]=norm_pct(ser,val) if ser is not None else np.nan
    arr=[v for v in scores.values() if not pd.isna(v)]
    return {RUN_KEY:scores},absv,(float(np.mean(arr)) if arr else np.nan)

# ============ Rendering ============
def render_card(player,team,pos,age,scores,sec_index,overall,verdict,run_scores=None,run_abs=None,run_index=np.nan,final_index=None, role5=None):
    fig,ax=plt.subplots(figsize=(18,12)); ax.axis("off")
    ax.text(0.02,0.96,f"{player} (věk {age})",fontsize=20,fontweight="bold",va="top")
    ax.text(0.02,0.93,f"Klub: {team}   Pozice: {pos}{('   Role (běh): '+role5) if role5 else ''}",fontsize=13,va="top")
    y0=0.88
    for title,lst,key in BLOCKS:
        ax.text(0.02,y0,title,fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
        for i,(_,lab) in enumerate(lst):
            v=scores[key].get(lab,np.nan); x=L if i%2==0 else R
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(v),alpha=0.85,lw=0))
            ax.text(x+0.005,y-0.001,f"{lab}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=9,va="center",ha="left")
            if i%2==1: y-=0.038
        y0=y-0.025
    if run_scores and RUN_KEY in run_scores and not pd.isna(run_index):
        ax.text(0.02,y0,f"Běžecká data (vs. CZ benchmark{f' – role {role5}' if role5 else ''})",fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
        for i,(_,lab,_) in enumerate(RUN):
            p=run_scores[RUN_KEY].get(lab,np.nan); a=(run_abs or {}).get(lab,np.nan); x=L if i%2==0 else R
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(p),alpha=0.85,lw=0))
            ax.text(x+0.005,y-0.001,f"{lab}: {'n/a' if pd.isna(a) else (f'{a:.2f}' if isinstance(a,(int,float,np.number)) else str(a))} ({'n/a' if pd.isna(p) else str(int(round(p)))+'%'})",fontsize=9,va="center",ha="left")
            if i%2==1: y-=0.038
        y0=y-0.025
    ax.text(0.55,0.9,"Souhrnné indexy (0–100 %) – vážené",fontsize=16,fontweight="bold",va="top"); y=0.85
    for k in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v=sec_index.get(k,np.nan); ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(v),alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"{k}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=13,va="center",ha="left"); y-=0.075
    if not pd.isna(run_index):
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(run_index),alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"Běžecký index: {int(round(run_index))}%",fontsize=13,va="center",ha="left"); y-=0.075
    v=overall if final_index is None else final_index
    ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(v),alpha=0.7,lw=0))
    ax.text(0.56,y-0.005,f"{'Celkový role-index' if final_index is None else 'Celkový index (herní + běžecký)'}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=14,fontweight="bold",va="center",ha="left")
    ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color='lightgrey',alpha=0.35,lw=0))
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center")
    return fig

# ============ Sidebar ============
st.sidebar.header("⚙️ Váhy sekcí")
base_w={"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
sec_w={k:st.sidebar.slider(k,0,100,base_w[k],1) for k in base_w}
tot=sum(sec_w.values()) or 1
for k in sec_w: sec_w[k]=100.0*sec_w[k]/tot
w_run_pct=st.sidebar.slider("Váha běžeckého indexu v celkovém hodnocení",0,50,20,5)
th_agg=st.sidebar.selectbox("Prahování vs Slavia – statistika",["Medián","Průměr"],index=0)

metric_w={}
with st.sidebar.expander("Váhy metrik v sekcích (volitelné)",False):
    for _,lst,key in BLOCKS:
        st.markdown(f"**{key}**")
        tmp={lab:st.slider(f"– {lab}",0,100,10,1,key=f"{key}_{lab}") for _,lab in lst}
        s=sum(tmp.values()) or 1; metric_w[key]={lab:w/s for lab,w in tmp.items()} if s else None

# ============ Indexy + prahy ============
def compute_herni(row, league_agg):
    scores,sec_idx=section_scores(row,league_agg,metric_w)
    return scores,sec_idx,role_index(sec_idx,sec_w)

def run_part(row, run_cz_df, run_df_for_row, team, age, nat, pos_text):
    role5=role5_safe(role5_from_pos(pos_text))
    if (run_cz_df is None) or (run_df_for_row is None) or not role5: return None,None,np.nan,role5
    cand=match_by_name(run_df_for_row, row.get("Player",""), team_hint=team, age_hint=age, nat_hint=nat, min_score=8)
    cz_base=run_cz_df[run_cz_df.get("Role5","").astype(str).str.upper()==role5]
    if cand.empty or cz_base.empty: return None,None,np.nan,role5
    plc=col_player(cz_base); cz_agg=cz_base.groupby(plc).mean(numeric_only=True)
    return *run_scores_for_row(cand.iloc[0],cz_agg), role5

final_from = lambda overall,run_idx,w_run: (1.0-w_run)*overall + w_run*run_idx if not pd.isna(run_idx) else overall

def slavia_thresholds(cz_game_df, cz_run_df, w_run, how="Medián"):
    thr={}
    if cz_game_df is None or cz_game_df.empty: return thr
    g=normalize(cz_game_df.copy())
    if not {"Player","Team","Position"}.issubset(g.columns): return thr
    slv=g[g["Team"].astype(str).map(is_slavia)]
    vals=[]
    for _,r in slv.iterrows():
        pg=pos_group(r["Position"]); rgx=POS_REGEX[pg]
        cz_pos=g[g["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
        if cz_pos.empty: continue
        agg=cz_pos.groupby("Player").mean(numeric_only=True)
        _,sec_idx,overall=compute_herni(r, agg)
        role5=role5_safe(role5_from_pos(r["Position"]))
        run_idx=np.nan
        if cz_run_df is not None and role5:
            base=cz_run_df[cz_run_df.get("Role5","").astype(str).str.upper()==role5]
            if not base.empty and col_player(base) in base.columns:
                pcol=col_player(base)
                rows=base[base[pcol].astype(str).map(_norm)==_norm(r["Player"])]
                if rows.empty:
                    fi,sn=_split_name(r["Player"])
                    rows=base[base[pcol].astype(str).map(lambda x: _split_name(x)==(fi,sn))]
                if not rows.empty:
                    cz_agg=base.groupby(pcol).mean(numeric_only=True)
                    _,_,run_idx=run_scores_for_row(rows.iloc[0],cz_agg)
        final=final_from(overall,run_idx,w_run)
        if not pd.isna(final) and role5: vals.append((role5,float(final)))
    if not vals: return thr
    df=pd.DataFrame(vals,columns=["Role5","Final"])
    grp=df.groupby("Role5")["Final"]
    return {k:(float(grp.mean()) if how=="Průměr" else float(grp.median())) for k,grp in df.groupby("Role5")}

# ============ Tabs ============
tab_card, tab_search = st.tabs(["Karta hráče (herní + běžecká)", "Vyhledávání hráčů"])

# --- TAB 1 ---
with tab_card:
    c1,c2=st.columns(2)
    with c1:
        league_file=st.file_uploader("CZ liga – herní (xlsx)",["xlsx"],key="league_card")
        run_cz_file=st.file_uploader("CZ běžecká data – benchmark (xlsx)",["xlsx"],key="run_cz_card")
    with c2:
        players_file=st.file_uploader("Hráč/hráči – herní (xlsx)",["xlsx"],key="players_card")
        run_players_file=st.file_uploader("Hráč/hráči – běžecká (xlsx)",["xlsx"],key="run_players_card")

    have_game = bool(league_file and players_file)
    have_run  = bool(run_cz_file and run_players_file)

    if not have_game and not have_run:
        st.info("➡️ Nahraj buď (a) CZ herní + hráčský herní export, nebo (b) CZ běžecký benchmark + běžecký export."); st.stop()

    # JEN běžecká
    if (not have_game) and have_run:
        cz_run=auto_fix_run_df(pd.read_excel(run_cz_file), None)
        any_run=auto_fix_run_df(pd.read_excel(run_players_file), None)
        pcol=col_player(any_run); sel=st.selectbox("Vyber hráče (běžecký export)", any_run[pcol].dropna().unique().tolist())
        row=any_run.loc[any_run[pcol]==sel].iloc[0]
        role5=role5_safe(any_run.loc[any_run[pcol]==sel].iloc[0].get("Role5","") or role5_from_pos(row.get("Position","")))
        cz_base=cz_run[cz_run.get("Role5","").astype(str).str.upper()==(role5 or "")]
        if cz_base.empty: r_scores,r_abs,run_idx={RUN_KEY:{}},{},np.nan
        else:
            cz_agg=cz_base.groupby(col_player(cz_base)).mean(numeric_only=True)
            r_scores,r_abs,run_idx=run_scores_for_row(row,cz_agg)
        verdict="ANO – běžecky vhodný (55%+)" if (not pd.isna(run_idx) and run_idx>=55) else ("OK – 45–55%" if (not pd.isna(run_idx) and run_idx>=45) else "NE – pod úrovní")
        fig=render_card(row.get("Player",""),row.get("Team",""),row.get("Position","—"),row.get("Age","n/a"),{},{},np.nan,verdict,r_scores,r_abs,run_idx,role5=role5)
        st.pyplot(fig); bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight"); st.download_button("📥 Stáhnout běžeckou kartu",data=bio.getvalue(),file_name=f"{sel}_run.png",mime="image/png"); st.stop()

    # Herní / kombinovaná
    league=normalize(pd.read_excel(league_file))
    players=normalize(pd.read_excel(players_file))
    run_cz_df=auto_fix_run_df(pd.read_excel(run_cz_file), league) if run_cz_file else None
    run_pl_df=auto_fix_run_df(pd.read_excel(run_players_file), players) if run_players_file else None

    w_run=w_run_pct/100.0
    slv_thr=slavia_thresholds(league, run_cz_df, w_run, how=th_agg)

    sel=st.selectbox("Vyber hráče (herní export)", players["Player"].dropna().unique().tolist())
    row=players.loc[players["Player"]==sel].iloc[0]
    player,team,pos,age,nat=row.get("Player",""),row.get("Team",""),row.get("Position",""),row.get("Age","n/a"),row.get("Nationality","")

    pg=pos_group(pos); rgx=POS_REGEX[pg]
    cz_pos=league[league["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
    agg=cz_pos.groupby("Player").mean(numeric_only=True)

    scores,sec_idx,overall=compute_herni(row, agg)
    r_scores,r_abs,run_idx,role5 = run_part(row, run_cz_df, run_pl_df, team, age, nat, pos)
    final_idx=final_from(overall, run_idx, w_run)
    thr=slv_thr.get(role5, np.nan)
    verdict = "ANO – potenciální posila do Slavie" if (not pd.isna(final_idx) and not pd.isna(thr) and final_idx>=thr) else "NE – nedosahuje úrovně Slavie (role)"

    fig=render_card(player,team,pos,age,scores,sec_idx,overall,verdict,r_scores,r_abs,run_idx,final_index=final_idx, role5=role5)
    st.pyplot(fig); bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight"); st.download_button("📥 Stáhnout kartu (PNG)",data=bio.getvalue(),file_name=f"{player}.png",mime="image/png")

# --- TAB 2 ---
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
        cz_df=normalize(load_xlsx(st.session_state["cz_bytes"]))
        fr_df=normalize(load_xlsx(st.session_state["fr_bytes"]))
        if "Position" not in cz_df.columns or "Position" not in fr_df.columns:
            st.error("V jednom ze souborů chybí sloupec s pozicí."); st.stop()
        cz_run_df=auto_fix_run_df(load_xlsx(st.session_state.get("cz_run_bytes")),cz_df) if "cz_run_bytes" in st.session_state else None
        fr_run_df=auto_fix_run_df(load_xlsx(st.session_state.get("fr_run_bytes")),fr_df) if "fr_run_bytes" in st.session_state else None
        w_run=w_run_pct/100.0
        slv_thr=slavia_thresholds(cz_df, cz_run_df, w_run, how=th_agg)

        def search():
            mask=pd.Series(False,index=fr_df.index)
            for p in pos_sel: mask |= fr_df["Position"].astype(str).str.contains(POS_REGEX[p],na=False,regex=True)
            base=fr_df.loc[mask].copy()
            def pick(names): return next((n for n in names if n in base.columns),None)
            mc,p_gc=pick(["Minutes","Minutes played","Min"]),pick(["Games","Matches"])
            if min_minutes and mc: base=base[pd.to_numeric(base[mc],errors="coerce").fillna(0)>=min_minutes]
            if min_games and p_gc: base=base[pd.to_numeric(base[p_gc],errors="coerce").fillna(0)>=min_games]

            rows,cards=[],[]
            for _,r in base.iterrows():
                pg=pos_group(r["Position"]); rgx=POS_REGEX[pg]
                cz_pos=cz_df[cz_df["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
                if cz_pos.empty: continue
                agg=cz_pos.groupby("Player").mean(numeric_only=True)
                scores,sec_idx,overall=compute_herni(r, agg)
                r_scores,r_abs,run_idx,role5 = run_part(r, cz_run_df, fr_run_df, r.get("Team",""), r.get("Age",None), r.get("Nationality",""), r["Position"])
                final_idx=final_from(overall, run_idx, w_run)
                thr=slv_thr.get(role5, np.nan)
                if not pd.isna(final_idx) and not pd.isna(thr) and final_idx>=thr:
                    verdict="ANO – potenciální posila do Slavie"
                    player=r.get("Player",""); team=r.get("Team",""); age=r.get("Age","n/a"); pos_txt=r.get("Position","")
                    rows.append({"Hráč":player,"Věk":age,"Klub":team,"Pozice":pos_txt,"Liga":league_name,"Role5":role5,
                                 "Index Def":sec_idx.get("Defenziva",np.nan),"Index Off":sec_idx.get("Ofenziva",np.nan),
                                 "Index Pass":sec_idx.get("Přihrávky",np.nan),"Index 1v1":sec_idx.get("1v1",np.nan),
                                 "Role-index (vážený)":overall,"Run index":run_idx,"Final index":final_idx,
                                 "Prahová hodnota Slavia (role)":thr,"Verdikt":verdict})
                    fig=render_card(player,team,pos_txt,age,scores,sec_idx,overall,verdict,r_scores,r_abs,run_idx,final_index=final_idx, role5=role5)
                    bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight"); plt.close(fig)
                    cards.append((str(player),bio.getvalue()))
            return pd.DataFrame(rows),cards

        res_df,cards=search()
        st.session_state.update(search_results=res_df,search_cards=cards,league_name=league_name)

    res_df=st.session_state.get("search_results")
    if res_df is None or res_df.empty:
        st.info("Zatím žádné výsledky – nahraj soubory a klikni na *Spustit vyhledávání*.")
    else:
        st.success(f"Nalezeno kandidátů (Verdikt = ANO): {len(res_df)}")
        st.dataframe(res_df.sort_values(["Role5","Final index"],ascending=[True,False]), use_container_width=True)
        st.download_button("📥 Stáhnout CSV (ANO)", data=res_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"kandidati_{st.session_state.get('league_name','liga')}_ANO.csv", mime="text/csv")
        zbuf=BytesIO()
        with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
            for name,png in (st.session_state.get("search_cards") or []):
                safe=str(name).replace("/","_").replace("\\","_"); zf.writestr(f"{safe}.png", png)
        st.download_button("🗂️ Stáhnout všechny karty (ZIP)", data=zbuf.getvalue(),
                           file_name=f"karty_{st.session_state.get('league_name','liga')}_ANO.zip", mime="application/zip")
        with st.expander("🖼️ Online karty (všichni s verdiktem ANO)"):
            for name,png in (st.session_state.get("search_cards") or []):
                st.image(png, caption=name, use_column_width=True)
