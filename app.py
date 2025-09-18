# app.py  — kompaktní verze (herní karty + vyhledávání + samostatná běžecká karta)
import io, re, unicodedata, zipfile
from io import BytesIO
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

st.set_page_config(page_title="Karty – Slavia standard (váhy + běžecká data)", layout="wide")
st.title("⚽ Generátor datových karet (váhový model + vyhledávání hráčů + běžecká data)")

# ------------ helpers ------------
@st.cache_data
def load_xlsx(file_bytes: bytes) -> pd.DataFrame: return pd.read_excel(BytesIO(file_bytes))

def color_for(v):
    v = np.nan if v is None else v
    if pd.isna(v): return "lightgrey"
    return "#FF4C4C" if v<=25 else "#FF8C00" if v<=50 else "#FFD700" if v<=75 else "#228B22"

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

def get_value_with_alias(row,key):
    if key in row.index: return row[key]
    for c in ALIASES.get(key,[]): 
        if c in row.index: return row[c]
    if key=="Cross accuracy, %" and "Accurate crosses, %" in row.index: return row["Accurate crosses, %"]
    return np.nan

# --- běh (kanonická jména) ---
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
 "Accelerations per 90":["Accelerations per 90","Accelerations/90","Accels per 90","High Acceleration Count P90 + Medium Acceleration Count P90",
                         "High Acceleration Count P90","Medium Acceleration Count P90"],
 "Decelerations per 90":["Decelerations per 90","Decelerations/90","Decels per 90","High Deceleration Count P90 + Medium Deceleration Count P90",
                         "High Deceleration Count P90","Medium Deceleration Count P90"],
 "High-speed distance per 90":["High-speed distance per 90","HS distance/90","High speed distance per 90","HSR Distance P90"]
}
def value_with_alias_run(row,key):
    if key in row.index: return row[key]
    for c in ALIASES_RUN.get(key,[]):
        if c in row.index: return row[c]
    return np.nan
def series_for_alias_run(df,key):
    if df is None or df.empty: return None
    if key in df.columns: return df[key]
    for c in ALIASES_RUN.get(key,[]):
        if c in df.columns: return df[c]
    return None

def _best_col(df,names):
    for n in names:
        if n in df.columns: return n

def get_pos_col(df): 
    return _best_col(df,["Position","Pos","position","Role","Primary position"])
def get_player_col(df): 
    return _best_col(df,["Player","Name","player","name","Short Name"])

def ensure_run_wide(df):
    if df is None or df.empty: return df
    if "Metric" in df.columns and "Value" in df.columns:
        pcol=get_pos_col(df); pl=get_player_col(df) or "Player"
        idx=[c for c in [pl,"Team",pcol,"Age"] if c and c in df.columns]
        wide=df.pivot_table(index=idx,columns="Metric",values="Value",aggfunc="mean").reset_index()
        if pl!="Player" and pl in wide.columns: wide=wide.rename(columns={pl:"Player"})
        if pcol and pcol!="Position" and pcol in wide.columns: wide=wide.rename(columns={pcol:"Position"})
        return wide
    pcol=get_pos_col(df); pl=get_player_col(df)
    if pcol and pcol!="Position": df=df.rename(columns={pcol:"Position"})
    if pl and pl!="Player": df=df.rename(columns={pl:"Player"})
    return df

def _strip_accents_lower(s):
    if not isinstance(s,str): return ""
    s=unicodedata.normalize("NFKD",s)
    s="".join([c for c in s if not unicodedata.combining(c)])
    return re.sub(r"\s+"," ",s).strip().lower()

def auto_fix_run_df(run_df, game_df=None):
    if run_df is None or run_df.empty: return run_df
    # přímé rename
    run_df = run_df.rename(columns={
        "Distance P90":"Total distance per 90",
        "HSR Distance P90":"High-speed distance per 90",
        "HI Count P90":"High-intensity runs per 90",
        "Sprint Count P90":"Sprints per 90",
        "PSV-99":"Max speed (km/h)"
    })
    # id aliasy
    idm={}
    if "Player" not in run_df.columns:
        c=get_player_col(run_df)
        if c: idm[c]="Player"
    if "Team" not in run_df.columns:
        c=_best_col(run_df,["Club","team"])
        if c: idm[c]="Team"
    if "Position" not in run_df.columns:
        c=get_pos_col(run_df)
        if c: idm[c]="Position"
    if idm: run_df=run_df.rename(columns=idm)

    run_df=ensure_run_wide(run_df)

    # dopočty
    if "Average speed (km/h)" not in run_df.columns and "M/min P90" in run_df.columns:
        run_df["Average speed (km/h)"]=pd.to_numeric(run_df["M/min P90"],errors="coerce")*0.06
    if "Accelerations per 90" not in run_df.columns:
        a=[]
        for c in ["High Acceleration Count P90","Medium Acceleration Count P90"]:
            if c in run_df.columns: a.append(pd.to_numeric(run_df[c],errors="coerce"))
        if a:
            s=a[0]
            for x in a[1:]: s=s.add(x,fill_value=0)
            run_df["Accelerations per 90"]=s
    if "Decelerations per 90" not in run_df.columns:
        d=[]
        for c in ["High Deceleration Count P90","Medium Deceleration Count P90"]:
            if c in run_df.columns: d.append(pd.to_numeric(run_df[c],errors="coerce"))
        if d:
            s=d[0]
            for x in d[1:]: s=s.add(x,fill_value=0)
            run_df["Decelerations per 90"]=s

    # doplnění Position z herních (volitelné)
    if "Position" not in run_df.columns and game_df is not None and not game_df.empty:
        g=game_df.copy()
        pc=get_player_col(g); 
        if pc and pc!="Player": g=g.rename(columns={pc:"Player"})
        if "Position" in g.columns and "Player" in g.columns:
            g=g[["Player","Position"]].dropna().groupby("Player",as_index=False).first()
            run_df["_k"]=run_df["Player"].astype(str).map(_strip_accents_lower)
            g["_k"]=g["Player"].astype(str).map(_strip_accents_lower)
            run_df=run_df.merge(g[["_k","Position"]],on="_k",how="left")
            run_df=run_df.drop(columns=["_k"],errors="ignore")

    for c in ["Player","Team","Position"]:
        if c in run_df.columns: run_df[c]=run_df[c].astype(str).str.strip()
    return run_df

# ------------ výpočty ------------
def series_for_alias(agg,key):
    if key in agg.columns: return agg[key]
    for c in ALIASES.get(key,[]):
        if c in agg.columns: return agg[c]
    if key=="Cross accuracy, %" and "Accurate crosses, %" in agg.columns: return agg["Accurate crosses, %"]

def normalize_metric(agg,key,value):
    s=series_for_alias(agg,key)
    if s is None or pd.isna(value): return np.nan
    s=pd.to_numeric(s,errors="coerce").dropna()
    v=pd.to_numeric(pd.Series([value]),errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn,mx=s.min(),s.max()
    if mx==mn: return 50.0
    return float(np.clip((v-mn)/(mx-mn)*100.0,0,100))

def compute_section_scores(player_row,agg,blocks,metric_weights=None):
    sec_scores,sec_index={},{}
    for _,lst,key in blocks:
        part={label: normalize_metric(agg,eng,get_value_with_alias(player_row,eng)) for eng,label in lst}
        sec_scores[key]=part
        if metric_weights and metric_weights.get(key):
            wsum=sum(w for w in metric_weights[key].values())
            sec_index[key]= (sum(part.get(l,np.nan)*w for l,w in metric_weights[key].items() if not pd.isna(part.get(l))) / wsum) if wsum else np.nan
        else:
            vals=[v for v in part.values() if not pd.isna(v)]
            sec_index[key]=float(np.mean(vals)) if vals else np.nan
    return sec_scores,sec_index

def weighted_role_index(sec_index,sec_weights):
    acc=tot=0.0
    for s in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v=sec_index.get(s,np.nan)
        if not pd.isna(v): 
            w=sec_weights.get(s,0)/100.0
            acc+=v*w; tot+=w
    return float(acc/tot) if tot>0 else np.nan

def normalize_run_metric(cz_agg,key,value):
    s=series_for_alias_run(cz_agg,key)
    if s is None or pd.isna(value): return np.nan
    s=pd.to_numeric(s,errors="coerce").dropna()
    v=pd.to_numeric(pd.Series([value]),errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn,mx=s.min(),s.max()
    if mx==mn: return 50.0
    return float(np.clip((v-mn)/(mx-mn)*100.0,0,100))

def compute_run_scores(player_row,cz_run_agg):
    if cz_run_agg is None or cz_run_agg.empty: return {RUN_KEY:{}},{},np.nan
    rs,ra={},{}
    for eng,label in RUN:
        val=value_with_alias_run(player_row,eng)
        if pd.isna(val) and eng=="Average speed (km/h)" and "M/min P90" in player_row.index:
            val=pd.to_numeric(player_row["M/min P90"],errors="coerce")*0.06
        ra[label]=val if not pd.isna(val) else np.nan
        rs[label]=normalize_run_metric(cz_run_agg,eng,val)
    vals=[v for v in rs.values() if not pd.isna(v)]
    return {RUN_KEY:rs},ra,(float(np.mean(vals)) if vals else np.nan)

# ------------ vizualizace ------------
def render_card_visual(player,team,pos,age,scores,sec_index,overall,verdict,run_scores=None,run_abs=None,run_index=np.nan,final_index=None):
    fig,ax=plt.subplots(figsize=(18,12)); ax.axis("off")
    ax.text(0.02,0.96,f"{player} (věk {age})",fontsize=20,fontweight="bold",va="top")
    ax.text(0.02,0.93,f"Klub: {team}   Pozice: {pos}",fontsize=13,va="top")
    y0=0.88
    for title,lst,key in blocks:
        ax.text(0.02,y0,title,fontsize=15,fontweight="bold",va="top"); y=y0-0.04
        for i,(_,lab) in enumerate(lst):
            v=scores[key].get(lab,np.nan); x=0.04 if i%2==0 else 0.26
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(v),alpha=0.85,lw=0))
            ax.text(x+0.005,y-0.001,f"{lab}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=9,va="center")
            if i%2==1: y-=0.038
        y0=y-0.025
    if run_scores and RUN_KEY in run_scores and len(run_scores[RUN_KEY])>0:
        ax.text(0.02,y0,"Běžecká data",fontsize=15,fontweight="bold",va="top"); y=y0-0.04
        for i,(_,lab) in enumerate(RUN):
            vp=run_scores[RUN_KEY].get(lab,np.nan); va=(run_abs or {}).get(lab,np.nan); x=0.04 if i%2==0 else 0.26
            ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(vp),alpha=0.85,lw=0))
            ax.text(x+0.005,y-0.001,f"{lab}: {'n/a' if pd.isna(va) else (f'{va:.2f}' if isinstance(va,(int,float,np.number)) else str(va))} ({'n/a' if pd.isna(vp) else str(int(round(vp)))+'%'})",fontsize=9,va="center")
            if i%2==1: y-=0.038
        y0=y-0.025
    ax.text(0.55,0.9,"Souhrnné indexy (0–100 %) – vážené",fontsize=16,fontweight="bold",va="top"); y=0.85
    for k in ["Defenziva","Ofenziva","Přihrávky","1v1"]:
        v=sec_index.get(k,np.nan); ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(v),alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"{k}: {'n/a' if pd.isna(v) else str(int(round(v)))+'%'}",fontsize=13,va="center"); y-=0.075
    if not pd.isna(run_index):
        ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(run_index),alpha=0.7,lw=0))
        ax.text(0.56,y-0.005,f"Běžecký index: {int(round(run_index))}%",fontsize=13,va="center"); y-=0.075
    label="Celkový role-index (vážený)" if (final_index is None) else "Celkový index (herní + běžecký)"
    val=overall if (final_index is None) else final_index
    ax.add_patch(Rectangle((0.55,y-0.03),0.38,0.05,color=color_for(val),alpha=0.7,lw=0))
    ax.text(0.56,y-0.005,f"{label}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}",fontsize=14,fontweight="bold",va="center")
    ax.add_patch(Rectangle((0.55,0.02),0.38,0.07,color='lightgrey',alpha=0.35,lw=0))
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center")
    return fig

def render_run_card_visual(player,team,pos,age,run_scores,run_abs,run_index):
    fig,ax=plt.subplots(figsize=(18,10)); ax.axis("off")
    ax.text(0.02,0.95,f"{player} (věk {age})",fontsize=20,fontweight="bold",va="top")
    ax.text(0.02,0.92,f"Klub: {team}   Pozice: {pos if pd.notna(pos) else '—'}",fontsize=13,va="top")
    ax.text(0.02,0.87,"Běžecká data (vs. CZ benchmark)",fontsize=16,fontweight="bold",va="top"); y=0.83
    for i,(_,lab) in enumerate(RUN):
        vp=run_scores.get(RUN_KEY,{}).get(lab,np.nan); va=(run_abs or {}).get(lab,np.nan); x=0.04 if i%2==0 else 0.26
        ax.add_patch(Rectangle((x,y-0.018),0.18,0.034,color=color_for(vp),alpha=0.85,lw=0))
        ax.text(x+0.005,y-0.001,f"{lab}: {'n/a' if pd.isna(va) else (f'{va:.2f}' if isinstance(va,(int,float,np.number)) else str(va))} ({'n/a' if pd.isna(vp) else str(int(round(vp)))+'%'})",fontsize=10,va="center")
        if i%2==1: y-=0.038
    ax.text(0.55,0.87,"Souhrn",fontsize=16,fontweight="bold",va="top")
    ax.add_patch(Rectangle((0.55,0.83-0.03),0.38,0.05,color=color_for(run_index),alpha=0.7,lw=0))
    ax.text(0.56,0.83-0.005,f"Běžecký index: {'n/a' if pd.isna(run_index) else str(int(round(run_index)))+'%'}",fontsize=14,va="center")
    return fig

# ------------ peers / pozice ------------
POS_REGEX={"CB/DF":r"(CB|DF)","RB":r"(RB)","LB":r"(LB)","WB/RWB/LWB":r"(WB|RWB|LWB)","DM":r"(DM)","CM":r"(CM)","AM":r"(AM)","RW":r"(RW)","LW":r"(LW)","CF/ST":r"(CF|ST|FW)"}
def resolve_pos_group(p):
    p=(p or "").upper()
    if any(k in p for k in ["CB","DF"]): return "CB/DF"
    for k in ["RB","LB","RWB","LWB","WB","DM","CM","AM","RW","LW","CF","ST","FW"]:
        if k in p: return {"RWB":"WB/RWB/LWB","LWB":"WB/RWB/LWB","WB":"WB/RWB/LWB","CF":"CF/ST","ST":"CF/ST","FW":"CF/ST"}.get(k,k)
    return "CM"
SLAVIA_PEERS={"RB":["D. Douděra","D. Hashioka"],"LB":["O. Zmrzlý","J. Bořil"],"WB/RWB/LWB":["D. Douděra","D. Hashioka","O. Zmrzlý"],
"CB/DF":["I. Ogbu","D. Zima","T. Holeš","J. Bořil"],"DM":["T. Holeš","O. Dorley","M. Sadílek"],
"CM":["C. Zafeiris","L. Provod","E. Prekop","M. Sadílek"],"AM":["C. Zafeiris","L. Provod","E. Prekop"],
"RW":["I. Schranz","Y. Sanyang","V. Kušej"],"LW":["I. Schranz","V. Kušej"],"CF/ST":["M. Chytil","T. Chorý"]}
def peers_for_pos_group(pg): return SLAVIA_PEERS.get(pg,[])

def compute_overall_for_row(row,cz_agg,sec_weights,metric_weights):
    s,sec=compute_section_scores(row,cz_agg,blocks,metric_weights); return s,sec,weighted_role_index(sec,sec_weights)

def avg_peer_index(cz_agg,pos_group,sec_weights,metric_weights):
    vals=[]
    for nm in peers_for_pos_group(pos_group):
        if nm not in cz_agg.index: continue
        r=cz_agg.loc[nm].copy(); r["Player"]=nm; r["Team"]=r.get("Team","Slavia"); r["Position"]=pos_group; r["Age"]=r.get("Age",np.nan)
        _,_,ov=compute_overall_for_row(r,cz_agg,sec_weights,metric_weights)
        if not np.isnan(ov): vals.append(ov)
    return float(np.mean(vals)) if vals else np.nan

def run_index_for_row(row,cz_run_df_pos):
    if cz_run_df_pos is None or cz_run_df_pos.empty: return np.nan,{},{}
    plc=get_player_col(cz_run_df_pos) or "Player"
    cz_tmp=cz_run_df_pos.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_run_df_pos.columns else cz_run_df_pos
    cz_run_agg=cz_tmp.groupby("Player").mean(numeric_only=True)
    return compute_run_scores(row,cz_run_agg)

# ------------ UI: sidebar ------------
st.sidebar.header("⚙️ Váhy sekcí")
default={"Defenziva":25,"Ofenziva":25,"Přihrávky":25,"1v1":25}
sec_weights={k: st.sidebar.slider(k,0,100,default[k],1) for k in default}
tot=sum(sec_weights.values()) or 1
for k in sec_weights: sec_weights[k]=100.0*sec_weights[k]/tot

with st.sidebar.expander("Váhy metrik v sekcích (volitelné)",expanded=False):
    metric_weights={}
    for title,lst,key in blocks:
        st.markdown(f"**{title}**")
        tmp={label: st.slider(f"– {label}",0,100,10,1,key=f"{key}_{label}") for _,label in lst}
        s=sum(tmp.values()); metric_weights[key]=None if s==0 else {lab: w/s for lab,w in tmp.items()}

# ------------ Tabs ------------
tab_card, tab_search, tab_runonly = st.tabs(["Karta hráče","Vyhledávání hráčů","Běžecká karta (jen běh)"])

# --- Tab 1: karta hráče ---
with tab_card:
    c1,c2=st.columns(2)
    with c1:
        league_file=st.file_uploader("CZ liga – herní data (xlsx)",type=["xlsx"],key="league_card")
        run_cz_file=st.file_uploader("CZ běžecká data (xlsx)",type=["xlsx"],key="run_cz_card")
    with c2:
        players_file=st.file_uploader("Hráč/hráči – herní data (xlsx)",type=["xlsx"],key="players_card")
        run_players_file=st.file_uploader("Hráč/hráči – běžecká data (xlsx) [volitelné]",type=["xlsx"],key="run_players_card")
    if not league_file or not players_file:
        st.info("➡️ Nahraj minimálně CZ herní dataset + hráčský herní export."); st.stop()
    try:
        league=pd.read_excel(league_file); players=pd.read_excel(players_file)
    except Exception as e:
        st.error(f"Chyba při načítání herních souborů: {e}"); st.stop()

    run_cz_df=auto_fix_run_df(pd.read_excel(run_cz_file),league) if run_cz_file else None
    run_players_df=auto_fix_run_df(pd.read_excel(run_players_file),players) if run_players_file else None

    sel=st.selectbox("Vyber hráče (herní export)", players["Player"].dropna().unique().tolist())
    row=players.loc[players["Player"]==sel].iloc[0]
    player=row.get("Player",""); team=row.get("Team",""); pos=row.get("Position","")
    pos_group=resolve_pos_group(str(pos)); rgx=POS_REGEX[pos_group]
    group=league[league["Position"].astype(str).str.contains(rgx,na=False,regex=True)].copy()
    agg=group.groupby("Player").mean(numeric_only=True)

    scores,block_idx=compute_section_scores(row,agg,blocks,metric_weights)
    overall=weighted_role_index(block_idx,sec_weights)

    run_scores=run_abs=None; run_index=np.nan
    if run_cz_df is not None and run_players_df is not None:
        posc=get_pos_col(run_cz_df); cz_pos=run_cz_df[run_cz_df[posc].astype(str).str.contains(rgx,na=False,regex=True)] if posc else pd.DataFrame()
        if not cz_pos.empty:
            plcol=get_player_col(run_players_df); cand=run_players_df.loc[run_players_df[plcol]==player] if plcol else pd.DataFrame()
            if not cand.empty:
                row_run=cand.iloc[0]; run_scores,run_abs,run_index=run_index_for_row(row_run,cz_pos)
        with st.expander("Kontrola běžeckých dat",expanded=False):
            cz_agg_tmp=None
            if not cz_pos.empty:
                plc=get_player_col(cz_pos) or "Player"
                cz_tmp=cz_pos.rename(columns={plc:"Player"}) if plc!="Player" and plc in cz_pos.columns else cz_pos
                cz_agg_tmp=cz_tmp.groupby("Player").mean(numeric_only=True)
            miss_cz=[lab for eng,lab in RUN if series_for_alias_run(cz_agg_tmp,eng) is None]
            miss_pl=[lab for eng,lab in RUN if pd.isna(value_with_alias_run(row_run if 'row_run' in locals() else pd.Series(dtype=object),eng))]
            st.write(f"Chybějící metriky v CZ benchmarku: {', '.join(miss_cz) if miss_cz else '—'}")
            st.write(f"Chybějící metriky u hráče: {', '.join(miss_pl) if miss_pl else '—'}")
            present=sum([0 if (run_scores is None or pd.isna(run_scores[RUN_KEY].get(lab,np.nan))) else 1 for _,lab in RUN])
            st.write(f"Metrik započteno do Run indexu: {present}/{len(RUN)}")
            if present<=4: st.warning("Běžecké hodnocení je málo spolehlivé (≤ 4 metrik).")

    peer_avg=avg_peer_index(agg,pos_group,sec_weights,metric_weights)
    verdict=("ANO – potenciální posila do Slavie" if (not np.isnan(peer_avg) and not np.isnan(overall) and overall>=peer_avg)
             else "NE – nedosahuje úrovně slávistických konkurentů")

    fig=render_card_visual(player,team,pos,row.get("Age","n/a"),scores,block_idx,overall,verdict,run_scores,run_abs,run_index)
    st.pyplot(fig)
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=180,bbox_inches="tight")
    st.download_button("📥 Stáhnout kartu jako PNG",data=buf.getvalue(),file_name=f"{player}.png",mime="image/png")

# --- Tab 2: vyhledávání ---
with tab_search:
    st.subheader("Vyhledávání kandidátů pro Slavii (benchmark = CZ liga)")
    colA,colB=st.columns(2)
    with colA:
        cz_file=st.file_uploader("CZ liga – herní (xlsx)",type=["xlsx"],key="cz_search")
        run_cz_file=st.file_uploader("CZ běžecká data (xlsx) [volitelné]",type=["xlsx"],key="cz_run_search")
    with colB:
        fr_file=st.file_uploader("Cizí liga – herní (xlsx)",type=["xlsx"],key="fr_search")
        run_fr_file=st.file_uploader("Cizí liga – běžecká (xlsx) [volitelné]",type=["xlsx"],key="fr_run_search")

    if cz_file: st.session_state["cz_bytes"]=cz_file.getvalue()
    if fr_file: st.session_state["fr_bytes"]=fr_file.getvalue()
    if run_cz_file: st.session_state["cz_run_bytes"]=run_cz_file.getvalue()
    if run_fr_file: st.session_state["fr_run_bytes"]=run_fr_file.getvalue()

    positions=st.multiselect("Pozice", list(POS_REGEX.keys()), default=list(POS_REGEX.keys()))
    c1,c2,c3=st.columns(3)
    with c1: league_name=st.text_input("Název ligy (zobrazí se ve výstupu)",value="Cizí liga")
    with c2: min_minutes=st.number_input("Min. minut (pokud ve zdroji)",min_value=0,value=0,step=100)
    with c3: min_games=st.number_input("Min. zápasů (pokud ve zdroji)",min_value=0,value=0,step=1)
    w_run=st.slider("Váha běžeckého indexu v celkovém hodnocení",0,50,0,5)/100.0

    if st.button("Spustit vyhledávání"):
        if "cz_bytes" not in st.session_state or "fr_bytes" not in st.session_state:
            st.error("Nahraj alespoň CZ herní + cizí liga herní."); st.stop()
        cz_df=load_xlsx(st.session_state["cz_bytes"]); fr_df=load_xlsx(st.session_state["fr_bytes"])
        cz_run_df=auto_fix_run_df(load_xlsx(st.session_state["cz_run_bytes"]),cz_df) if "cz_run_bytes" in st.session_state else None
        fr_run_df=auto_fix_run_df(load_xlsx(st.session_state["fr_run_bytes"]),fr_df) if "fr_run_bytes" in st.session_state else None

        def search_candidates(cz_df,fr_df,positions,sec_weights,metric_weights,min_minutes=None,min_games=None,league_name="",cz_run_df=None,fr_run_df=None,w_run=0.0):
            mask=pd.Series(False,index=fr_df.index)
            for p in positions: mask|=fr_df["Position"].astype(str).str.contains(POS_REGEX[p],na=False,regex=True)
            base=fr_df.loc[mask].copy()
            def pick(df,names): return next((n for n in names if n in df.columns),None)
            mc=pick(base,["Minutes","Minutes played","Min"]); gc=pick(base,["Games","Matches"])
            if min_minutes and mc: base=base[pd.to_numeric(base[mc],errors="coerce").fillna(0)>=min_minutes]
            if min_games and gc: base=base[pd.to_numeric(base[gc],errors="coerce").fillna(0)>=min_games]

            rows,cards=[],[]
            for _,r in base.iterrows():
                pos_group=resolve_pos_group(str(r.get("Position",""))); rgx=POS_REGEX[pos_group]
                cz_pos=cz_df[cz_df["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
                if cz_pos.empty: continue
                cz_agg=cz_pos.groupby("Player").mean(numeric_only=True)

                scores,sec_idx,overall=compute_overall_for_row(r,cz_agg,sec_weights,metric_weights)

                run_idx=np.nan; run_scores=run_abs=None
                if cz_run_df is not None and fr_run_df is not None:
                    posc=get_pos_col(cz_run_df); cz_run_pos=cz_run_df[cz_run_df[posc].astype(str).str.contains(rgx,na=False,regex=True)] if posc else pd.DataFrame()
                    pl=get_player_col(fr_run_df); cand=fr_run_df.loc[fr_run_df[pl]==r.get("Player","")] if pl else pd.DataFrame()
                    if not cz_run_pos.empty and not cand.empty:
                        r_run=cand.iloc[0]; run_scores,run_abs,run_idx=run_index_for_row(r_run,cz_run_pos)

                final_idx= (1.0-w_run)*overall + w_run*run_idx if (not pd.isna(run_idx) and w_run>0) else overall
                peer_avg=avg_peer_index(cz_agg,pos_group,sec_weights,metric_weights)
                verdict=("ANO – potenciální posila do Slavie" if (not np.isnan(peer_avg) and not np.isnan(final_idx) and final_idx>=peer_avg)
                         else "NE – nedosahuje úrovně slávistických konkurentů")

                if verdict.startswith("ANO"):
                    player=r.get("Player",""); team=r.get("Team",""); pos=r.get("Position",""); age=r.get("Age","n/a")
                    rows.append({"Hráč":player,"Věk":age,"Klub":team,"Pozice":pos,"Liga":league_name,
                                 "Index Def":sec_idx.get("Defenziva",np.nan),"Index Off":sec_idx.get("Ofenziva",np.nan),
                                 "Index Pass":sec_idx.get("Přihrávky",np.nan),"Index 1v1":sec_idx.get("1v1",np.nan),
                                 "Role-index (vážený)":overall,"Run index":run_idx if not pd.isna(run_idx) else None,
                                 "Final index":final_idx if not pd.isna(run_idx) and w_run>0 else None,
                                 "Verdikt":verdict})
                    fig=render_card_visual(player,team,pos,age,scores,sec_idx,overall,verdict,run_scores,run_abs,run_idx,
                                           final_index=(final_idx if not pd.isna(run_idx) and w_run>0 else None))
                    bio=BytesIO(); fig.savefig(bio,format="png",dpi=180,bbox_inches="tight"); plt.close(fig)
                    cards.append((str(player),bio.getvalue()))
            return pd.DataFrame(rows),cards

        res_df,cards=search_candidates(cz_df,fr_df,positions,sec_weights,metric_weights,
                                       min_minutes if min_minutes>0 else None,min_games if min_games>0 else None,
                                       league_name,cz_run_df,fr_run_df,w_run)
        st.session_state.update({"search_results":res_df,"search_cards":cards,"fr_df":fr_df,"cz_df":cz_df,
                                 "fr_run_df":fr_run_df,"cz_run_df":cz_run_df})

    res_df=st.session_state.get("search_results")
    if res_df is None or res_df.empty:
        st.info("Zatím žádné výsledky – nahraj soubory a klikni na *Spustit vyhledávání*.")
    else:
        st.success(f"Nalezeno kandidátů: {len(res_df)}"); st.dataframe(res_df,use_container_width=True)
        st.download_button("📥 Stáhnout CSV s kandidáty",
            data=res_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"kandidati_{st.session_state.get('search_league','liga')}.csv",mime="text/csv")
        zbuf=BytesIO()
        with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
            for name,png in (st.session_state.get("search_cards") or []):
                safe=str(name).replace("/","_").replace("\\","_"); zf.writestr(f"{safe}.png",png)
        st.download_button("🗂️ Stáhnout všechny karty (ZIP)",data=zbuf.getvalue(),
                           file_name=f"karty_{st.session_state.get('search_league','liga')}.zip",mime="application/zip")
        sel=st.selectbox("Zobraz kartu hráče", res_df["Hráč"].tolist())
        fr_df_cached=st.session_state.get("fr_df"); cz_df_cached=st.session_state.get("cz_df")
        fr_run_df_cached=st.session_state.get("fr_run_df"); cz_run_df_cached=st.session_state.get("cz_run_df")
        if sel and fr_df_cached is not None and cz_df_cached is not None:
            r=fr_df_cached.loc[fr_df_cached["Player"]==sel].iloc[0]
            pos_group=resolve_pos_group(str(r.get("Position",""))); rgx=POS_REGEX[pos_group]
            cz_pos=cz_df_cached[cz_df_cached["Position"].astype(str).str.contains(rgx,na=False,regex=True)]
            if not cz_pos.empty:
                cz_agg=cz_pos.groupby("Player").mean(numeric_only=True)
                scores,sec_idx,overall=compute_overall_for_row(r,cz_agg,sec_weights,metric_weights)
                run_idx=np.nan; run_scores=run_abs=None
                if cz_run_df_cached is not None and fr_run_df_cached is not None:
                    posc=get_pos_col(cz_run_df_cached); cz_run_pos=cz_run_df_cached[cz_run_df_cached[posc].astype(str).str.contains(rgx,na=False,regex=True)] if posc else pd.DataFrame()
                    pl=get_player_col(fr_run_df_cached); cand=fr_run_df_cached.loc[fr_run_df_cached[pl]==sel] if pl else pd.DataFrame()
                    if not cz_run_pos.empty and not cand.empty:
                        r_run=cand.iloc[0]; run_scores,run_abs,run_idx=run_index_for_row(r_run,cz_run_pos)
                final_index=(1.0- (st.session_state.get('w_run_val',0))) * overall + (st.session_state.get('w_run_val',0)) * run_idx if not pd.isna(run_idx) else None
                peer_avg=avg_peer_index(cz_agg,pos_group,sec_weights,metric_weights)
                base=final_index if (final_index is not None) else overall
                verdict=("ANO – potenciální posila do Slavie" if (not np.isnan(peer_avg) and not np.isnan(base) and base>=peer_avg)
                         else "NE – nedosahuje úrovně slávistických konkurentů")
                fig=render_card_visual(r.get("Player",""),r.get("Team",""),r.get("Position",""),r.get("Age","n/a"),
                                       scores,sec_idx,overall,verdict,run_scores,run_abs,run_idx,final_index)
                st.pyplot(fig)

# --- Tab 3: samostatná běžecká karta ---
with tab_runonly:
    st.subheader("Běžecká karta (samostatně – bez herních metrik)")
    cL,cR=st.columns(2)
    with cL: run_cz_bench=st.file_uploader("CZ běžecká data (xlsx) – benchmark",type=["xlsx"],key="run_cz_only")
    with cR: run_any=st.file_uploader("Běžecká data – libovolná liga (xlsx)",type=["xlsx"],key="run_any_only")
    if not run_cz_bench or not run_any:
        st.info("➡️ Nahraj **oba** soubory: CZ benchmark + běžecký export ligy/klubu."); st.stop()
    try:
        czb=pd.read_excel(run_cz_bench); anyb=pd.read_excel(run_any)
    except Exception as e:
        st.error(f"Chyba při načítání běžeckých souborů: {e}"); st.stop()
    czb=auto_fix_run_df(czb); anyb=auto_fix_run_df(anyb)
    plcol=get_player_col(anyb) or "Player"
    players_run=anyb[plcol].dropna().astype(str).sort_values().unique().tolist() if plcol in anyb.columns else []
    if not players_run: st.warning("Ve zvoleném běžeckém souboru není sloupec se jménem hráče."); st.stop()
    sel=st.selectbox("Vyber hráče (běžecký export)", players_run)
    row=anyb.loc[anyb[plcol].astype(str)==str(sel)].iloc[0]
    plc=get_player_col(czb) or "Player"; cz_tmp=czb.rename(columns={plc:"Player"}) if plc!="Player" and plc in czb.columns else czb
    cz_agg=cz_tmp.groupby("Player").mean(numeric_only=True)
    run_scores,run_abs,run_idx=compute_run_scores(row,cz_agg)
    with st.expander("Kontrola běžeckých dat",expanded=False):
        miss_cz=[lab for eng,lab in RUN if series_for_alias_run(cz_agg,eng) is None]
        miss_pl=[lab for eng,lab in RUN if pd.isna(value_with_alias_run(row,eng))]
        st.write(f"Chybějící metriky v CZ benchmarku: {', '.join(miss_cz) if miss_cz else '—'}")
        st.write(f"Chybějící metriky u hráče: {', '.join(miss_pl) if miss_pl else '—'}")
        present=sum([0 if pd.isna(run_scores[RUN_KEY].get(lab,np.nan)) else 1 for _,lab in RUN])
        st.write(f"Metrik započteno do Run indexu: {present}/{len(RUN)}")
        if present<=4: st.warning("Běžecké hodnocení je málo spolehlivé (≤ 4 metrik).")
    fig=render_run_card_visual(str(row.get(plcol,"")),str(row.get("Team","")),row.get("Position","—") if "Position" in row.index else "—",row.get("Age","n/a"),run_scores,run_abs,run_idx)
    st.pyplot(fig)
    b=io.BytesIO(); fig.savefig(b,format="png",dpi=180,bbox_inches="tight")
    st.download_button("📥 Stáhnout běžeckou kartu (PNG)",data=b.getvalue(),file_name=f"{row.get(plcol,'player')}_run.png",mime="image/png")

