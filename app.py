# app.py — Slavia datacards (zkrácená a přehlednější verze)
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

def _norm(s, mode="txt"):
    """Normalizace pro jména, týmy, národnosti."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+"," ",s).strip().lower()
    if mode=="team":
        s = re.sub(r"\b(fk|fc|sc|ac|cf|afc|sv|us|cd|ud|bk|sk|ks|ucl|ii|b)\b"," ",s)
    return s

def color_for(v):
    if pd.isna(v): return "lightgrey"
    v=float(v)
    return "#FF0000" if v<=20 else "#FF8C00" if v<=40 else "#FFD700" if v<=60 else "#90EE90" if v<=80 else "#006400"

def _best_col(df, names): return next((c for c in names if c in df.columns), None)
def get_player_col(df): return _best_col(df,["Player","Name","Short Name"])
def get_team_col(df):   return _best_col(df,["Team","Club"])
def get_pos_col(df):    return _best_col(df,["Position","Pos","Role","Primary position"])
def get_age_col(df):    return _best_col(df,["Age"])
def get_nat_col(df):    return _best_col(df,["Nationality","Nation","Country","Citizenship"])

def normalize_core_cols(df):
    if df is None or df.empty: return df
    m={src:dst for src,dst in [(get_player_col(df),"Player"),(get_team_col(df),"Team"),(get_pos_col(df),"Position")] if src and src!=dst}
    return df.rename(columns=m) if m else df

# ---------- Alias dictionary ----------
ALIASES = {
    # herní
    "Cross accuracy, %":["Accurate crosses, %"],
    "Progressive passes per 90":["Progressive passes/90"],
    "Passes to final third per 90":["Passes to final third/90"],
    "Dribbles per 90":["Dribbles/90"],
    "Progressive runs per 90":["Progressive runs/90"],
    "Second assists per 90":["Second assists/90"],
    # běžecké
    "Total distance per 90":["Total distance/90","Distance P90"],
    "High-intensity runs per 90":["High intensity runs/90","HIR/90"],
    "Sprints per 90":["Sprints/90","Sprint Count P90"],
    "Max speed (km/h)":["Top speed","PSV-99"],
    "Average speed (km/h)":["Avg speed","M/min P90"],
    "Accelerations per 90":["Accels/90","High Acceleration Count P90","Medium Acceleration Count P90"],
    "Decelerations per 90":["Decels/90","High Deceleration Count P90","Medium Deceleration Count P90"],
    "High-speed distance per 90":["HS distance/90","HSR Distance P90"],
}

def get_val(row, key):
    if key in row.index: return row[key]
    for c in ALIASES.get(key,[]): 
        if c in row.index: return row[c]
    return np.nan

def get_series(df, key):
    if key in df.columns: return df[key]
    for c in ALIASES.get(key,[]):
        if c in df.columns: return df[c]
    return None

def norm_metric(pop, key, val):
    s=get_series(pop,key)
    if s is None or pd.isna(val): return np.nan
    s=pd.to_numeric(s,errors="coerce").dropna(); v=pd.to_numeric(pd.Series([val]),errors="coerce").iloc[0]
    if s.empty or pd.isna(v): return np.nan
    mn,mx=s.min(),s.max()
    return 50 if mx==mn else float(np.clip((v-mn)/(mx-mn)*100,0,100))

def draw_box(ax, x, y, label, val, abs_val=None, w=0.18, h=0.034, fs=9):
    ax.add_patch(Rectangle((x,y-h/2),w,h,color=color_for(val),alpha=0.85,lw=0))
    if abs_val is None:
        text=f"{label}: {'n/a' if pd.isna(val) else str(int(round(val)))+'%'}"
    else:
        ta="n/a" if pd.isna(abs_val) else (f"{abs_val:.2f}" if isinstance(abs_val,(int,float,np.number)) else str(abs_val))
        tp="n/a" if pd.isna(val) else f"{int(round(val))}%"
        text=f"{label}: {ta} ({tp})"
    ax.text(x+0.005,y,f"{text}",fontsize=fs,va="center",ha="left")

# ---------- Role5 mapování ----------
WYS_TO_ROLE = {
    "RCB":"CB","LCB":"CB","CB":"CB","RCB3":"CB","LCB3":"CB",
    "RB":"RB","LB":"RB","RB5":"RB","LB5":"RB","RWB":"RB","LWB":"RB",
    "DMF":"CM","CM":"CM","AMF":"CM","RDMF":"CM","LDMF":"CM","RCMF":"CM","LCMF":"CM",
    "RW":"RW","LW":"RW","RAMF":"RW","LAMF":"RW","WINGER":"RW","AMFR":"RW","AMFL":"RW","LWF":"RW","RWF":"RW",
    "CF":"CF","ST":"CF","FW":"CF"
}
def role5_from_pos_text(pos):
    if not pos: return ""
    first = str(pos).split(",")[0].strip().upper()
    return WYS_TO_ROLE.get(first,"")

# ---------- Metriky ----------
SECTIONS = {
    "Kreativita":["Cross accuracy, %","Progressive passes per 90","Passes to final third per 90"],
    "Driblink & individuální akce":["Dribbles per 90","Progressive runs per 90"],
    "Finální fáze":["Second assists per 90"]
}
RUN_METRICS = [
    "Total distance per 90","High-intensity runs per 90","Sprints per 90",
    "Max speed (km/h)","Average speed (km/h)",
    "Accelerations per 90","Decelerations per 90","High-speed distance per 90"
]

def compute_role_index(player_row, pop_df):
    sec_scores={}; all_vals=[]
    for sec,metrics in SECTIONS.items():
        sec_scores[sec]={}
        for m in metrics:
            val=get_val(player_row,m)
            score=norm_metric(pop_df,m,val)
            sec_scores[sec][m]=score
            if not pd.isna(score): all_vals.append(score)
    role_idx=np.nanmean(all_vals) if all_vals else np.nan
    return sec_scores,role_idx

def compute_run_index(player_row, pop_df):
    run_scores={}; all_vals=[]
    for m in RUN_METRICS:
        val=get_val(player_row,m)
        score=norm_metric(pop_df,m,val)
        run_scores[m]=score
        if not pd.isna(score): all_vals.append(score)
    run_idx=np.nanmean(all_vals) if all_vals else np.nan
    return run_scores,run_idx

# ---------- Render karty ----------
def render_card(player, team, pos, age, sec_scores, run_scores, role_idx, run_idx, final_idx, verdict):
    fig,ax=plt.subplots(figsize=(18,12)); ax.axis("off")
    ax.text(0.02,0.96,f"{player} (věk {age})",fontsize=20,fontweight="bold",va="top")
    ax.text(0.02,0.93,f"Klub: {team}   Pozice: {pos}",fontsize=13,va="top")

    # Herní sekce
    y0=0.88
    for title,sc in sec_scores.items():
        ax.text(0.02,y0,title,fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
        for i,(lab,val) in enumerate(sc.items()):
            x=L if i%2==0 else R
            draw_box(ax,x,y,lab,val)
            if i%2==1: y-=0.038
        y0=y-0.025

    # Běh
    if run_scores and not pd.isna(run_idx):
        ax.text(0.02,y0,"Běžecká data",fontsize=15,fontweight="bold",va="top"); y=y0-0.04; L,R=0.04,0.26
        for i,(lab,val) in enumerate(run_scores.items()):
            x=L if i%2==0 else R
            draw_box(ax,x,y,lab,val,val)
            if i%2==1: y-=0.038
        y0=y-0.025

    # Souhrn
    ax.text(0.55,0.9,"Souhrnné indexy",fontsize=16,fontweight="bold",va="top"); y=0.85
    draw_box(ax,0.55,y,"Role-index",role_idx); y-=0.075
    if not pd.isna(run_idx): 
        draw_box(ax,0.55,y,"Běžecký index",run_idx); y-=0.075
    draw_box(ax,0.55,y,"Finální index",final_idx)
    ax.text(0.74,0.055,f"Verdikt: {verdict}",fontsize=12,ha="center",va="center")
    return fig

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Nastavení")
w_run=st.sidebar.slider("Váha běžeckého indexu (%)",0,50,20,5)/100

# ---------- Tabs ----------
tab_card,tab_search=st.tabs(["Karta hráče","Vyhledávání hráčů"])

with tab_card:
    st.subheader("🎴 Karta hráče")
    uploaded_files = st.file_uploader("Nahraj soubory (CZ+HOL herní i běžecké)",type=["xlsx"],accept_multiple_files=True)
    if uploaded_files:
        dfs={f.name:normalize_core_cols(load_xlsx(f.read())) for f in uploaded_files}
        all_df=pd.concat([df for df in dfs.values() if df is not None],ignore_index=True)
        all_df["Role5"]=all_df["Position"].map(role5_from_pos_text)

        player_name=st.text_input("Zadej jméno hráče")
        if player_name:
            row=all_df[all_df["Player"].str.contains(player_name,case=False,na=False)]
            if not row.empty:
                r=row.iloc[0]
                role=r["Role5"]; team=r.get("Team",""); age=r.get("Age","")
                bench=all_df[all_df["Role5"]==role]

                sec_scores,role_idx=compute_role_index(r,bench)
                run_scores,run_idx=compute_run_index(r,bench)

                final_idx=(1-w_run)*role_idx + w_run*(run_idx if not pd.isna(run_idx) else role_idx)

                slavia=bench[bench["Team"].str.contains("Slavia",case=False,na=False)]
                th=np.nanmean([compute_role_index(x,bench)[1] for _,x in slavia.iterrows()])
                verdict="ANO" if final_idx>=th else "NE"

                fig=render_card(r["Player"],team,role,age,sec_scores,run_scores,role_idx,run_idx,final_idx,verdict)
                st.pyplot(fig)
            else:
                st.warning("Hráč nenalezen.")

with tab_search:
    st.subheader("🔍 Vyhledávání hráčů")
    uploaded_files = st.file_uploader("Nahraj soubory (CZ+HOL herní i běžecké)",type=["xlsx"],accept_multiple_files=True,key="search")
    if uploaded_files:
        dfs={f.name:normalize_core_cols(load_xlsx(f.read())) for f in uploaded_files}
        all_df=pd.concat([df for df in dfs.values() if df is not None],ignore_index=True)
        all_df["Role5"]=all_df["Position"].map(role5_from_pos_text)

        thresholds={}
        for role in all_df["Role5"].dropna().unique():
            bench=all_df[all_df["Role5"]==role]
            slavia=bench[bench["Team"].str.contains("Slavia",case=False,na=False)]
            if not slavia.empty:
                thresholds[role]=np.nanmean([compute_role_index(x,bench)[1] for _,x in slavia.iterrows()])

        candidates=[]
        for _,r in all_df.iterrows():
            role=r["Role5"]
            if not role or role not in thresholds: continue
            bench=all_df[all_df["Role5"]==role]
            sec_scores,role_idx=compute_role_index(r,bench)
            run_scores,run_idx=compute_run_index(r,bench)
            final_idx=(1-w_run)*role_idx + w_run*(run_idx if not pd.isna(run_idx) else role_idx)
            verdict="ANO" if final_idx>=thresholds[role] else "NE"
            if verdict=="ANO":
                candidates.append({
                    "Player":r["Player"],"Team":r.get("Team",""),"Role":role,
                    "Role index":round(role_idx,1) if role_idx else np.nan,
                    "Run index":round(run_idx,1) if run_idx else np.nan,
                    "Final index":round(final_idx,1) if final_idx else np.nan
                })

        if candidates:
            df=pd.DataFrame(candidates).sort_values("Final index",ascending=False)
            st.dataframe(df,use_container_width=True)
        else:
            st.info("Nenalezeni žádní vhodní hráči.")

