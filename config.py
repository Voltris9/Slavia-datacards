# -*- coding: utf-8 -*-
import re

# ---------- Herní bloky ----------
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

# ---------- Alias herních metrik ----------
ALIASES = {
    "Cross accuracy, %": ["Accurate crosses, %","Cross accuracy, %"],
    "Progressive passes per 90": ["Progressive passes per 90","Progressive passes/90"],
    "Passes to final third per 90": ["Passes to final third per 90","Passes to final third/90"],
    "Dribbles per 90": ["Dribbles per 90","Dribbles/90"],
    "Progressive runs per 90": ["Progressive runs per 90","Progressive runs/90"],
    "Second assists per 90": ["Second assists per 90","Second assists/90"],
}

# ---------- Běžecké metriky ----------
RUN = [
    ("Total distance per 90", "Total distance /90"),
    ("High-intensity runs per 90", "High-intensity runs /90"),
    ("Sprints per 90", "Sprints /90"),
    ("Max speed (km/h)", "Max speed (km/h)"),
    ("Average speed (km/h)", "Average speed (km/h)"),
    ("Accelerations per 90", "Accelerations /90"),
    ("Decelerations per 90", "Decelerations /90"),
    ("High-speed distance per 90", "High-speed distance /90"),
]
RUN_KEY = "Běh"

ALIASES_RUN = {
    "Total distance per 90": ["Total distance per 90","Total distance/90","Distance per 90","Total distance (km) per 90","Distance P90"],
    "High-intensity runs per 90": ["High-intensity runs per 90","High intensity runs per 90","High intensity runs/90","HIR/90","HI Count P90"],
    "Sprints per 90": ["Sprints per 90","Sprints/90","Number of sprints per 90","Sprint Count P90"],
    "Max speed (km/h)": ["Max speed (km/h)","Top speed","Max velocity","Max speed","PSV-99","TOP 5 PSV-99"],
    "Average speed (km/h)": ["Average speed (km/h)","Avg speed","Average velocity","M/min P90"],
    "Accelerations per 90": ["Accelerations per 90","Accelerations/90","Accels per 90","High Acceleration Count P90 + Medium Acceleration Count P90"],
    "Decelerations per 90": ["Decelerations per 90","Decelerations/90","Decels per 90","High Deceleration Count P90 + Medium Deceleration Count P90"],
    "High-speed distance per 90": ["High-speed distance per 90","HS distance/90","High speed distance per 90","HSR Distance P90"],
}
CUSTOM_RUN_RENAME = {
    "Distance P90": "Total distance per 90",
    "HSR Distance P90": "High-speed distance per 90",
    "HI Count P90": "High-intensity runs per 90",
    "Sprint Count P90": "Sprints per 90",
    "High Acceleration Count P90": "High Acceleration Count P90",
    "Medium Acceleration Count P90": "Medium Acceleration Count P90",
    "High Deceleration Count P90": "High Deceleration Count P90",
    "Medium Deceleration Count P90": "Medium Deceleration Count P90",
    "PSV-99": "Max speed (km/h)",
    "Average speed (km/h)": "Average speed (km/h)",
}

# ---------- Pozice ----------
POS_REGEX = {
    "CB/DF": r"(CB|DF)", "RB": r"(RB)", "LB": r"(LB)",
    "WB/RWB/LWB": r"(WB|RWB|LWB)", "DM": r"(DM)", "CM": r"(CM)",
    "AM": r"(AM)", "RW": r"(RW)", "LW": r"(LW)", "CF/ST": r"(CF|ST|FW)",
}
def resolve_pos_group(pos_str: str) -> str:
    p = (pos_str or "").upper()
    if ("CB" in p) or ("DF" in p): return "CB/DF"
    if "RB" in p: return "RB"
    if "LB" in p: return "LB"
    if ("RWB" in p) or ("LWB" in p) or ("WB" in p): return "WB/RWB/LWB"
    if "DM" in p: return "DM"
    if "CM" in p: return "CM"
    if "AM" in p: return "AM"
    if "RW" in p: return "RW"
    if "LW" in p: return "LW"
    if ("CF" in p) or ("ST" in p) or ("FW" in p): return "CF/ST"
    return "CM"

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
