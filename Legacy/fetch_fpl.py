import json
from pathlib import Path
import requests
import pandas as pd 
import numpy as np 

BASE_URL = "https://fantasy.premierleague.com/api"
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

#PART 1: IMPORTING AND CLEANING THE DATA FOR FPL SINGLE GAMEWEEK OPTIMIZER
def fetch_bootstrap_static() -> dict:
    """fetch main FPL dataset (players, teams, positions, etc)"""
    url = f"{BASE_URL}/bootstrap-static/"
    resp = requests.get(url, timeout =10)
    resp.raise_for_status()
    return resp.json()

def save_raw_snapshot(data: dict, name: str = "bootstrap_static.json") -> Path:
    """Save raw JSON snapshot for reproducability"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

POSITION_MAP= {1: "GK", 2:"DEF", 3: "MID", 4: "FWD"}

def build_players_df(data: dict) -> pd.DataFrame:
    """Create a clean player dataframe with all variable needed for the single gameweek optimizer"""
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])[["id","name"]].rename(
        columns={"id": "team_id", "name":"team_name"}
        )

    df = players [
        [
            "id",
            "first_name",
            "second_name",
            "team",
            "element_type",
            "now_cost",
            "total_points",
            "minutes",
            "form",
            "points_per_game",
            "selected_by_percent",
            "status", #a: player is available, d: 25-75% chance of playing, i = injured, s = suspended, n = unavailable (trasnfered out of PL), u=unavailable (not match fit)
        ]
    ].copy()

    df.rename(
        columns={
            "id":"player_id",
            "first_name":"first_name",
            "second_name":"last_name",
            "team":"team_id",
            "now_cost":"price_tenths",
        },
        inplace=True
    )
    #human-readable fields
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["price_tenths"] / 10.0 #e.g 75 -> 7.5m
    df["name"] = df["first_name"].str.strip() + " " + df["last_name"].str.strip()

    #drop redundant columns
    df.drop(columns=["first_name","last_name"], inplace=True, errors="ignore")
    df.drop(columns=["price_tenths"], inplace=True, errors="ignore")
    
    #Convert numeric columns safely

    numeric_cols = ["total_points", "minutes", "form", "points_per_game", "selected_by_percent", "price"]
    df[numeric_cols] = df[numeric_cols].round(3)
    for col in numeric_cols:
        df[col]=pd.to_numeric(df[col], errors="coerce").fillna(0)

    #key metics for the project
    df["points_per_million"] = df["total_points"] / df["price"].replace(0, np.nan)
    df["points_per_90"] = df["total_points"] / (df["minutes"].replace(0, np.nan) / 90)

    #Round metrics for readability
    round_cols = [
        "form",
        "points_per_game",
        "selected_by_percent",
        "price",
        "points_per_million",
        "points_per_90",
    ]
    df[round_cols] = df[round_cols].round(2)

    #join team names
    df = df.merge (teams, on="team_id", how ="left")

    #re-order columns to be optimizer freindly
    cols = [
        "player_id",
        "name",
        "team_id",
        "team_name",
        "position",
        "element_type",
        "status",
        "price",
        "total_points",
        "minutes",
        "form",
        "points_per_game",
        "points_per_million",
        "points_per_90",
        "selected_by_percent",
    ]
    df = df.reindex(columns=cols)
    
    print("FINAL column order", list(df.columns)) #debug check
    print("df columns:", df.columns)
    print("teams columns:", teams.columns)

    return df


def save_players_csv(df: pd.DataFrame, name: str = "players_clean.csv") -> Path:
    """Saved clean player DataSet"""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / name
    df.to_csv(path, index=False)
    print ("Saving to:", path.resolve())
    return path 


def main():
    data = fetch_bootstrap_static()
    save_raw_snapshot(data)
    players_df = build_players_df(data)
    save_players_csv(players_df)
    print(f"Saved{len(players_df)} players to data/processed/players_clean.csv")

if __name__ == "__main__":
    main()
