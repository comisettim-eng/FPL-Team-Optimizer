import time
from pathlib import Path
from typing import List

import numpy as np 
import pandas as pd 
import requests

BASE_URL = "https://fantasy.premierleague.com/api"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"

PLAYERS_CSV = PROC_DIR / "players_clean.csv"
TRAIN_CSV = PROC_DIR / "player_gw_training.csv"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_players(path: Path = PLAYERS_CSV) -> pd.DataFrame:
    """Load cleaned players dataset."""
    if not path.exists():
        raise FileNotFoundError(f"players_clean.csv not found at: {path}")
    df = pd.read_csv(path)
    #can filter here if needed
    return df 

def fetch_player_history(player_id: int) -> pd.DataFrame:
    """
    Fetch per-gameweek history for a single player from the FPL API.
    Returns a DataFrame with at least:
      round, total_points, minutes, goals_scored, assists,
      clean_sheets, was_home, opponent_team
    """
    url = f"{BASE_URL}/element-summary/{player_id}/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    history = data.get("history", [])
    if not history:
        return pd.DataFrame()

    df_hist = pd.DataFrame(history)

    # Keep only useful columns; others are dropped
    cols_keep = [
        "round",
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "was_home",
        "opponent_team",
        "yellow_cards",
        "red_cards",
        "penalties_missed",
        "own_goals",
        "bonus", #bonus points for being one of the best players of the game
        "saves",
    ]
    cols_available = [c for c in cols_keep if c in df_hist.columns]
    df_hist = df_hist[cols_available].copy()
    
    # OPTIONAL transfer columns -> add as zeros if missing
    for col in ["transfers_in_event", "transfers_out_event"]:
        if col not in df_hist.columns:
            df_hist[col] = 0  # default if API doesn’t provide it

    return df_hist

def fecth_fixtures() -> pd.DataFrame:
    """
    Fetch fixtures from FPL API and build a table with (gw, team_id, opponent_team, fixture_difficulty).
    """
    url = f"{BASE_URL}/fixtures/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    fixtures = resp.json()
    if not fixtures:
        return pd.DataFrame()

    fdf = pd.DataFrame(fixtures)

    #only care about schedules fixtures with an associated gameweek
    cols_needed = [
        "event",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
    ]
    cols_available = [c for c in cols_needed if c in fdf.columns]
    fdf = fdf[cols_available].dropna(subset=["event"])

    rows = []
    for _, r in fdf.iterrows():
        gw = int(r["event"])
        team_h = int(r["team_h"])
        team_a = int(r["team_a"])
        diff_h = r.get("team_h_difficulty", np.nan)
        diff_a = r.get("team_a_difficulty", np.nan)

        #row from home team's perspective:
        rows.append(
            {
                "gw": gw,
                "team_id": team_h,
                "opponent_team": team_a,
                "fixture_difficulty": diff_h,
                "was_home": 1,
            }
        )
        #row from away team's perspective
        rows.append(
            {
                "gw": gw,
                "team_id": team_a,
                "opponent_team": team_h,
                "fixture_difficulty": diff_a,
                "was_home": 0,
            }
        )
    fixtures_long = pd.DataFrame(rows)
    return fixtures_long

# ---------------------------------------------------------------------
# Build training table
# ---------------------------------------------------------------------

def build_training_table(max_players: int | None = None) -> pd.DataFrame:
    """Build player_gw_training table.

    Each row = (player, gameweek) with:
    - target_points
    - static info (team_id, position, price)
    - raw stats ( minutes, goals, assists, clean sheets, etc.)
    - lag feautures (prev_points, prev_minutes)
    - rolling features over past 3/5/8 GWs (only using *past* info)
    - fixture difficulty for that GW

    IMPORTANT:
      All features for GW t use data from GWs < t
      so there is no target leakage.
    """
    players = load_players()
    fixtures_df = fecth_fixtures()

    if max_players is not None:
        players = players.head(max_players).copy()

    all_rows: List[pd.DataFrame] = []

    for idx, row in players.iterrows():
        pid = int(row["player_id"])

        try:
            hist_df = fetch_player_history(pid)
        except requests.HTTPError as e:
            print(f"[WARN] Could not fetch history for player {pid}: {e}")
            continue

        if hist_df.empty:
            continue

        #add player-level info (static)
        hist_df["player_id"] = pid
        hist_df["team_id"] = row.get("team_id", np.nan)
        hist_df["position"] = row.get("position", np.nan)
        hist_df["price"] = row.get("price", np.nan)
        hist_df["status"] = row.get("status","a")

        all_rows.append(hist_df)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} players...")

    if not all_rows:
        raise RuntimeError("No player history could be fetched. Check API or inputs")

    df = pd.concat(all_rows, ignore_index=True)

    #rename round to gw 
    df = df.rename(columns={"round": "gw"})

    #sort for consistant rolling calculations
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)

    #target variable
    df["target_points"] = df["total_points"].astype(float)

    #merge fixture difficulty
    if not fixtures_df.empty:
        df = df.merge(
            fixtures_df,
            on=["gw", "team_id", "opponent_team"],
            how="left",
            suffixes=("", "_fix"),

        )
        #fill missing difficulty with a neutral value (median or 3)
        if "fixture_difficulty" in df.columns:
            median_diff = df["fixture_difficulty"].median()
            df["fixture_difficulty"] = df["fixture_difficulty"].fillna(
                median_diff if not np.isnan(median_diff) else 3.0
            )
        else:
            df["fixture_difficulty"] = 3.0
        # if was_home not present from fixtures (e.g. missing merge), default to 0
        if "was_home" not in df.columns:
            df["was_home"] = 0
        df["was_home"] = df["was_home"].fillna(0).astype(int)
    else:
        df["fixture_difficulty"] = 3.0 #neutral fall back
        df["was_home"] = df["was_home"].fillna(0).astype(int)

    # Groupby for lag + rolling features
    g = df.groupby("player_id", group_keys=False)

    # Lag features (previous GW)
    df["prev_points"] = g["total_points"].shift(1)
    df["prev_minutes"] = g["minutes"].shift(1)

    # -----------------------------------------------------------------
    #  lag *per-match stats* so they refer to PREVIOUS GW
    # -----------------------------------------------------------------
    lag_stat_cols = [
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "yellow_cards",
        "red_cards",
        "penalties_missed",
        "own_goals",
        "bonus",
        "saves",
        "transfers_in_event",
        "transfers_out_event",
    ]
    for col in lag_stat_cols:
        if col in df.columns:
            df[col] = g[col].shift(1)

    # -----------------------------------------------------------------
    # Rolling windows for points/minutes (3, 5, 8)
    # -----------------------------------------------------------------
    for w in [3, 5, 8]:
        df[f"roll_pts_{w}"] = (
            g["total_points"].shift(1).rolling(window=w, min_periods=1).sum()
        )
        df[f"roll_min_{w}"] = (
            g["minutes"].shift(1).rolling(window=w, min_periods=1).sum()
        )
        # points per 90 over this window
        minutes_col = df[f"roll_min_{w}"].replace(0, np.nan)
        df[f"roll_pts_per90_{w}"] = (
            df[f"roll_pts_{w}"] / (minutes_col / 90.0)
        )
        df[f"roll_pts_per90_{w}"] = df[f"roll_pts_per90_{w}"].fillna(0.0)

    # keep existing 3-GW rolling for goals/assists etc.
    df["roll_goals_3"] = (
        g["goals_scored"].shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["roll_assists_3"] = (
        g["assists"].shift(1).rolling(window=3, min_periods=1).sum()
    )

    df["roll_yellow_3"] = (
        g["yellow_cards"].shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["roll_red_3"] = (
        g["red_cards"].shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["roll_pen_miss_3"] = (
        g["penalties_missed"].shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["roll_og_3"] = (
        g["own_goals"].shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["roll_bonus_3"] = (
        g["bonus"].shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["roll_saves_3"] = (
        g["saves"].shift(1).rolling(window=3, min_periods=1).sum()
    )

    # -----------------------------------------------------------------
    # "Nailedness" features over last 5 GWs 
    # -----------------------------------------------------------------
    minutes_last5 = g["minutes"].shift(1).rolling(window=5, min_periods=1).sum()
    games_last5 = (
        (g["minutes"].shift(1) > 0).astype(float).rolling(window=5, min_periods=1).sum()
    )

    df["games_played_5"] = games_last5
    df["minutes_share_5"] = minutes_last5 / (5 * 90.0)

    # Replace NaNs (for first GWs of each player) with 0
    lag_roll_cols = [
        "prev_points",
        "prev_minutes",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "yellow_cards",
        "red_cards",
        "penalties_missed",
        "own_goals",
        "bonus",
        "saves",
        "transfers_in_event",
        "transfers_out_event",
        "roll_pts_3",
        "roll_min_3",
        "roll_pts_5",
        "roll_min_5",
        "roll_pts_8",
        "roll_min_8",
        "roll_goals_3",
        "roll_assists_3",
        "roll_yellow_3",
        "roll_red_3",
        "roll_pen_miss_3",
        "roll_og_3",
        "roll_bonus_3",
        "roll_saves_3",
        "roll_pts_per90_3",
        "roll_pts_per90_5",
        "roll_pts_per90_8",
        "games_played_5",
        "minutes_share_5",
    ]
    existing_lag_roll = [c for c in lag_roll_cols if c in df.columns]
    df[existing_lag_roll] = df[existing_lag_roll].fillna(0.0)

    #ensure types:
    float_cols = [
        "target_points",
        "price",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "yellow_cards",
        "red_cards",
        "penalties_missed",
        "own_goals",
        "bonus",
        "saves",
        "prev_points",
        "prev_minutes",
        "roll_pts_3",
        "roll_min_3",
        "roll_pts_5",
        "roll_min_5",
        "roll_pts_8",
        "roll_min_8",
        "roll_pts_per90_3",
        "roll_pts_per90_5",
        "roll_pts_per90_8",
        "roll_goals_3",
        "roll_assists_3",
        "roll_yellow_3",
        "roll_red_3",
        "roll_pen_miss_3",
        "roll_og_3",
        "roll_bonus_3",
        "roll_saves_3",
        "games_played_5",
        "minutes_share_5",
        "fixture_difficulty",
        "transfers_in_event",
        "transfers_out_event",
    ]
    existing_float_cols = [c for c in float_cols if c in df.columns]
    df[existing_float_cols] = df[existing_float_cols].astype(float)

    df["was_home"] = df["was_home"].astype(int)
    df["opponent_team"] = df["opponent_team"].astype(int)
    df["team_id"] = df["team_id"].astype(int, errors="ignore")

    return df

def save_training_table(df:pd.DataFrame, path: Path = TRAIN_CSV) -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved training table with {len(df)} rows to {path}")
    return path

def main():
    print("Building player_gw_training table from FPL API...")
    df_train = build_training_table(max_players=None) #max_player=None for full data - set = 200 for testing
    save_training_table(df_train)

if __name__=="__main__":
    main()