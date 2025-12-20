# fpl_ml_pipeline.py
"""
Unified end-to-end FPL ML pipeline

Steps:
1. Ingest data:
   - Historical seasons 2016-17..2024-25 from vaastav
   - Current season 2025-26 from official FPL API
   -> data/processed/<season>_players_clean.csv
   -> data/processed/<season>_gw_history.csv
   -> data/processed/players_clean_all_seasons.csv
   -> data/processed/gw_history_all_seasons.csv

2. Feature engineering:
   -> data/processed/player_gw_features.csv
   -> data/processed/team_gw_features.csv
   -> data/processed/fixture_gw_features.csv
   -> data/processed/player_gw_training.csv   (features only, NO target)

3. Train ML models (RF / LGBM / XGB) to predict next-GW points:
   -> models/ml_models/<model>_points_model_seasons.pkl
   -> results/ml_backtests/model_comparison_<season>_gw1_<N>.csv
   -> results/ml_backtests/predictions_<season>_gw1_<N>_<model>.csv
   -> data/processed/player_gw_predictions_<season>_gw1_<N>_<model>.csv (LP-ready)

4. LP-based ML squad optimization (season simulation with 1 transfer per GW):
   -> results/ml_season_solutions/<model>/<model>_team_GWxx.csv
   -> results/ml_season_solutions/<model>/<model>_starting_xi_GWxx.csv

5. Compare model squads vs real FPL points and FPL average:
   -> results/ml_backtests/model_season_scores_api_gw2_<N>.csv
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests
import pulp as pl

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib


# ======================================================================
# GLOBAL CONFIG
# ======================================================================

FPL_GITHUB_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)
FPL_API_BASE = "https://fantasy.premierleague.com/api"

SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

CURRENT_SEASON = "2025-26"

# Assume script lives at project root (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "ml_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_DIR = PROJECT_ROOT / "results" / "ml_backtests"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
BASE_SOLUTIONS_DIR = PROJECT_ROOT / "results" / "ml_season_solutions"
BASE_SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Formation-like limits on the 15-man squad
FORMATION_LIMITS = {
    "GK": (2, 2),
    "DEF": (5, 5),
    "MID": (5, 5),
    "FWD": (3, 3),
}

# Model keys
MODELS = ["rf", "lgbm", "xgb"]
MODEL_NAME = {
    "rf": "RandomForest",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
}

# Training / validation config
VAL_SEASON = CURRENT_SEASON
VAL_MAX_GW = 9

# Prediction window used by optimizer
PREDICTION_START_GW = 2
PREDICTION_END_GW = VAL_MAX_GW

# Template for LP-ready predictions
PREDICTIONS_TEMPLATE = "player_gw_predictions_{season}_gw1_{max_gw}_{model}.csv"

# Extra stat columns to keep in GW history where available
EXTRA_STAT_COLS = [
    # core attacking / defensive stats
    "assists",
    "goals_scored",
    "goals_conceded",
    "clean_sheets",
    "own_goals",
    "penalties_conceded",
    "penalties_missed",
    "penalties_saved",
    "yellow_cards",
    "red_cards",
    "saves",
    "winning_goals",

    # underlying stats / indexes
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "ea_index",

    # expected stats (newer seasons)
    "xP",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",

    # usage / involvement
    "starts",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",

    # passing / possession / defending detail
    "attempted_passes",
    "completed_passes",
    "key_passes",
    "big_chances_created",
    "big_chances_missed",
    "offside",
    "open_play_crosses",
    "dribbles",
    "fouls",
    "tackled",
    "tackles",
    "clearances_blocks_interceptions",
    "recoveries",
    "errors_leading_to_goal",
    "errors_leading_to_goal_attempt",
    "target_missed",
    "defensive_contribution",

    # fixture / context columns
    "fixture",
    "kickoff_time",
    "kickoff_time_formatted",
    "team_a_score",
    "team_h_score",
    "was_home",
    "opponent_team",

    # misc
    "loaned_in",
    "loaned_out",
    "value",
    "in_dreamteam",
    "modified",
]


# ======================================================================
# SMALL HELPERS
# ======================================================================

def standardize_team_id(df: pd.DataFrame, col: str = "team_id") -> pd.DataFrame:
    """Force team_id to a consistent nullable int dtype for safe merges."""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def read_csv_robust(url: str, **kwargs) -> pd.DataFrame:
    """
    Robust CSV reader for GitHub raw files.

    - First try normal pandas.read_csv
    - On UnicodeDecodeError, retry with encoding='latin1' and engine='python'
    """
    try:
        return pd.read_csv(url, **kwargs)
    except UnicodeDecodeError:
        kwargs.setdefault("encoding", "latin1")
        kwargs.setdefault("engine", "python")
        return pd.read_csv(url, **kwargs)


# ======================================================================
# 1) DATA INGESTION: HISTORICAL (VAASTAV) + CURRENT SEASON (FPL API)
# ======================================================================

def fetch_players_raw_for_season(season: str) -> pd.DataFrame:
    url = f"{FPL_GITHUB_BASE}/{season}/players_raw.csv"
    df = read_csv_robust(url)
    return df


def fetch_gws_for_season(season: str, max_gw: int = 60) -> pd.DataFrame:
    """
    Fetch all gwX.csv files for a season and stack them (vaastav).
    """
    all_gws = []
    for gw in range(1, max_gw + 1):
        url = f"{FPL_GITHUB_BASE}/{season}/gws/gw{gw}.csv"
        try:
            df = read_csv_robust(
                url,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            print(f"[{season}] Stopped at gw{gw}: {e}")
            break

        df["season"] = season
        df["gameweek"] = gw
        all_gws.append(df)
        print(f"[{season}] Loaded gw{gw} with {len(df)} rows")

    if not all_gws:
        raise RuntimeError(f"No gwX.csv files found for season {season}")

    merged = pd.concat(all_gws, ignore_index=True, sort=False)
    return merged


def fetch_master_team_list() -> pd.DataFrame:
    url = f"{FPL_GITHUB_BASE}/master_team_list.csv"
    df = read_csv_robust(url)
    df.rename(columns={"team": "team_id"}, inplace=True)
    df = standardize_team_id(df, "team_id")
    return df


def save_raw_snapshot_csv(df: pd.DataFrame, season: str, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved raw snapshot to: {path}")
    return path


def fetch_bootstrap_static() -> dict:
    """Fetch main FPL dataset (players, teams, etc.) for the current live season."""
    url = f"{FPL_API_BASE}/bootstrap-static/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_players_raw_from_bootstrap(data: dict) -> pd.DataFrame:
    """
    Build a 'players_raw'-like DataFrame from bootstrap-static for CURRENT_SEASON
    with the same columns expected by build_players_df.
    """
    players = pd.DataFrame(data["elements"])
    cols = [
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
        "status",
    ]
    return players.reindex(columns=cols).copy()


def fetch_gws_for_current_season(bootstrap_data: dict, max_gw: int = 60) -> pd.DataFrame:
    """
    Fetch all gameweeks for CURRENT_SEASON from the official FPL API.
    """
    players_static = pd.DataFrame(bootstrap_data["elements"]).copy()
    players_static["name"] = (
        players_static["first_name"].fillna("").astype(str).str.strip()
        + " "
        + players_static["second_name"].fillna("").astype(str).str.strip()
    ).str.strip()
    players_static.rename(
        columns={
            "id": "player_id",
            "team": "team_id",
            "now_cost": "value",
        },
        inplace=True,
    )

    all_gws = []

    for gw in range(1, max_gw + 1):
        url = f"{FPL_API_BASE}/event/{gw}/live/"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"[{CURRENT_SEASON}] Stopped at GW{gw}: HTTP {resp.status_code}")
            break

        data = resp.json()
        if "elements" not in data or not data["elements"]:
            print(f"[{CURRENT_SEASON}] Stopped at GW{gw}: no elements in response")
            break

        elems = pd.DataFrame(data["elements"])
        stats_df = pd.json_normalize(elems["stats"])
        stats_df["player_id"] = elems["id"]

        stat_cols_from_api = [
            "minutes",
            "total_points",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_missed",
            "penalties_saved",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "expected_goals_conceded",
            "starts",
        ]

        subset_cols = [c for c in stat_cols_from_api if c in stats_df.columns]
        gw_df = stats_df[subset_cols + ["player_id"]].copy()
        gw_df["gameweek"] = gw

        gw_df = gw_df.merge(
            players_static[["player_id", "name", "team_id", "value"]],
            on="player_id",
            how="left",
        )

        gw_df.rename(
            columns={
                "player_id": "element",
                "team_id": "team",
                "value": "value",
            },
            inplace=True,
        )

        gw_df["season"] = CURRENT_SEASON
        all_gws.append(gw_df)
        print(f"[{CURRENT_SEASON}] Loaded GW{gw} with {len(gw_df)} rows")

    if not all_gws:
        raise RuntimeError(f"No gameweeks found for {CURRENT_SEASON} via FPL API")

    merged = pd.concat(all_gws, ignore_index=True, sort=False)
    return merged


def build_players_df(
    players: pd.DataFrame, teams_lookup: pd.DataFrame, season: str
) -> pd.DataFrame:
    cols_to_keep = [
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
        "status",
    ]
    df = players.reindex(columns=cols_to_keep).copy()

    df.rename(
        columns={
            "id": "player_id",
            "first_name": "first_name",
            "second_name": "last_name",
            "team": "team_id",
            "now_cost": "price_tenths",
        },
        inplace=True,
    )

    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["price_tenths"] / 10.0
    df["name"] = (
        df["first_name"].fillna("").astype(str).str.strip()
        + " "
        + df["last_name"].fillna("").astype(str).str.strip()
    ).str.strip()

    df.drop(columns=["first_name", "last_name", "price_tenths"], inplace=True, errors="ignore")

    df["season"] = season

    numeric_cols = [
        "total_points",
        "minutes",
        "form",
        "points_per_game",
        "selected_by_percent",
        "price",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df[numeric_cols] = df[numeric_cols].round(3)

    price_nonzero = df["price"].replace(0, np.nan)
    minutes_nonzero_90 = df["minutes"].replace(0, np.nan) / 90.0

    df["points_per_million"] = df["total_points"] / price_nonzero
    df["points_per_90"] = df["total_points"] / minutes_nonzero_90

    round_cols = [
        "form",
        "points_per_game",
        "selected_by_percent",
        "price",
        "points_per_million",
        "points_per_90",
    ]
    for col in round_cols:
        df[col] = df[col].round(2)

    df = standardize_team_id(df, "team_id")
    season_teams = teams_lookup[teams_lookup["season"] == season].copy()
    season_teams = standardize_team_id(season_teams, "team_id")

    df = df.merge(
        season_teams[["team_id", "team_name"]],
        on="team_id",
        how="left",
    )

    cols_final = [
        "season",
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
    df = df.reindex(columns=cols_final)

    print(f"[{season}] FINAL player column order: {list(df.columns)}")
    return df


def build_gw_history_df(
    gw_raw: pd.DataFrame,
    players_raw: pd.DataFrame,
    teams_lookup: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """
    Build per-player per-GW history, preserving as many useful
    per-match stats as possible across all seasons.
    """
    df = gw_raw.copy()

    if "element" in df.columns:
        df.rename(columns={"element": "player_id"}, inplace=True)
    if "team" in df.columns and "team_id" not in df.columns:
        df.rename(columns={"team": "team_id"}, inplace=True)

    if "GW" in df.columns:
        df.rename(columns={"GW": "gameweek"}, inplace=True)
    elif "round" in df.columns and "gameweek" not in df.columns:
        df.rename(columns={"round": "gameweek"}, inplace=True)

    if "value" in df.columns:
        df["price"] = df["value"] / 10.0
    elif "price" not in df.columns:
        df["price"] = np.nan

    meta = players_raw[["id", "first_name", "second_name", "element_type", "team"]].copy()
    meta.rename(columns={"id": "player_id", "team": "team_id"}, inplace=True)
    meta["player_id"] = pd.to_numeric(meta["player_id"], errors="coerce").astype("Int64")

    meta["name"] = (
        meta["first_name"].fillna("").astype(str).str.strip()
        + " "
        + meta["second_name"].fillna("").astype(str).str.strip()
    ).str.strip()
    meta["position"] = meta["element_type"].map(POSITION_MAP)
    meta.drop(columns=["first_name", "second_name"], inplace=True)

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.merge(meta, on="player_id", how="left", suffixes=("", "_from_players"))

    if "team_id_from_players" in df.columns:
        df["team_id"] = df["team_id"].fillna(df["team_id_from_players"])
        df.drop(columns=["team_id_from_players"], inplace=True)

    df["position"] = df["position"].replace({"GKP": "GK"})

    df = standardize_team_id(df, "team_id")
    season_teams = teams_lookup[teams_lookup["season"] == season].copy()
    season_teams = standardize_team_id(season_teams, "team_id")

    df = df.merge(
        season_teams[["team_id", "team_name"]],
        on="team_id",
        how="left",
    )

    df["season"] = season

    cols_base = [
        "season",
        "gameweek",
        "player_id",
        "name",
        "team_id",
        "team_name",
        "position",
        "element_type",
        "price",
        "minutes",
        "total_points",
    ]

    extra_present = [c for c in EXTRA_STAT_COLS if c in df.columns]

    cols_final = []
    for c in cols_base + extra_present:
        if c not in cols_final:
            cols_final.append(c)

    df = df.reindex(columns=cols_final).copy()

    print(f"[{season}] GW history columns: {list(df.columns)}")
    return df


def save_players_csv(df: pd.DataFrame, season: str, name: str = "players_clean.csv") -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = PROC_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved cleaned players to: {path}")
    return path


def save_gw_history_csv(df: pd.DataFrame, season: str, name: str = "gw_history.csv") -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = PROC_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved GW history to: {path}")
    return path


def save_combined_players_csv(df: pd.DataFrame, name: str = "players_clean_all_seasons.csv") -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved combined players dataset ({len(df)} rows) to: {path}")
    return path


def save_combined_gw_history_csv(df: pd.DataFrame, name: str = "gw_history_all_seasons.csv") -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved combined GW history dataset ({len(df)} rows) to: {path}")
    return path


def process_all_seasons():
    """
    Run the ingestion for all historical seasons and the current season,
    then save combined tables.
    """
    teams_lookup = fetch_master_team_list()
    all_players = []
    all_gw = []

    # Historical via vaastav
    for season in SEASONS:
        print(f"\n=== Processing season {season} (vaastav) ===")
        players_raw = fetch_players_raw_for_season(season)
        save_raw_snapshot_csv(players_raw, season, "players_raw.csv")

        players_clean = build_players_df(players_raw, teams_lookup, season)
        save_players_csv(players_clean, season)
        all_players.append(players_clean)

        gw_raw = fetch_gws_for_season(season)
        save_raw_snapshot_csv(gw_raw, season, "gw_raw_stack.csv")

        gw_history = build_gw_history_df(gw_raw, players_raw, teams_lookup, season)
        save_gw_history_csv(gw_history, season)
        all_gw.append(gw_history)

    # Current season via official API
    print(f"\n=== Processing current season {CURRENT_SEASON} (FPL API) ===")
    bootstrap = fetch_bootstrap_static()
    players_raw_current = build_players_raw_from_bootstrap(bootstrap)
    save_raw_snapshot_csv(players_raw_current, CURRENT_SEASON, "players_raw.csv")

    players_clean_current = build_players_df(players_raw_current, teams_lookup, CURRENT_SEASON)
    save_players_csv(players_clean_current, CURRENT_SEASON)
    all_players.append(players_clean_current)

    gw_raw_current = fetch_gws_for_current_season(bootstrap_data=bootstrap)
    save_raw_snapshot_csv(gw_raw_current, CURRENT_SEASON, "gw_raw_stack.csv")

    gw_history_current = build_gw_history_df(
        gw_raw_current, players_raw_current, teams_lookup, CURRENT_SEASON
    )
    save_gw_history_csv(gw_history_current, CURRENT_SEASON)
    all_gw.append(gw_history_current)

    combined_players = pd.concat(all_players, ignore_index=True)
    save_combined_players_csv(combined_players)

    combined_gw = pd.concat(all_gw, ignore_index=True)
    save_combined_gw_history_csv(combined_gw)

    print("\nDone ingestion.")
    print(f"Seasons processed: {len(SEASONS) + 1} (including {CURRENT_SEASON})")
    print(f"Total player rows: {len(combined_players)}")
    print(f"Total GW rows    : {len(combined_gw)}")


# ======================================================================
# 2) FEATURE ENGINEERING
# ======================================================================

GW_HISTORY_CSV = PROC_DIR / "gw_history_all_seasons.csv"
PLAYERS_ALL_CSV = PROC_DIR / "players_clean_all_seasons.csv"
TRAIN_CSV = PROC_DIR / "player_gw_training.csv"


def load_gw_history() -> pd.DataFrame:
    """
    Load and clean gw_history_all_seasons.csv.
    """
    df = pd.read_csv(GW_HISTORY_CSV, low_memory=False)
    df["season"] = df["season"].astype(str)

    num_cols = [
        "gameweek",
        "player_id",
        "team_id",
        "price",
        "minutes",
        "total_points",
        "opponent_team",
        "selected",
        "transfers_in",
        "transfers_out",
        "transfers_balance",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "was_home" in df.columns:
        mapping = {
            True: True,
            False: False,
            "True": True,
            "False": False,
            1: True,
            0: False,
            "1": True,
            "0": False,
        }
        df["was_home"] = df["was_home"].map(mapping)

    # Fix team_ids for recent seasons if needed
    recent_seasons = {"2020-21", "2021-22", "2022-23", "2023-24", "2024-25"}

    if PLAYERS_ALL_CSV.exists():
        players_all = pd.read_csv(PLAYERS_ALL_CSV)
        players_all["season"] = players_all["season"].astype(str)
        players_all["player_id"] = pd.to_numeric(
            players_all["player_id"], errors="coerce"
        )

        players_sub = players_all[
            players_all["season"].isin(recent_seasons)
        ][["season", "player_id", "team_id", "team_name"]].copy()

        df = df.merge(
            players_sub,
            on=["season", "player_id"],
            how="left",
            suffixes=("", "_p"),
        )

        if "team_id_p" in df.columns:
            df["team_id"] = df["team_id"].where(
                ~df["team_id"].isna(), df["team_id_p"]
            )
            df.drop(columns=["team_id_p"], inplace=True)

        if "team_name_p" in df.columns:
            df["team_name"] = df["team_name"].where(
                df["team_name"].notna() & (df["team_name"] != ""),
                df["team_name_p"],
            )
            df.drop(columns=["team_name_p"], inplace=True)

        if "team_id" in df.columns:
            df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    else:
        print(
            f"WARNING: {PLAYERS_ALL_CSV} not found. "
            "Cannot repair team_id/team_name gaps for 2020-21..2024-25."
        )

    return df


def rolling_features(group: pd.DataFrame, value_col: str, windows=(3, 5), prefix=""):
    group = group.sort_values("gameweek").copy()
    group[f"{prefix}{value_col}_lag1"] = group[value_col].shift(1)

    for w in windows:
        roll = (
            group[value_col]
            .rolling(window=w, min_periods=1)
            .mean()
            .shift(1)
        )
        group[f"{prefix}{value_col}_avg_{w}"] = roll

    return group


def build_player_gw_features(gw: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "season",
        "gameweek",
        "player_id",
        "name",
        "team_id",
        "team_name",
        "position",
        "element_type",
        "price",
        "minutes",
        "total_points",
        "was_home",
        "opponent_team",
    ]
    base_cols = [c for c in base_cols if c in gw.columns]

    df = gw[base_cols].copy()
    df = df.sort_values(["season", "player_id", "gameweek"])

    def player_group_func(g):
        g = rolling_features(g, "total_points", windows=(3, 5), prefix="p_")
        g = rolling_features(g, "minutes", windows=(3, 5), prefix="p_")
        g["p_cum_points_before"] = g["total_points"].cumsum().shift(1)
        g["p_games_played_before"] = (g["minutes"] > 0).astype(int).cumsum().shift(1)
        return g

    df = df.groupby(["season", "player_id"], group_keys=False).apply(player_group_func)

    feature_cols = [c for c in df.columns if c.startswith("p_")]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    df = df.drop_duplicates(subset=["season", "gameweek", "player_id"])
    return df


def build_team_gw_features(gw: pd.DataFrame) -> pd.DataFrame:
    df = gw.copy()
    if "team_id" not in df.columns:
        raise ValueError("gw history must contain 'team_id' to build team features.")

    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df = df[~df["team_id"].isna()].copy()

    group_cols = ["season", "gameweek", "team_id"]
    if "team_name" in df.columns:
        group_cols.append("team_name")

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            team_total_points=("total_points", "sum"),
            team_total_minutes=("minutes", "sum"),
            team_avg_price=("price", "mean"),
        )
        .reset_index()
    )

    def first_non_null(s: pd.Series):
        s_non = s.dropna()
        return s_non.iloc[0] if len(s_non) > 0 else np.nan

    if "opponent_team" in df.columns:
        opp = (
            df.groupby(group_cols, dropna=False)["opponent_team"]
            .apply(first_non_null)
            .reset_index()
        )
        agg = agg.merge(opp, on=group_cols, how="left")

    if "was_home" in df.columns:
        wh = (
            df.groupby(group_cols, dropna=False)["was_home"]
            .apply(first_non_null)
            .reset_index()
        )
        agg = agg.merge(wh, on=group_cols, how="left")

    agg = agg.sort_values(["season", "team_id", "gameweek"])

    def team_group_func(g):
        g = rolling_features(g, "team_total_points", windows=(3, 5), prefix="t_")
        g = rolling_features(g, "team_total_minutes", windows=(3, 5), prefix="t_")
        g["t_points_per_90"] = (
            g["team_total_points"]
            / (g["team_total_minutes"] / 90.0).replace(0, np.nan)
        )
        g["t_points_per_90"] = g["t_points_per_90"].fillna(0.0)
        return g

    agg = agg.groupby(["season", "team_id"], group_keys=False).apply(team_group_func)

    feat_cols = [c for c in agg.columns if c.startswith("t_")]
    agg[feat_cols] = agg[feat_cols].fillna(0.0)

    return agg


def build_fixture_gw_features(team_gw: pd.DataFrame) -> pd.DataFrame:
    df = team_gw.copy()

    required = {"season", "gameweek", "team_id"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"team_gw missing columns: {missing_required}")

    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df = df[~df["team_id"].isna() & ~df["gameweek"].isna()].copy()

    if "opponent_team" not in df.columns:
        print("No opponent_team column; returning team_gw unchanged.")
        return df

    df["opponent_team"] = pd.to_numeric(df["opponent_team"], errors="coerce")
    if df["opponent_team"].isna().all():
        print("All opponent_team NaN; returning team_gw unchanged.")
        return df

    team_feature_cols = [
        c
        for c in df.columns
        if c.startswith("t_") or c in ["team_total_points", "team_total_minutes"]
    ]

    opp = df[["season", "gameweek", "team_id"] + team_feature_cols].copy()
    opp = opp.rename(columns={"team_id": "opponent_team"})
    opp = opp.rename(columns={col: f"opp_{col}" for col in team_feature_cols})

    merged = df.merge(
        opp,
        on=["season", "gameweek", "opponent_team"],
        how="left",
    )

    opp_cols = [c for c in merged.columns if c.startswith("opp_")]
    merged[opp_cols] = merged[opp_cols].fillna(0.0)

    return merged


def build_player_training_table(
    player_gw: pd.DataFrame, fixture_gw: pd.DataFrame
) -> pd.DataFrame:
    merge_cols = ["season", "gameweek", "team_id"]
    merge_cols = [c for c in merge_cols if c in player_gw.columns and c in fixture_gw.columns]

    fixture_cols = [
        c
        for c in fixture_gw.columns
        if c not in ["team_name", "opponent_team", "was_home"]
        and c not in ["season", "gameweek", "team_id"]
    ]

    df = player_gw.merge(
        fixture_gw[["season", "gameweek", "team_id"] + fixture_cols],
        on=merge_cols,
        how="left",
    )

    fx_cols = [c for c in df.columns if c.startswith("t_") or c.startswith("opp_")]
    df[fx_cols] = df[fx_cols].fillna(0.0)

    if {"season", "gameweek", "player_id"}.issubset(df.columns):
        df = df.sort_values(["season", "player_id", "gameweek"])
        df = df.drop_duplicates(subset=["season", "gameweek", "player_id"])

    return df


def check_for_duplicates(df, keys, name="DataFrame"):
    if not set(keys).issubset(df.columns):
        print(f"[{name}] Missing columns for duplicate check: {keys}")
        return

    dups = df[df.duplicated(subset=keys, keep=False)]
    if len(dups) > 0:
        print(f"\n⚠️  [{name}] DUPLICATES for {keys}: {len(dups)} rows")
        print(dups[keys].head())
    else:
        print(f"[{name}] No duplicates for {keys}.")


def check_missing_ids(df, cols, name="DataFrame"):
    for col in cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"⚠️  [{name}] {missing} rows have missing {col}")
        else:
            print(f"[{name}] Column {col} not present.")


def check_was_home_distribution(df, name="DataFrame"):
    if "was_home" not in df.columns:
        print(f"[{name}] No was_home column.")
        return
    print(f"\n[{name}] was_home distribution:")
    print(df["was_home"].value_counts(dropna=False))


def check_numeric_nan(df, cols, name="DataFrame"):
    for col in cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                print(f"⚠️  [{name}] {n_missing} NaNs in '{col}'")
        else:
            print(f"[{name}] Column {col} not found.")


def sanity_check_outputs(player_gw, team_gw, fixture_gw, training_df):
    print("\n==============================")
    print("🔍 SANITY CHECKS")
    print("==============================\n")

    print("\n### PLAYER_GW_FEATURES ###\n")
    check_for_duplicates(player_gw, ["season", "gameweek", "player_id"], "player_gw")
    check_missing_ids(player_gw, ["player_id", "team_id"], "player_gw")
    check_numeric_nan(player_gw, ["minutes", "total_points", "price"], "player_gw")
    check_was_home_distribution(player_gw, "player_gw")

    print("\n### TEAM_GW_FEATURES ###\n")
    check_for_duplicates(team_gw, ["season", "gameweek", "team_id"], "team_gw")
    check_missing_ids(team_gw, ["team_id", "opponent_team"], "team_gw")
    check_numeric_nan(team_gw, ["team_total_points", "team_total_minutes"], "team_gw")

    print("\n### FIXTURE_GW_FEATURES ###\n")
    check_for_duplicates(fixture_gw, ["season", "gameweek", "team_id"], "fixture_gw")
    check_missing_ids(fixture_gw, ["team_id", "opponent_team"], "fixture_gw")

    print("\n### PLAYER_GW_TRAINING ###\n")
    check_for_duplicates(training_df, ["season", "gameweek", "player_id"], "player_gw_training")
    check_missing_ids(training_df, ["player_id", "team_id"], "player_gw_training")
    check_numeric_nan(training_df, ["minutes", "total_points", "price"], "player_gw_training")

    print("\n🔎 SANITY CHECKS COMPLETE\n")


def build_all_feature_tables():
    print("Loading gw_history_all_seasons.csv ...")
    gw = load_gw_history()

    print("Building player GW features ...")
    player_gw = build_player_gw_features(gw)
    player_gw_path = PROC_DIR / "player_gw_features.csv"
    player_gw.to_csv(player_gw_path, index=False)
    print(f"Saved player GW features to: {player_gw_path}")

    print("Building team GW features ...")
    team_gw = build_team_gw_features(gw)
    team_gw_path = PROC_DIR / "team_gw_features.csv"
    team_gw.to_csv(team_gw_path, index=False)
    print(f"Saved team GW features to: {team_gw_path}")

    print("Building fixture GW features (team + opponent) ...")
    fixture_gw = build_fixture_gw_features(team_gw)
    fixture_gw_path = PROC_DIR / "fixture_gw_features.csv"
    fixture_gw.to_csv(fixture_gw_path, index=False)
    print(f"Saved fixture GW features to: {fixture_gw_path}")

    print("Building final player training table (features only, no target) ...")
    training_df = build_player_training_table(player_gw, fixture_gw)
    training_path = PROC_DIR / "player_gw_training.csv"
    training_df.to_csv(training_path, index=False)
    print(f"Saved player training table to: {training_path}")

    print("\nDone feature engineering. Shapes:")
    print("  player_gw_features :", player_gw.shape)
    print("  team_gw_features   :", team_gw.shape)
    print("  fixture_gw_features:", fixture_gw.shape)
    print("  player_gw_training :", training_df.shape)

    sanity_check_outputs(player_gw, team_gw, fixture_gw, training_df)


# ======================================================================
# 3) MODEL TRAINING
# ======================================================================

def load_training_data(path: Path = TRAIN_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    return df


def add_next_gw_target(df: pd.DataFrame, target_col: str = "target_next_gw_points") -> pd.DataFrame:
    if "total_points" not in df.columns:
        raise ValueError("Column 'total_points' is required to build the target.")

    if "gameweek" not in df.columns:
        raise ValueError("Column 'gameweek' missing from training data.")

    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df = df.sort_values(["season", "player_id", "gameweek"]).copy()

    df[target_col] = (
        df.groupby(["season", "player_id"])["total_points"]
        .shift(-1)
    )

    df = df[~df[target_col].isna()].copy()
    return df


def prepare_features(df: pd.DataFrame):
    target_col = "target_next_gw_points"
    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' missing. "
            "Make sure add_next_gw_target() has been applied."
        )

    drop_cols = {
        "season",
        "gameweek",
        "player_id",
        "name",
        "team_id",
        "team_name",
        "position",
        "element_type",
        target_col,
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    return X, y, feature_cols


def train_and_evaluate_model(name, model, X_train, y_train, X_valid, y_valid):
    print(f"\n=== Training {name} ===")
    start = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start
    print(f"⏱ {name} training time: {train_time:.2f} seconds")

    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))

    print(f"{name} - MAE : {mae:.3f}")
    print(f"{name} - RMSE: {rmse:.3f}")

    return {
        "name": name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "train_time": train_time,
        "y_pred": y_pred,
    }


def train_all_models():
    total_start = time.time()

    print(f"Loading training data from {TRAIN_CSV}")
    df = load_training_data()

    print("Adding target_next_gw_points ...")
    df = add_next_gw_target(df)
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")

    valid_df = df[(df["season"] == VAL_SEASON) & (df["gameweek"] <= VAL_MAX_GW)].copy()
    train_df = df[df["season"] != VAL_SEASON].copy()

    if valid_df.empty:
        raise ValueError(
            f"Validation set is empty after building target! "
            f"Check that season={VAL_SEASON} with gameweek <= {VAL_MAX_GW} exists."
        )

    print(
        f"Train rows (with target): {len(train_df)}, "
        f"Valid rows (with target): {len(valid_df)}"
    )

    X_train, y_train, feature_cols = prepare_features(train_df)
    X_valid, y_valid, _ = prepare_features(valid_df)

    valid_meta = valid_df[["season", "gameweek", "player_id"]].reset_index(drop=True)
    y_valid_array = y_valid.to_numpy()

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=100,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=1,
        ),
    }

    short_key = {
        "RandomForest": "rf",
        "LightGBM": "lgbm",
        "XGBoost": "xgb",
    }

    results = []
    per_model_preds = {}

    for name, model in models.items():
        res = train_and_evaluate_model(
            name, model, X_train, y_train, X_valid, y_valid
        )
        results.append(
            {
                "model": name,
                "mae": res["mae"],
                "rmse": res["rmse"],
                "train_time_sec": res["train_time"],
            }
        )
        per_model_preds[name] = res

    results_df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    comp_path = BACKTEST_DIR / f"model_comparison_{VAL_SEASON}_gw1_{VAL_MAX_GW}.csv"
    results_df.to_csv(comp_path, index=False)

    print(f"\nSaved model comparison table to: {comp_path}")
    print(results_df.to_string(index=False))

    for name, res in per_model_preds.items():
        sk = short_key[name]

        model_path = MODELS_DIR / f"{sk}_points_model_seasons.pkl"
        meta_path = MODELS_DIR / f"{sk}_points_model_seasons_metadata.json"

        joblib.dump({"model": res["model"], "feature_cols": feature_cols}, model_path)

        metadata = {
            "type": name,
            "short_key": sk,
            "train_seasons": sorted(train_df["season"].unique().tolist()),
            "test_season": VAL_SEASON,
            "test_gameweeks": f"1-{VAL_MAX_GW}",
            "mae_valid": float(res["mae"]),
            "rmse_valid": float(res["rmse"]),
            "train_time_sec": float(res["train_time"]),
            "feature_cols": feature_cols,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved {name} model + metadata (key={sk}).")

        y_pred = np.asarray(res["y_pred"])
        preds_df = valid_meta.copy()
        preds_df["y_true"] = y_valid_array
        preds_df["y_pred"] = y_pred

        preds_backtest_path = (
            BACKTEST_DIR / f"predictions_{VAL_SEASON}_gw1_{VAL_MAX_GW}_{sk}.csv"
        )
        preds_df.to_csv(preds_backtest_path, index=False)
        print(
            f"Saved detailed {VAL_SEASON} GW1–{VAL_MAX_GW} predictions for "
            f"{name} to: {preds_backtest_path}"
        )

        lp_df = preds_df[["season", "gameweek", "player_id", "y_pred"]].rename(
            columns={"y_pred": "predicted_next_points"}
        )
        lp_path = PROC_DIR / f"player_gw_predictions_{VAL_SEASON}_gw1_{VAL_MAX_GW}_{sk}.csv"
        lp_df.to_csv(lp_path, index=False)
        print(f"Saved LP-ready predictions for {name} to: {lp_path}")

    total_time = time.time() - total_start
    print(f"\n⏱ Total training runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


# ======================================================================
# 4) LP TEAM OPTIMIZER + ML SEASON SIM
# ======================================================================

def build_team_optimizer(
    players: pd.DataFrame,
    budget: float,
    objective_col: str = "predicted_points",
    prev_squad_ids: Optional[List[int]] = None,
    transfers_per_gw: int = 0,
):
    """
    players: dataframe with columns:
        [player_id, name, team_name, team_id, position, price, predicted_points]
    """
    df = players.copy()
    df["price"] = df["price"].astype(float)
    df["predicted_points"] = df[objective_col].astype(float)
    df["player_id"] = df["player_id"].astype(int)

    player_ids = df["player_id"].tolist()

    model = pl.LpProblem("FPL_Team_Optimization", pl.LpMaximize)

    x = pl.LpVariable.dicts("select", player_ids, cat="Binary")
    s = pl.LpVariable.dicts("starter", player_ids, cat="Binary")
    c = pl.LpVariable.dicts("captain", player_ids, cat="Binary")

    model += pl.lpSum([
        s[i] * df.loc[df.player_id == i, "predicted_points"].values[0]
        + c[i] * df.loc[df.player_id == i, "predicted_points"].values[0]
        for i in player_ids
    ])

    model += pl.lpSum([x[i] for i in player_ids]) == 15
    model += pl.lpSum([s[i] for i in player_ids]) == 11
    model += pl.lpSum([c[i] for i in player_ids]) == 1

    for i in player_ids:
        model += c[i] <= s[i]
        model += s[i] <= x[i]

    model += pl.lpSum([x[i] * df.loc[df.player_id == i, "price"].values[0]
                       for i in player_ids]) <= budget

    for pos, (min_req, max_req) in FORMATION_LIMITS.items():
        player_ids_pos = df[df["position"] == pos]["player_id"].tolist()
        if player_ids_pos:
            model += pl.lpSum([x[i] for i in player_ids_pos]) >= min_req
            model += pl.lpSum([x[i] for i in player_ids_pos]) <= max_req

    for tid in df["team_id"].unique():
        pids_team = df[df["team_id"] == tid]["player_id"].tolist()
        model += pl.lpSum([x[i] for i in pids_team]) <= 3

    if prev_squad_ids is not None:
        prev_set = set(prev_squad_ids)
        new_arrivals = [
            x[i] for i in player_ids if i not in prev_set
        ]
        model += pl.lpSum(new_arrivals) == transfers_per_gw

    model.solve(pl.PULP_CBC_CMD(msg=0))

    rows = []
    for i in player_ids:
        rows.append({
            "player_id": i,
            "name": df.loc[df.player_id == i, "name"].values[0],
            "team_name": df.loc[df.player_id == i, "team_name"].values[0],
            "team_id": df.loc[df.player_id == i, "team_id"].values[0],
            "position": df.loc[df.player_id == i, "position"].values[0],
            "price": df.loc[df.player_id == i, "price"].values[0],
            "predicted_points": df.loc[df.player_id == i, "predicted_points"].values[0],
            "is_selected": int(pl.value(x[i])),
            "is_starter": int(pl.value(s[i])),
            "is_captain": int(pl.value(c[i])),
        })

    best_team = pd.DataFrame(rows)
    best_team = best_team[best_team["is_selected"] == 1].reset_index(drop=True)

    return best_team, model


def load_players_current_season():
    cur_path = PROC_DIR / f"{CURRENT_SEASON}_players_clean.csv"
    if not cur_path.exists():
        raise FileNotFoundError(
            f"Current-season players file not found: {cur_path}\n"
            "Run data ingestion first."
        )

    df = pd.read_csv(cur_path)

    required_cols = [
        "player_id",
        "name",
        "team_name",
        "team_id",
        "position",
        "price",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in current-season players file: {missing}")

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


def load_predictions_table(model_key: str) -> pd.DataFrame:
    pred_path = PROC_DIR / PREDICTIONS_TEMPLATE.format(
        season=CURRENT_SEASON,
        max_gw=PREDICTION_END_GW,
        model=model_key,
    )

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}\n"
            "Run model training first."
        )

    df = pd.read_csv(pred_path)
    df["season"] = df["season"].astype(str)
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")

    if "predicted_next_points" not in df.columns:
        raise ValueError(
            f"Column 'predicted_next_points' missing in {pred_path.name}"
        )

    return df


def build_predictions_for_gw(gw: int, players: pd.DataFrame, preds_all: pd.DataFrame):
    preds_season = preds_all[preds_all["season"] == CURRENT_SEASON].copy()
    if preds_season.empty:
        raise ValueError(f"No predictions rows for season={CURRENT_SEASON}.")

    max_lookup_gw = int(preds_season["gameweek"].max())

    if gw <= 1:
        raise ValueError("GW must be >= 2 for optimizer.")

    lookup_gw = gw - 1
    if lookup_gw > max_lookup_gw:
        lookup_gw = max_lookup_gw

    preds_gw = preds_season[preds_season["gameweek"] == lookup_gw].copy()
    if preds_gw.empty:
        raise ValueError(
            f"No predictions found for season={CURRENT_SEASON}, lookup_gw={lookup_gw}"
        )

    merged = players.merge(
        preds_gw[["player_id", "predicted_next_points"]],
        on="player_id",
        how="inner",
    )
    merged = merged.rename(columns={"predicted_next_points": "predicted_points"})

    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged["predicted_points"] = pd.to_numeric(
        merged["predicted_points"], errors="coerce"
    )
    merged = merged.dropna(subset=["price", "predicted_points"]).copy()

    merged = (
        merged.sort_values("predicted_points", ascending=False)
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )

    return merged


def optimize_season_for_model(
    model_key: str,
    start_gw: int = PREDICTION_START_GW,
    end_gw: int = PREDICTION_END_GW,
    budget: float = 100.0,
) -> Dict[str, pd.DataFrame]:
    print(
        f"\n=============================="
        f"\n Optimizing season for model: {model_key} ({MODEL_NAME.get(model_key, model_key)})"
        f"\n=============================="
    )

    players_cur = load_players_current_season()
    preds_all = load_predictions_table(model_key)

    prev_squad_ids: Optional[List[int]] = None
    all_squads: Dict[str, pd.DataFrame] = {}
    total_season_pred_points = 0.0

    solutions_dir = BASE_SOLUTIONS_DIR / model_key
    solutions_dir.mkdir(parents=True, exist_ok=True)

    for gw in range(start_gw, end_gw + 1):
        print(f"\n=== {model_key} - ML-based LP optimization for GW {gw} ===")

        preds_df = build_predictions_for_gw(gw, players_cur, preds_all)
        print(f"Loaded {len(preds_df)} unique players with predictions for GW {gw}.")

        transfers_per_gw = 0 if prev_squad_ids is None else 1

        best_team, model = build_team_optimizer(
            players=preds_df,
            budget=budget,
            objective_col="predicted_points",
            prev_squad_ids=prev_squad_ids,
            transfers_per_gw=transfers_per_gw,
        )

        status = pl.LpStatus[model.status]
        print("LP status:", status)

        gw_pred_points = (
            best_team["predicted_points"]
            * best_team["is_starter"].astype(int)
            * (1 + best_team["is_captain"].astype(int))
        ).sum()
        total_season_pred_points += gw_pred_points

        print(
            f"Predicted points for starting XI (with captain) in GW{gw}: "
            f"{gw_pred_points:.2f}"
        )

        cols_to_show = [
            "is_starter",
            "is_captain",
            "position",
            "name",
            "team_name",
            "price",
            "predicted_points",
        ]
        print("\nOptimal ML squad for this GW:")
        print(best_team[cols_to_show].to_string(index=False))

        gw_tag = f"GW{gw:02d}"

        out_path = solutions_dir / f"{model_key}_team_{gw_tag}.csv"
        best_team.to_csv(out_path, index=False)
        print(f"Saved optimized ML squad for {gw_tag} to: {out_path}")

        xi_path = solutions_dir / f"{model_key}_starting_xi_{gw_tag}.csv"
        best_team[best_team["is_starter"] == 1].to_csv(xi_path, index=False)
        print(f"Saved starting XI for {gw_tag} to: {xi_path}")

        prev_squad_ids = best_team["player_id"].tolist()
        all_squads[gw_tag] = best_team

    print(
        f"\n=== {model_key} season summary (GW{start_gw}–GW{end_gw}) ===\n"
        f"Total predicted points over season (starting XI + captain): {total_season_pred_points:.2f}"
    )

    return all_squads


# ======================================================================
# 5) REAL FPL SCORE COMPARISON
# ======================================================================

_live_cache: Dict[int, pd.DataFrame] = {}
_avg_cache: Dict[int, float] = {}


def fetch_gw_live_points(gw: int) -> pd.DataFrame:
    if gw in _live_cache:
        return _live_cache[gw].copy()

    url = f"{FPL_API_BASE}/event/{gw}/live/"
    resp = requests.get(url, timeout=15)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch FPL live data for GW {gw}: HTTP {resp.status_code}"
        )

    data = resp.json()
    elements = data.get("elements", [])

    rows = []
    for el in elements:
        stats = el.get("stats", {}) or {}
        rows.append(
            {
                "player_id": el.get("id"),
                "total_points": stats.get("total_points", 0),
                "minutes": stats.get("minutes", 0),
                "goals_scored": stats.get("goals_scored", 0),
                "assists": stats.get("assists", 0),
                "clean_sheets": stats.get("clean_sheets", 0),
                "goals_conceded": stats.get("goals_conceded", 0),
                "saves": stats.get("saves", 0),
                "bonus": stats.get("bonus", 0),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        df["total_points"] = (
            pd.to_numeric(df["total_points"], errors="coerce").fillna(0.0)
        )

    _live_cache[gw] = df.copy()
    return df


def fetch_fpl_average_scores_from_bootstrap() -> Dict[int, float]:
    if _avg_cache:
        return _avg_cache

    url = f"{FPL_API_BASE}/bootstrap-static/"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    events = data.get("events", [])

    for ev in events:
        gw = ev.get("id")
        avg = ev.get("average_entry_score")
        if gw is not None and avg is not None:
            _avg_cache[int(gw)] = float(avg)

    return _avg_cache


def load_starting_xi(model_key: str, gw: int) -> pd.DataFrame:
    gw_tag = f"GW{gw:02d}"
    xi_path = BASE_SOLUTIONS_DIR / model_key / f"{model_key}_starting_xi_{gw_tag}.csv"

    if not xi_path.exists():
        raise FileNotFoundError(
            f"Starting XI file not found for model={model_key}, GW={gw}: {xi_path}\n"
            f"Run the season optimizer with end_gw >= {gw}."
        )

    df = pd.read_csv(xi_path)

    if "player_id" not in df.columns:
        raise ValueError(f"Column 'player_id' missing in {xi_path}")
    if "is_captain" not in df.columns:
        raise ValueError(f"Column 'is_captain' missing in {xi_path}")

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")

    df["is_captain"] = (
        df["is_captain"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    )

    return df[["player_id", "is_captain"]].copy()


def compute_model_score_for_gw(model_key: str, gw: int) -> float:
    xi = load_starting_xi(model_key, gw)
    live = fetch_gw_live_points(gw)

    merged = xi.merge(
        live[["player_id", "total_points"]],
        on="player_id",
        how="left"
    )

    merged["total_points"] = merged["total_points"].fillna(0.0)
    merged["multiplier"] = 1 + merged["is_captain"].astype(int)
    merged["effective_points"] = merged["total_points"] * merged["multiplier"]

    return float(merged["effective_points"].sum())


def compare_models_over_gws(start_gw: int = 2, end_gw: int = PREDICTION_END_GW) -> pd.DataFrame:
    if start_gw < 2:
        raise ValueError("start_gw must be >= 2, because ML optimizer starts GW2.")

    try:
        avg_map = fetch_fpl_average_scores_from_bootstrap()
    except Exception as e:
        print(f"[WARNING] Could not fetch FPL averages: {e}")
        avg_map = {}

    rows: List[Dict] = []
    totals = {m: 0.0 for m in MODELS}
    total_avg = 0.0

    for gw in range(start_gw, end_gw + 1):
        print(f"\n=== Computing scores for GW {gw} ===")

        row = {"gameweek": gw}

        for model_key in MODELS:
            score = compute_model_score_for_gw(model_key, gw)
            row[f"{model_key}_score"] = score
            totals[model_key] += score

            print(f"  {MODEL_NAME[model_key]}: {score:.2f}")

        if gw in avg_map:
            avg = avg_map[gw]
            row["fpl_average_score"] = avg
            total_avg += avg
            print(f"  FPL average: {avg:.2f}")
        else:
            row["fpl_average_score"] = float("nan")
            print("  FPL average: N/A")

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("gameweek").reset_index(drop=True)

    print("\n=== TOTAL SCORES ===")
    for m in MODELS:
        print(f"{MODEL_NAME[m]} total: {totals[m]:.2f}")

    if avg_map:
        print(f"FPL average total: {total_avg:.2f}")
    else:
        print("FPL average total: (missing)")

    out_path = BACKTEST_DIR / f"model_season_scores_api_gw{start_gw}_{end_gw}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved comparison to: {out_path}")
    return df


# ======================================================================
# WRAPPERS + MAIN
# ======================================================================

def run_data_ingestion():
    print("\n==============================")
    print("STEP 1: DATA INGESTION")
    print("==============================\n")
    process_all_seasons()


def run_feature_engineering():
    print("\n==============================")
    print("STEP 2: FEATURE ENGINEERING")
    print("==============================\n")
    build_all_feature_tables()


def run_model_training():
    print("\n==============================")
    print("STEP 3: MODEL TRAINING")
    print("==============================\n")
    train_all_models()


def run_season_optimizers():
    print("\n==============================")
    print("STEP 4: ML+LP SEASON OPTIMIZATION")
    print("==============================\n")
    for m in MODELS:
        optimize_season_for_model(
            model_key=m,
            start_gw=PREDICTION_START_GW,
            end_gw=PREDICTION_END_GW,
            budget=100.0,
        )


def run_model_evaluation():
    print("\n==============================")
    print("STEP 5: MODEL EVALUATION vs REAL FPL")
    print("==============================\n")
    compare_models_over_gws(start_gw=2, end_gw=PREDICTION_END_GW)


def main():
    """
    End-to-end pipeline:
    1) Ingestion
    2) Feature engineering
    3) Model training
    4) ML+LP optimization
    5) Evaluation vs real FPL & average team
    """
    total_start = time.time()

    run_data_ingestion()
    run_feature_engineering()
    run_model_training()
    run_season_optimizers()
    run_model_evaluation()

    total_time = time.time() - total_start
    print(f"\n==============================")
    print(f"FULL PIPELINE DONE in {total_time:.2f} seconds "
          f"({total_time/60:.2f} minutes)")
    print("==============================\n")


if __name__ == "__main__":
    main()