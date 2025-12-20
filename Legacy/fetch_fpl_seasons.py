#use more than only the past few gameweeks - use seasons 2016/17 until 2024/25

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
# GitHub base for vaastav FPL dataset
FPL_GITHUB_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

# Seasons available (you can extend this list later if needed)
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

# Project paths (same style as your existing script)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# -------------------------------------------------------------------
# FETCHING DATA FROM GITHUB
# -------------------------------------------------------------------
def fetch_players_raw_for_season(season: str) -> pd.DataFrame:
    """
    Fetch players_raw.csv for a given season from the vaastav GitHub repo.
    """
    url = f"{FPL_GITHUB_BASE}/{season}/players_raw.csv"
    df = pd.read_csv(url)
    return df


def fetch_master_team_list() -> pd.DataFrame:
    """
    Fetch master_team_list.csv which maps (season, team_id) -> team_name.
    """
    url = f"{FPL_GITHUB_BASE}/master_team_list.csv"
    df = pd.read_csv(url)
    # Ensure consistent column names
    # master_team_list has columns: season, team, team_name
    df.rename(columns={"team": "team_id"}, inplace=True)
    return df


def save_raw_snapshot_csv(df: pd.DataFrame, season: str, name: str = "players_raw.csv") -> Path:
    """
    Save raw CSV snapshot for reproducibility.

    Files are saved under:
        data/raw/{season}_{name}
    e.g. data/raw/2016-17_players_raw.csv
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved raw snapshot to: {path.resolve()}")
    return path


# -------------------------------------------------------------------
# BUILD CLEAN PLAYER DF (SIMILAR SHAPE AS YOUR ORIGINAL)
# -------------------------------------------------------------------
def build_players_df(players: pd.DataFrame, teams_lookup: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Create a clean player dataframe with all variables needed for the optimizer
    for a single season, using players_raw.csv + master_team_list.csv.

    This mirrors the structure of your original build_players_df, with an
    extra 'season' column.
    """
    # Subset to the columns we care about
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

    # Rename columns for consistency
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

    # Human-readable fields
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["price_tenths"] / 10.0  # e.g. 75 -> 7.5m
    df["name"] = (
        df["first_name"].fillna("").astype(str).str.strip()
        + " "
        + df["last_name"].fillna("").astype(str).str.strip()
    ).str.strip()

    # Drop redundant columns
    df.drop(columns=["first_name", "last_name"], inplace=True, errors="ignore")
    df.drop(columns=["price_tenths"], inplace=True, errors="ignore")

    # Add season column
    df["season"] = season

    # Convert numeric columns safely
    numeric_cols = [
        "total_points",
        "minutes",
        "form",
        "points_per_game",
        "selected_by_percent",
        "price",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)

    df[numeric_cols] = df[numeric_cols].round(3)

    # Key metrics for the project
    price_nonzero = df["price"].replace(0, np.nan)
    minutes_nonzero_90 = df["minutes"].replace(0, np.nan) / 90.0

    df["points_per_million"] = df["total_points"] / price_nonzero
    df["points_per_90"] = df["total_points"] / minutes_nonzero_90

    # Round metrics for readability
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

    # Join team names using master_team_list
    season_teams = teams_lookup[teams_lookup["season"] == season].copy()
    df = df.merge(
        season_teams[["team_id", "team_name"]],
        on="team_id",
        how="left",
    )

    # Re-order columns to be optimizer-friendly
    cols_final = [
        "season",          # extra vs your original
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

    print(f"[{season}] FINAL column order:", list(df.columns))
    return df


# -------------------------------------------------------------------
# SAVING CLEANED DATA
# -------------------------------------------------------------------
def save_players_csv(df: pd.DataFrame, season: str, name: str = "players_clean.csv") -> Path:
    """
    Save clean player dataset for a given season.

    Files are saved under:
        data/processed/{season}_{name}
    e.g. data/processed/2016-17_players_clean.csv
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = PROC_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved cleaned players to: {path.resolve()}")
    return path


def save_combined_players_csv(df: pd.DataFrame, name: str = "players_clean_all_seasons.csv") -> Path:
    """
    Save a single combined CSV with all seasons stacked together.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved combined players dataset ({len(df)} rows) to: {path.resolve()}")
    return path


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    teams_lookup = fetch_master_team_list()
    all_seasons_dfs = []

    for season in SEASONS:
        print(f"\n=== Processing season {season} ===")
        players_raw = fetch_players_raw_for_season(season)
        save_raw_snapshot_csv(players_raw, season=season, name="players_raw.csv")

        players_df = build_players_df(players_raw, teams_lookup=teams_lookup, season=season)
        save_players_csv(players_df, season=season)

        all_seasons_dfs.append(players_df)

    # Concatenate all seasons into one big dataset
    combined_df = pd.concat(all_seasons_dfs, ignore_index=True)
    save_combined_players_csv(combined_df)

    print(f"\nDone. Processed {len(SEASONS)} seasons.")
    print(f"Total player rows across all seasons: {len(combined_df)}")


if __name__ == "__main__":
    main()
