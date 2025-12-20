"""
Build clean player-level and gameweek-level datasets from vaastav FPL repo
for seasons 2016-17 to 2024-25, and append 2025-26 from the official FPL API.

Outputs:
- data/processed/<season>_players_clean.csv
- data/processed/<season>_gw_history.csv
- data/processed/players_clean_all_seasons.csv
- data/processed/gw_history_all_seasons.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests  # for official FPL API

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
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

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# -------------------------------------------------------------------
# Which extra per-match stats we try to preserve in gw_history
# (if present in the raw season gwX.csv or API response)
# -------------------------------------------------------------------
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
    "ea_index",                 # older seasons

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

    # passing / possession / defending detail (earlier and 25-26 schema)
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
    "modified",                 # added in 24-25 / 25-26
]

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# FETCHING DATA FROM GITHUB (2016-17 -> 2024-25)
# -------------------------------------------------------------------
def fetch_players_raw_for_season(season: str) -> pd.DataFrame:
    url = f"{FPL_GITHUB_BASE}/{season}/players_raw.csv"
    df = read_csv_robust(url)
    return df


def fetch_gws_for_season(season: str, max_gw: int = 60) -> pd.DataFrame:
    """
    Fetch all gwX.csv files for a season and stack them (vaastav).

    - Tries gw1.csv, gw2.csv, ... up to max_gw
    - Stops when it hits the first error (typically 404)
    """
    all_gws = []
    for gw in range(1, max_gw + 1):
        url = f"{FPL_GITHUB_BASE}/{season}/gws/gw{gw}.csv"
        try:
            df = read_csv_robust(
                url,
                engine="python",      # tolerant parser
                on_bad_lines="skip",  # skip malformed lines
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


# -------------------------------------------------------------------
# OFFICIAL FPL API FETCHING (2025-26)
# -------------------------------------------------------------------
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

    Uses:
    - /event/{gw}/live for per-GW stats
    - bootstrap-static (passed in as bootstrap_data) for player meta (name, team, cost)

    We try to align column names with vaastav gwX.csv where possible.
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
            "now_cost": "value",  # tenths of a million, like vaastav
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

        # Map official stats to vaastav-style columns where possible
        # (many names already match; we just choose subset we care about)
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

        # merge in static info (name, team, value)
        gw_df = gw_df.merge(
            players_static[["player_id", "name", "team_id", "value"]],
            on="player_id",
            how="left",
        )

        # rename to mimic vaastav gwX.csv schema
        gw_df.rename(
            columns={
                "player_id": "element",  # vaastav uses 'element' as player id
                "team_id": "team",
                "value": "value",        # tenths of a million
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


# -------------------------------------------------------------------
# BUILD CLEAN PLAYER DF (SEASON SUMMARY)
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# BUILD GW HISTORY (PLAYER x GW)
# -------------------------------------------------------------------
def build_gw_history_df(
    gw_raw: pd.DataFrame,
    players_raw: pd.DataFrame,
    teams_lookup: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """
    Build per-player per-GW history, preserving as many useful
    per-match stats as possible across all seasons.

    - Normalizes IDs: element -> player_id, round -> gameweek
    - Uses players_raw to define element_type, team_id, position
    - Combines with team_name via master_team_list
    - Keeps a wide set of FPL stats (goals, assists, BPS, ICT, xG, xA, etc.)
    """
    df = gw_raw.copy()

    # basic IDs
    if "element" in df.columns:
        df.rename(columns={"element": "player_id"}, inplace=True)
    if "team" in df.columns and "team_id" not in df.columns:
        df.rename(columns={"team": "team_id"}, inplace=True)

    if "GW" in df.columns:
        df.rename(columns={"GW": "gameweek"}, inplace=True)
    elif "round" in df.columns and "gameweek" not in df.columns:
        df.rename(columns={"round": "gameweek"}, inplace=True)

    # price
    if "value" in df.columns:
        df["price"] = df["value"] / 10.0
    elif "price" not in df.columns:
        df["price"] = np.nan

    # build meta from players_raw
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

    # merge meta
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.merge(meta, on="player_id", how="left", suffixes=("", "_from_players"))

    # team_id from players_raw wins if needed
    if "team_id_from_players" in df.columns:
        df["team_id"] = df["team_id"].fillna(df["team_id_from_players"])
        df.drop(columns=["team_id_from_players"], inplace=True)

    # normalize GK label
    df["position"] = df["position"].replace({"GKP": "GK"})

    # attach team_name
    df = standardize_team_id(df, "team_id")
    season_teams = teams_lookup[teams_lookup["season"] == season].copy()
    season_teams = standardize_team_id(season_teams, "team_id")

    df = df.merge(
        season_teams[["team_id", "team_name"]],
        on="team_id",
        how="left",
    )

    # ensure these base columns exist
    df["season"] = season

    # final column ordering
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

    # choose extra stats that actually exist in this season's data
    extra_present = [c for c in EXTRA_STAT_COLS if c in df.columns]

    # build final list without duplicates
    cols_final = []
    for c in cols_base + extra_present:
        if c not in cols_final:
            cols_final.append(c)

    df = df.reindex(columns=cols_final).copy()

    print(f"[{season}] GW history columns: {list(df.columns)}")
    return df


# -------------------------------------------------------------------
# SAVING
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    teams_lookup = fetch_master_team_list()
    all_players = []
    all_gw = []

    # 1) Historical seasons from vaastav (2016-17 -> 2024-25)
    for season in SEASONS:
        print(f"\n=== Processing season {season} (vaastav) ===")

        # players
        players_raw = fetch_players_raw_for_season(season)
        save_raw_snapshot_csv(players_raw, season, "players_raw.csv")

        players_clean = build_players_df(players_raw, teams_lookup, season)
        save_players_csv(players_clean, season)
        all_players.append(players_clean)

        # GW-level
        gw_raw = fetch_gws_for_season(season)
        save_raw_snapshot_csv(gw_raw, season, "gw_raw_stack.csv")

        gw_history = build_gw_history_df(gw_raw, players_raw, teams_lookup, season)
        save_gw_history_csv(gw_history, season)
        all_gw.append(gw_history)

    # 2) Current / new season (2025-26) from official FPL API
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

    # 3) Combine everything (2016-17 -> 2025-26)
    combined_players = pd.concat(all_players, ignore_index=True)
    save_combined_players_csv(combined_players)

    combined_gw = pd.concat(all_gw, ignore_index=True)
    save_combined_gw_history_csv(combined_gw)

    print("\nDone.")
    print(f"Seasons processed: {len(SEASONS) + 1} (including {CURRENT_SEASON})")
    print(f"Total player rows: {len(combined_players)}")
    print(f"Total GW rows    : {len(combined_gw)}")


if __name__ == "__main__":
    main()