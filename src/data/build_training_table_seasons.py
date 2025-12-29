"""
build_training_table_seasons.py

Build raw+clean season datasets (players & gw_history) across seasons,
then build ML-ready feature tables from the combined gw_history.

Seasons:
- 2016-17 .. 2024-25 from vaastav repo
- 2025-26 from official FPL API (bootstrap-static + event/{gw}/live)

Outputs (data/processed):
- <season>_players_clean.csv
- <season>_gw_history.csv
- players_clean_all_seasons.csv
- gw_history_all_seasons.csv

PLUS (ML-ready feature tables):
- player_gw_features.csv
- team_gw_features.csv
- fixture_gw_features.csv
- player_gw_training.csv   (features only; target is built in train_compare_models.py)
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

GW_HISTORY_CSV = PROC_DIR / "gw_history_all_seasons.csv"
PLAYERS_ALL_CSV = PROC_DIR / "players_clean_all_seasons.csv"  # used to fix team_id gaps


# -------------------------------------------------------------------
# Which extra per-match stats try to preserve in gw_history
# (if present in the raw season gwX.csv or API response)
# -------------------------------------------------------------------
EXTRA_STAT_COLS = [
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
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "ea_index",
    "xP",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "starts",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
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
    "fixture",
    "kickoff_time",
    "kickoff_time_formatted",
    "team_a_score",
    "team_h_score",
    "was_home",
    "opponent_team",
    "loaned_in",
    "loaned_out",
    "value",
    "in_dreamteam",
    "modified",
]

# -------------------------------------------------------------------
# HELPERS (DATA FETCHING)
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


def fetch_players_raw_for_season(season: str) -> pd.DataFrame:
    url = f"{FPL_GITHUB_BASE}/{season}/players_raw.csv"
    return read_csv_robust(url)


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

    return pd.concat(all_gws, ignore_index=True, sort=False)


def fetch_master_team_list() -> pd.DataFrame:
    url = f"{FPL_GITHUB_BASE}/master_team_list.csv"
    df = read_csv_robust(url)
    df.rename(columns={"team": "team_id"}, inplace=True)
    return standardize_team_id(df, "team_id")


def save_raw_snapshot_csv(df: pd.DataFrame, season: str, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved raw snapshot to: {path}")
    return path


# -------------------------------------------------------------------
# OFFICIAL FPL API FETCHING (CURRENT_SEASON)
# -------------------------------------------------------------------
def fetch_bootstrap_static() -> dict:
    url = f"{FPL_API_BASE}/bootstrap-static/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_players_raw_from_bootstrap(data: dict) -> pd.DataFrame:
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
            },
            inplace=True,
        )

        gw_df["season"] = CURRENT_SEASON
        all_gws.append(gw_df)
        print(f"[{CURRENT_SEASON}] Loaded GW{gw} with {len(gw_df)} rows")

    if not all_gws:
        raise RuntimeError(f"No gameweeks found for {CURRENT_SEASON} via FPL API")

    return pd.concat(all_gws, ignore_index=True, sort=False)


# -------------------------------------------------------------------
# CLEAN PLAYER DF (SEASON SUMMARY)
# -------------------------------------------------------------------
def build_players_df(players: pd.DataFrame, teams_lookup: pd.DataFrame, season: str) -> pd.DataFrame:
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

    numeric_cols = ["total_points", "minutes", "form", "points_per_game", "selected_by_percent", "price"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df[numeric_cols] = df[numeric_cols].round(3)

    price_nonzero = df["price"].replace(0, np.nan)
    minutes_nonzero_90 = df["minutes"].replace(0, np.nan) / 90.0

    df["points_per_million"] = df["total_points"] / price_nonzero
    df["points_per_90"] = df["total_points"] / minutes_nonzero_90

    for col in ["form", "points_per_game", "selected_by_percent", "price", "points_per_million", "points_per_90"]:
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
    return df.reindex(columns=cols_final)


# -------------------------------------------------------------------
# CLEAN GW HISTORY (PLAYER x GW)
# -------------------------------------------------------------------
def build_gw_history_df(
    gw_raw: pd.DataFrame,
    players_raw: pd.DataFrame,
    teams_lookup: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
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

    return df.reindex(columns=cols_final).copy()


# -------------------------------------------------------------------
# SAVING (BASE DATA)
# -------------------------------------------------------------------
def save_players_csv(df: pd.DataFrame, season: str, name: str = "players_clean.csv") -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / f"{season}_{name}"
    df.to_csv(path, index=False)
    print(f"[{season}] Saved cleaned players to: {path}")
    return path


def save_gw_history_csv(df: pd.DataFrame, season: str, name: str = "gw_history.csv") -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / f"{season}_{name}"
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


# ===================================================================
#  ML FEATURE BUILD PART
# ===================================================================

def load_gw_history_for_features() -> pd.DataFrame:
    """
    Load gw_history_all_seasons.csv and:
      - type cleanup
      - robust normalization of was_home
      - fix missing team_id/team_name for 2020-21..2024-25 via players_clean_all_seasons.csv
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
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
        }
        df["was_home"] = df["was_home"].map(mapping)

    recent_seasons = {"2020-21", "2021-22", "2022-23", "2023-24", "2024-25"}
    if PLAYERS_ALL_CSV.exists():
        players_all = pd.read_csv(PLAYERS_ALL_CSV, low_memory=False)
        players_all["season"] = players_all["season"].astype(str)
        players_all["player_id"] = pd.to_numeric(players_all["player_id"], errors="coerce")

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
            df["team_id"] = df["team_id"].where(~df["team_id"].isna(), df["team_id_p"])
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
        s_non_null = s.dropna()
        return s_non_null.iloc[0] if len(s_non_null) > 0 else np.nan

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
        raise ValueError(f"team_gw missing required columns: {missing_required}")

    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df = df[~df["team_id"].isna() & ~df["gameweek"].isna()].copy()

    if "opponent_team" not in df.columns:
        print("No 'opponent_team' column in team_gw; returning unchanged.")
        return df

    df["opponent_team"] = pd.to_numeric(df["opponent_team"], errors="coerce")
    if df["opponent_team"].isna().all():
        print("All opponent_team are NaN; returning unchanged (no opponent features).")
        return df

    team_feature_cols = [
        c for c in df.columns
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


def build_player_training_table(player_gw: pd.DataFrame, fixture_gw: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["season", "gameweek", "team_id"]
    merge_cols = [c for c in merge_cols if c in player_gw.columns and c in fixture_gw.columns]

    fixture_cols = [
        c for c in fixture_gw.columns
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


def build_and_save_ml_feature_tables() -> None:
    """
    Build ML-ready feature tables from the already-saved gw_history_all_seasons.csv
    and write them to data/processed/.
    """
    if not GW_HISTORY_CSV.exists():
        raise FileNotFoundError(
            f"Missing {GW_HISTORY_CSV}. Run the data build first."
        )

    print("Loading gw_history_all_seasons.csv for feature build ...")
    gw = load_gw_history_for_features()

    print("Building player GW features ...")
    player_gw = build_player_gw_features(gw)
    player_gw_path = PROC_DIR / "player_gw_features.csv"
    player_gw.to_csv(player_gw_path, index=False)
    print(f"Saved: {player_gw_path}")

    print("Building team GW features ...")
    team_gw = build_team_gw_features(gw)
    team_gw_path = PROC_DIR / "team_gw_features.csv"
    team_gw.to_csv(team_gw_path, index=False)
    print(f"Saved: {team_gw_path}")

    print("Building fixture GW features ...")
    fixture_gw = build_fixture_gw_features(team_gw)
    fixture_gw_path = PROC_DIR / "fixture_gw_features.csv"
    fixture_gw.to_csv(fixture_gw_path, index=False)
    print(f"Saved: {fixture_gw_path}")

    print("Building player training table (features only, no target) ...")
    training_df = build_player_training_table(player_gw, fixture_gw)
    training_path = PROC_DIR / "player_gw_training.csv"
    training_df.to_csv(training_path, index=False)
    print(f"Saved: {training_path}")


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

    # 2) Current season (2025-26) from official FPL API
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

    # 4) Build ML feature tables 
    print("\n=== Building ML-ready feature tables ===")
    build_and_save_ml_feature_tables()

    print("\nDone.")
    print(f"Seasons processed: {len(SEASONS) + 1} (including {CURRENT_SEASON})")
    print(f"Total player rows: {len(combined_players)}")
    print(f"Total GW rows    : {len(combined_gw)}")


if __name__ == "__main__":
    main()