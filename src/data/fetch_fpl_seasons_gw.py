"""
Build clean player-level and gameweek-level datasets from vaastav FPL repo
for seasons 2016-17 to 2025-26.

Repo-first, with FPL API fallback for missing 2025-26 gameweeks.

Outputs:
- data/processed/<season>_players_clean.csv
- data/processed/<season>_gw_history.csv
- data/processed/players_clean_all_seasons.csv
- data/processed/gw_history_all_seasons.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
FPL_GITHUB_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)
FPL_API_BASE = "https://fantasy.premierleague.com/api"

HISTORICAL_SEASONS = [
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
SEASONS = HISTORICAL_SEASONS + [CURRENT_SEASON]

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# -------------------------------------------------------------------
# EXTRA GW-LEVEL STAT COLUMNS (superset across seasons)
# -------------------------------------------------------------------
GW_STAT_COLS = [
    # "classic" per-GW stats (2016-18)
    "assists",
    "attempted_passes",
    "big_chances_created",
    "big_chances_missed",
    "bonus",
    "bps",
    "clean_sheets",
    "clearances_blocks_interceptions",
    "completed_passes",
    "creativity",
    "dribbles",
    "ea_index",
    "errors_leading_to_goal",
    "errors_leading_to_goal_attempt",
    "fixture",
    "fouls",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "key_passes",
    "loaned_in",
    "loaned_out",
    "offside",
    "open_play_crosses",
    "opponent_team",
    "own_goals",
    "penalties_conceded",
    "penalties_missed",
    "penalties_saved",
    "recoveries",
    "red_cards",
    "round",
    "saves",
    "selected",
    "tackled",
    "tackles",
    "target_missed",
    "team_a_score",
    "team_h_score",
    "threat",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "was_home",
    "winning_goals",
    "yellow_cards",
    # xG era extras (2020-21+)
    "xP",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "starts",
    "modified",
    "defensive_contribution",
]

# numeric subset we want to coerce explicitly
GW_NUMERIC_COLS = [
    "assists",
    "attempted_passes",
    "big_chances_created",
    "big_chances_missed",
    "bonus",
    "bps",
    "clean_sheets",
    "clearances_blocks_interceptions",
    "completed_passes",
    "creativity",
    "dribbles",
    "ea_index",
    "errors_leading_to_goal",
    "errors_leading_to_goal_attempt",
    "fixture",
    "fouls",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "key_passes",
    "loaned_in",
    "loaned_out",
    "minutes",
    "offside",
    "open_play_crosses",
    "opponent_team",
    "own_goals",
    "penalties_conceded",
    "penalties_missed",
    "penalties_saved",
    "recoveries",
    "red_cards",
    "round",
    "saves",
    "selected",
    "tackled",
    "tackles",
    "target_missed",
    "team_a_score",
    "team_h_score",
    "threat",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "winning_goals",
    "yellow_cards",
    "xP",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "starts",
    "defensive_contribution",
]


# -------------------------------------------------------------------
# GENERIC HELPERS
# -------------------------------------------------------------------
def standardize_team_id(df: pd.DataFrame, col: str = "team_id") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def read_csv_robust(url: str, **kwargs) -> pd.DataFrame:
    """
    Robust CSV reader for GitHub raw files.
    First try normal read_csv; on Unicode errors, fall back to latin1 + engine='python'.
    """
    try:
        return pd.read_csv(url, **kwargs)
    except UnicodeDecodeError:
        kwargs.setdefault("encoding", "latin1")
        kwargs.setdefault("engine", "python")
        kwargs.pop("low_memory", None)
        return pd.read_csv(url, **kwargs)


def save_raw_snapshot_csv(df: pd.DataFrame, season: str, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved raw snapshot to: {path}")
    return path


# -------------------------------------------------------------------
# FETCH FROM GITHUB
# -------------------------------------------------------------------
def fetch_players_raw_for_season(season: str) -> pd.DataFrame:
    url = f"{FPL_GITHUB_BASE}/{season}/players_raw.csv"
    df = read_csv_robust(url)
    return df


def fetch_master_team_list() -> pd.DataFrame:
    url = f"{FPL_GITHUB_BASE}/master_team_list.csv"
    df = read_csv_robust(url)
    df.rename(columns={"team": "team_id"}, inplace=True)
    df = standardize_team_id(df, "team_id")
    return df


# -------------------------------------------------------------------
# FPL API HELPERS (for CURRENT_SEASON fallback)
# -------------------------------------------------------------------
def fetch_bootstrap_static() -> dict:
    resp = requests.get(f"{FPL_API_BASE}/bootstrap-static/", timeout=10)
    resp.raise_for_status()
    return resp.json()


def augment_team_list_from_bootstrap(
    teams_lookup: pd.DataFrame, bootstrap_data: dict, season: str
) -> pd.DataFrame:
    """
    For CURRENT_SEASON, master_team_list often lacks rows.
    Augment with team_id/team_name from bootstrap-static.
    """
    teams_cur = pd.DataFrame(bootstrap_data["teams"])[["id", "name"]].rename(
        columns={"id": "team_id", "name": "team_name"}
    )
    teams_cur["season"] = season
    teams_cur = standardize_team_id(teams_cur, "team_id")

    combined = pd.concat([teams_lookup, teams_cur], ignore_index=True)
    return combined


def fetch_gw_from_api(season: str, gw: int) -> pd.DataFrame:
    """
    Fetch a single gameweek from FPL API for CURRENT_SEASON as fallback.

    We only reconstruct:
    - element (player_id)
    - minutes
    - total_points

    Other columns will be NaN for these rows.
    """
    if season != CURRENT_SEASON:
        raise ValueError("API fallback is only used for CURRENT_SEASON")

    url = f"{FPL_API_BASE}/event/{gw}/live/"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"API GW{gw} HTTP {resp.status_code}")

    data = resp.json()
    rows = []
    for el in data.get("elements", []):
        pid = el["id"]
        stats = el.get("stats", {}) or {}
        rows.append(
            {
                "element": pid,
                "minutes": stats.get("minutes", 0),
                "total_points": stats.get("total_points", 0),
            }
        )

    df = pd.DataFrame(rows)
    print(f"[{season}] Loaded gw{gw} from API with {len(df)} rows")
    return df


# -------------------------------------------------------------------
# GW FETCHING: STACK gwX.csv, API FALLBACK FOR 2025-26
# -------------------------------------------------------------------
def fetch_gws_stack_for_season(season: str, max_gw: int = 60) -> pd.DataFrame:
    """
    For a given season, read gws/gw1.csv, gw2.csv, ... and stack them.

    RULES:
    - For CURRENT_SEASON (2025-26):
        * Hard cap at GW9.
        * For each GW in 1..9: try repo, if 404 then try API.
        * If BOTH repo and API are missing a GW -> raise error.
    - For older seasons:
        * Read GW1..GWk until the first missing repo file.
        * Ensure no gaps in 1..k.
    """

    # ----- HARD CAP FOR CURRENT SEASON -----
    if season == CURRENT_SEASON:
        max_gw = 9
        print(f"[{season}] HARD CAP: expecting GW1–GW9 (repo + API fallback)")

    all_gws = []
    found_gws = set()

    for gw in range(1, max_gw + 1):
        url = f"{FPL_GITHUB_BASE}/{season}/gws/gw{gw}.csv"

        # try repo first
        try:
            df = read_csv_robust(
                url,
                engine="python",
                on_bad_lines="skip",
            )
            print(f"[{season}] Loaded GW{gw} from repo with {len(df)} rows")
        except Exception as e_repo:
            if season == CURRENT_SEASON:
                # API fallback for current season
                try:
                    df = fetch_gw_from_api(season, gw)
                    print(
                        f"[{season}] Loaded GW{gw} from API fallback with {len(df)} rows"
                    )
                except Exception as e_api:
                    raise RuntimeError(
                        f"[{season}] ERROR: Missing GW{gw} in repo AND API.\n"
                        f"2025-26 must contain all GWs 1–9.\n"
                        f"Repo error: {e_repo}\nAPI error: {e_api}"
                    )
            else:
                # historical season: first missing repo file = natural end
                print(f"[{season}] Repo missing GW{gw}. Stopping season here.")
                break

        df["season"] = season
        df["gameweek"] = gw
        all_gws.append(df)
        found_gws.add(gw)

    # ---------- VALIDATION ----------
    if not all_gws:
        raise RuntimeError(f"[{season}] ERROR: No GW data found at all.")

    if season == CURRENT_SEASON:
        # For 2025-26 we require a full block GW1..9 (repo OR API)
        expected = set(range(1, 10))  # 1..9
        missing = expected - found_gws
        if missing:
            raise RuntimeError(
                f"[{season}] ERROR: Missing GWs {sorted(missing)}. "
                "2025-26 must contain GW1–GW9 (from repo or API)."
            )
        print(f"[{season}] All required GWs 1–9 present (repo+API).")
    else:
        # older seasons: ensure no gaps 1..max_found
        max_found = max(found_gws)
        expected = set(range(1, max_found + 1))
        missing = expected - found_gws
        if missing:
            raise RuntimeError(
                f"[{season}] ERROR: Season has gaps. Missing GWs {sorted(missing)} "
                f"in 1..{max_found}."
            )

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

    df.drop(
        columns=["first_name", "last_name", "price_tenths"], inplace=True, errors="ignore"
    )

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
    Robust GW history builder handling all schemas from 2016-17 to 2025-26.

    - Normalizes IDs: element -> player_id, round -> gameweek.
    - Ignores name/position/team in gw_raw and rebuilds them from players_raw.
    - Keeps a rich set of per-GW stats (assists, bonus, ict_index, xG, etc.)
      whenever available in the original gw files.
    - Combines with master_team_list (plus bootstrap augmentation for 2025-26).
    """
    df = gw_raw.copy()

    # --- normalize IDs ---
    if "element" in df.columns:
        df.rename(columns={"element": "player_id"}, inplace=True)
    if "id" in df.columns and "player_id" not in df.columns:
        # in some schemas 'id' is the row id of the GW entry; we prefer element,
        # but we keep this fallback only if 'element' is not present
        df.rename(columns={"id": "player_id"}, inplace=True)

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    if "GW" in df.columns:
        df.rename(columns={"GW": "gameweek"}, inplace=True)
    elif "round" in df.columns and "gameweek" not in df.columns:
        df.rename(columns={"round": "gameweek"}, inplace=True)

    # price from 'value'
    if "value" in df.columns:
        df["price"] = df["value"] / 10.0
    elif "price" not in df.columns:
        df["price"] = np.nan

    # ensure gameweek present
    if "gameweek" not in df.columns and "round" in df.columns:
        df.rename(columns={"round": "gameweek"}, inplace=True)

    df["season"] = season

    # Drop any existing name/position/team columns — we rebuild them from players_raw
    for col in ["name", "position", "team"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- build meta from players_raw: name, element_type, team_id ---
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

    # merge GW rows with meta on player_id
    df = df.merge(meta, on="player_id", how="left")

    # rename for clarity
    df.rename(columns={"team_id": "team_id_meta"}, inplace=True)

    # final team_id comes from players_raw only (constant for season)
    df["team_id"] = df["team_id_meta"]

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

    # drop helper
    if "team_id_meta" in df.columns:
        df.drop(columns=["team_id_meta"], inplace=True)

    # --- final column ordering ---
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

    # extra stat columns we keep if present
    stat_cols_present = [c for c in GW_STAT_COLS if c in df.columns]

    # avoid duplicates: 'total_points', 'minutes', 'opponent_team', 'value', etc.
    stat_cols_present = [
        c for c in stat_cols_present if c not in {"total_points", "minutes", "price"}
    ]

    cols_final = [c for c in cols_base if c in df.columns] + stat_cols_present
    df = df.reindex(columns=cols_final).copy()

    # --- numeric cleanup for known numeric stats ---
    for col in GW_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # was_home as bool if present
    if "was_home" in df.columns:
        # some seasons encode was_home as bool/int/string; normalize to bool
        df["was_home"] = df["was_home"].astype(str).str.lower().isin(
            ["true", "1", "yes", "y"]
        )

    print(f"[{season}] GW history columns: {list(df.columns)}")
    return df


# -------------------------------------------------------------------
# SAVING HELPERS
# -------------------------------------------------------------------
def save_players_csv(
    df: pd.DataFrame, season: str, name: str = "players_clean.csv"
) -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = PROC_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved cleaned players to: {path}")
    return path


def save_gw_history_csv(
    df: pd.DataFrame, season: str, name: str = "gw_history.csv"
) -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{season}_{name}"
    path = PROC_DIR / filename
    df.to_csv(path, index=False)
    print(f"[{season}] Saved GW history to: {path}")
    return path


def save_combined_players_csv(
    df: pd.DataFrame, name: str = "players_clean_all_seasons.csv"
) -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved combined players dataset ({len(df)} rows) to: {path}")
    return path


def save_combined_gw_history_csv(
    df: pd.DataFrame, name: str = "gw_history_all_seasons.csv"
) -> Path:
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

    # 1) Historical seasons (repo only)
    for season in HISTORICAL_SEASONS:
        print(f"\n=== Processing season {season} (repo) ===")

        players_raw = fetch_players_raw_for_season(season)
        save_raw_snapshot_csv(players_raw, season, "players_raw.csv")

        players_clean = build_players_df(players_raw, teams_lookup, season)
        save_players_csv(players_clean, season)
        all_players.append(players_clean)

        gw_raw = fetch_gws_stack_for_season(season)
        save_raw_snapshot_csv(gw_raw, season, "gw_stack.csv")

        gw_history = build_gw_history_df(gw_raw, players_raw, teams_lookup, season)
        save_gw_history_csv(gw_history, season)
        all_gw.append(gw_history)

    # 2) Current season (repo + API fallback for missing GWs)
    print(f"\n=== Processing season {CURRENT_SEASON} (repo + API fallback) ===")

    bootstrap = fetch_bootstrap_static()
    teams_lookup_aug = augment_team_list_from_bootstrap(
        teams_lookup, bootstrap, CURRENT_SEASON
    )

    players_raw_cur = fetch_players_raw_for_season(CURRENT_SEASON)
    save_raw_snapshot_csv(players_raw_cur, CURRENT_SEASON, "players_raw.csv")

    players_clean_cur = build_players_df(
        players_raw_cur, teams_lookup_aug, CURRENT_SEASON
    )
    save_players_csv(players_clean_cur, CURRENT_SEASON)
    all_players.append(players_clean_cur)

    gw_raw_cur = fetch_gws_stack_for_season(CURRENT_SEASON)
    save_raw_snapshot_csv(gw_raw_cur, CURRENT_SEASON, "gw_stack.csv")

    gw_history_cur = build_gw_history_df(
        gw_raw_cur, players_raw_cur, teams_lookup_aug, CURRENT_SEASON
    )
    save_gw_history_csv(gw_history_cur, CURRENT_SEASON)
    all_gw.append(gw_history_cur)

    # 3) Combine all seasons
    combined_players = pd.concat(all_players, ignore_index=True)
    save_combined_players_csv(combined_players)

    combined_gw = pd.concat(all_gw, ignore_index=True)
    save_combined_gw_history_csv(combined_gw)

    print("\nDone.")
    print(f"Seasons processed: {len(SEASONS)}")
    print(f"Total player rows: {len(combined_players)}")
    print(f"Total GW rows    : {len(combined_gw)}")


if __name__ == "__main__":
    main()
