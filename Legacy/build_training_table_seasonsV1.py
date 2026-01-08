"""
Build ML-ready feature tables from gw_history_all_seasons.csv.

Outputs in data/processed:
- player_gw_features.csv
- team_gw_features.csv
- fixture_gw_features.csv
- player_gw_training.csv 
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"

GW_HISTORY_CSV = PROC_DIR / "gw_history_all_seasons.csv"
PLAYERS_ALL_CSV = PROC_DIR / "players_clean_all_seasons.csv"  # used to fix team_id gaps


# -------------------------------------------------------------------
# LOAD & FIX GW HISTORY
# -------------------------------------------------------------------
def load_gw_history() -> pd.DataFrame:
    """
    Load gw_history_all_seasons.csv and perform:
      - basic type cleanup
      - robust normalization of was_home
      - fix missing team_id/team_name for 2020-21..2024-25
        by joining with players_clean_all_seasons.csv on (season, player_id)
    """
    # low_memory=False to avoid mixed-type DtypeWarnings
    df = pd.read_csv(GW_HISTORY_CSV, low_memory=False)

    # Basic normalization
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

    # Robust was_home handling:
    # supports bools, "True"/"False", 1/0, "1"/"0"
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
    

    # ------------------------------------------------------------------
    # FIX team_id/team_name for 2020-21..2024-25 using players_clean_all_seasons
    # ------------------------------------------------------------------
    recent_seasons = {"2020-21", "2021-22", "2022-23", "2023-24", "2024-25"}

    if PLAYERS_ALL_CSV.exists():
        players_all = pd.read_csv(PLAYERS_ALL_CSV)
        players_all["season"] = players_all["season"].astype(str)
        players_all["player_id"] = pd.to_numeric(
            players_all["player_id"], errors="coerce"
        )

        # Restrict to recent seasons where gaps were a problem
        players_sub = players_all[
            players_all["season"].isin(recent_seasons)
        ][["season", "player_id", "team_id", "team_name"]].copy()

        # Merge onto gw history (this can overwrite or fill nulls)
        df = df.merge(
            players_sub,
            on=["season", "player_id"],
            how="left",
            suffixes=("", "_p"),
        )

        # Coalesce original vs players-based for team_id / team_name
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

        # Ensure numeric type for team_id
        if "team_id" in df.columns:
            df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")

    else:
        print(
            f"WARNING: {PLAYERS_ALL_CSV} not found. "
            "Cannot repair team_id/team_name gaps for 2020-21..2024-25."
        )

    return df


# -------------------------------------------------------------------
# ROLLING FEATURE HELPER
# -------------------------------------------------------------------
def rolling_features(group: pd.DataFrame, value_col: str, windows=(3, 5), prefix=""):
    """
    Compute lagged rolling features on a sorted group (by gameweek).
    Returns a DataFrame with new columns added.
    """
    group = group.sort_values("gameweek").copy()

    # simple last GW
    group[f"{prefix}{value_col}_lag1"] = group[value_col].shift(1)

    for w in windows:
        roll = (
            group[value_col]
            .rolling(window=w, min_periods=1)
            .mean()
            .shift(1)  # use ONLY past gameweeks
        )
        group[f"{prefix}{value_col}_avg_{w}"] = roll

    return group


# -------------------------------------------------------------------
# PLAYER-LEVEL FEATURES
# -------------------------------------------------------------------
def build_player_gw_features(gw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, player_id, gameweek) with rolling form features.

    Uses "universal" columns that exist across seasons:
    - minutes, total_points, price, opponent_team, was_home, etc.
    """
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

    # compute rolling features per (season, player_id)
    def player_group_func(g):
        g = rolling_features(g, "total_points", windows=(3, 5), prefix="p_")
        g = rolling_features(g, "minutes", windows=(3, 5), prefix="p_")

        # cumulative stats BEFORE this GW
        g["p_cum_points_before"] = g["total_points"].cumsum().shift(1)
        g["p_games_played_before"] = (g["minutes"] > 0).astype(int).cumsum().shift(1)

        return g

    df = df.groupby(["season", "player_id"], group_keys=False).apply(player_group_func)

    # fill NaNs in features (but keep price/minutes/total_points NaN if missing)
    feature_cols = [c for c in df.columns if c.startswith("p_")]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Deduplicate if needed
    df = df.drop_duplicates(subset=["season", "gameweek", "player_id"])

    return df


# -------------------------------------------------------------------
# TEAM-LEVEL FEATURES
# -------------------------------------------------------------------
def build_team_gw_features(gw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, team_id, gameweek) with rolling team strength features.

    Safeguards:
    - Drop rows with missing team_id before aggregation.
    - Aggregate on (season, gameweek, team_id, team_name) ONLY.
    - Attach opponent_team / was_home as first *non-null* values per team+GW,
      instead of including them in the group key.
    """
    df = gw.copy()

    # Drop rows without team_id
    if "team_id" not in df.columns:
        raise ValueError("gw history must contain 'team_id' to build team features.")

    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df = df[~df["team_id"].isna()].copy()

    # Basic grouping columns
    group_cols = ["season", "gameweek", "team_id"]
    if "team_name" in df.columns:
        group_cols.append("team_name")

    # Aggregate core team-level stats
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            team_total_points=("total_points", "sum"),
            team_total_minutes=("minutes", "sum"),
            team_avg_price=("price", "mean"),
        )
        .reset_index()
    )

    # Helper: first non-null value
    def first_non_null(s: pd.Series):
        s_non_null = s.dropna()
        return s_non_null.iloc[0] if len(s_non_null) > 0 else np.nan

    # Attach stable opponent_team / was_home per team+GW (if available)
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

    # Sort for rolling features
    agg = agg.sort_values(["season", "team_id", "gameweek"])

    def team_group_func(g):
        g = rolling_features(g, "team_total_points", windows=(3, 5), prefix="t_")
        g = rolling_features(g, "team_total_minutes", windows=(3, 5), prefix="t_")

        # basic "strength" proxy
        g["t_points_per_90"] = (
            g["team_total_points"]
            / (g["team_total_minutes"] / 90.0).replace(0, np.nan)
        )
        g["t_points_per_90"] = g["t_points_per_90"].fillna(0.0)

        return g

    agg = agg.groupby(["season", "team_id"], group_keys=False).apply(team_group_func)

    # fill NaNs in team features
    feat_cols = [c for c in agg.columns if c.startswith("t_")]
    agg[feat_cols] = agg[feat_cols].fillna(0.0)

    return agg


# -------------------------------------------------------------------
# FIXTURE-LEVEL FEATURES (TEAM + OPPONENT)
# -------------------------------------------------------------------
def build_fixture_gw_features(team_gw: pd.DataFrame) -> pd.DataFrame:
    """
    For each (season, gameweek, team_id), attach opponent team features.
    Produces one row per team per GW with both team and opponent strength.

    Safeguards:
    - Require valid team_id / gameweek.
    - If opponent_team is missing or entirely NaN, return team_gw unchanged
      (no fake "zero opponent" features).
    """
    df = team_gw.copy()

    # Ensure required columns exist
    required = {"season", "gameweek", "team_id"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(
            f"team_gw is missing required columns: {missing_required}"
        )

    # Normalize IDs
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")

    # Drop rows without a valid team_id or gameweek
    df = df[~df["team_id"].isna() & ~df["gameweek"].isna()].copy()

    # Opponent info is optional; if not present or all NaN, we just return df as-is
    if "opponent_team" not in df.columns:
        print("No 'opponent_team' column in team_gw; returning team_gw unchanged.")
        return df

    df["opponent_team"] = pd.to_numeric(df["opponent_team"], errors="coerce")
    if df["opponent_team"].isna().all():
        print(
            "All 'opponent_team' values are NaN; returning team_gw unchanged "
            "without opponent features."
        )
        return df

    # features for own team
    team_feature_cols = [
        c
        for c in df.columns
        if c.startswith("t_") or c in ["team_total_points", "team_total_minutes"]
    ]

    # Build opponent table from valid team rows
    opp = df[["season", "gameweek", "team_id"] + team_feature_cols].copy()
    opp = opp.rename(columns={"team_id": "opponent_team"})
    opp = opp.rename(
        columns={col: f"opp_{col}" for col in team_feature_cols}
    )

    # Merge opponent features back
    merged = df.merge(
        opp,
        on=["season", "gameweek", "opponent_team"],
        how="left",
    )

    # fill opponent feature NaNs
    opp_cols = [c for c in merged.columns if c.startswith("opp_")]
    merged[opp_cols] = merged[opp_cols].fillna(0.0)

    return merged


# -------------------------------------------------------------------
# BUILD TRAINING TABLE (PLAYER FEATURES + TEAM & OPP)
# -------------------------------------------------------------------
def build_player_training_table(
    player_gw: pd.DataFrame, fixture_gw: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge player-level and fixture-level features.

    This version NO LONGER creates `target_next_points`.
    It only outputs features + original gw columns (including total_points).
    The target will be defined at training time, outside this script.
    """
    # merge fixture info onto player rows
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

    # fill NaNs from fixture features
    fx_cols = [c for c in df.columns if c.startswith("t_") or c.startswith("opp_")]
    df[fx_cols] = df[fx_cols].fillna(0.0)

    # Ensure unique rows per (season, gameweek, player_id)
    if {"season", "gameweek", "player_id"}.issubset(df.columns):
        df = df.sort_values(["season", "player_id", "gameweek"])
        df = df.drop_duplicates(subset=["season", "gameweek", "player_id"])

    # NOTE: NO target_next_points here anymore.
    return df


# -------------------------------------------------------------------
# SANITY CHECKS FOR OUTPUT TABLES
# -------------------------------------------------------------------
def check_for_duplicates(df, keys, name="DataFrame"):
    """
    Print duplicates for specified key columns.
    """
    if not set(keys).issubset(df.columns):
        print(f"[{name}] Missing required columns for duplicate check: {keys}")
        return

    dups = df[df.duplicated(subset=keys, keep=False)]
    if len(dups) > 0:
        print(f"\n⚠️  [{name}] DUPLICATES FOUND for keys {keys}: {len(dups)} rows")
        print(dups[keys].head())
    else:
        print(f"[{name}] No duplicates for keys {keys}.")


def check_missing_ids(df, cols, name="DataFrame"):
    """
    Report missing identifiers such as team_id or player_id.
    """
    for col in cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"⚠️  [{name}] {missing} rows have missing {col}")
        else:
            print(f"[{name}] Column {col} not present.")


def check_was_home_distribution(df, name="DataFrame"):
    """
    Show distribution of home/away flags to detect parsing issues.
    """
    if "was_home" not in df.columns:
        print(f"[{name}] No was_home column.")
        return
    print(f"\n[{name}] was_home distribution:")
    print(df["was_home"].value_counts(dropna=False))


def check_numeric_nan(df, cols, name="DataFrame"):
    """
    For numeric columns, report number of NaNs.
    """
    for col in cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                print(f"⚠️  [{name}] {n_missing} NaNs in numeric column '{col}'")
        else:
            print(f"[{name}] Column {col} not found.")


def sanity_check_outputs(player_gw, team_gw, fixture_gw, training_df):
    """
    Run a complete diagnostic scan on all generated tables.
    """

    print("\n==============================")
    print("🔍 RUNNING SANITY CHECKS")
    print("==============================\n")

    # --------------------------
    # PLAYER_GW FEATURES
    # --------------------------
    print("\n### PLAYER_GW_FEATURES ###\n")
    check_for_duplicates(
        player_gw, ["season", "gameweek", "player_id"], "player_gw"
    )
    check_missing_ids(player_gw, ["player_id", "team_id"], "player_gw")
    check_numeric_nan(
        player_gw,
        ["minutes", "total_points", "price"],
        "player_gw",
    )
    check_was_home_distribution(player_gw, "player_gw")

    # --------------------------
    # TEAM_GW FEATURES
    # --------------------------
    print("\n### TEAM_GW_FEATURES ###\n")
    check_for_duplicates(
        team_gw, ["season", "gameweek", "team_id"], "team_gw"
    )
    check_missing_ids(team_gw, ["team_id", "opponent_team"], "team_gw")
    check_numeric_nan(
        team_gw,
        ["team_total_points", "team_total_minutes"],
        "team_gw",
    )

    # --------------------------
    # FIXTURE_GW FEATURES
    # --------------------------
    print("\n### FIXTURE_GW_FEATURES ###\n")
    check_for_duplicates(
        fixture_gw, ["season", "gameweek", "team_id"], "fixture_gw"
    )
    check_missing_ids(
        fixture_gw, ["team_id", "opponent_team"], "fixture_gw"
    )

    # --------------------------
    # PLAYER_GW_TRAINING
    # --------------------------
    print("\n### PLAYER_GW_TRAINING ###\n")
    check_for_duplicates(
        training_df, ["season", "gameweek", "player_id"], "player_gw_training"
    )
    check_missing_ids(
        training_df, ["player_id", "team_id"], "player_gw_training"
    )
    check_numeric_nan(
        training_df,
        ["minutes", "total_points", "price"],
        "player_gw_training",
    )

    print("\n🔎 COMPLETED SANITY CHECKS\n")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
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

    print("\nDone. Shapes:")
    print("  player_gw_features :", player_gw.shape)
    print("  team_gw_features   :", team_gw.shape)
    print("  fixture_gw_features:", fixture_gw.shape)
    print("  player_gw_training :", training_df.shape)

    # Run sanity checks at the end
    sanity_check_outputs(player_gw, team_gw, fixture_gw, training_df)


if __name__ == "__main__":
    main()