import sys
from pathlib import Path
from typing import List, Tuple

import argparse  # NEW
import joblib
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.build_training_table import (
    load_players,
    fetch_player_history,
)

BASE_URL = "https://fantasy.premierleague.com/api"

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "ml_models"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "ml_predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

PLAYERS_CSV = DATA_DIR / "players_clean.csv"
MODEL_PATH = MODELS_DIR / "rf_points_model.pkl"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_next_gameweek() -> int:
    """
    Use bootstrap-static to find the next (unplayed) gameweek.
    """
    resp = requests.get(f"{BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    events = resp.json()["events"]

    future_events = [e for e in events if not e.get("finished")]
    if not future_events:
        raise RuntimeError(
            "No future gameweeks found in bootstrap-static (season may be finished)"
        )

    next_gw = min(e["id"] for e in future_events)
    return int(next_gw)


def get_gw_fixtures(gw: int) -> pd.DataFrame:
    """
    Fetch fixtures and return a table for a given GW with:
        gw, team_id, opponent_team, was_home, fixture_difficulty
    (both home and away perspective for each match).
    """
    resp = requests.get(f"{BASE_URL}/fixtures/")
    resp.raise_for_status()
    fixtures = resp.json()
    if not fixtures:
        return pd.DataFrame()

    fdf = pd.DataFrame(fixtures)
    fdf = fdf[fdf["event"] == gw].copy()

    if fdf.empty:
        return pd.DataFrame()

    rows = []
    for _, r in fdf.iterrows():
        gw_num = int(r["event"])
        team_h = int(r["team_h"])
        team_a = int(r["team_a"])
        diff_h = r.get("team_h_difficulty", np.nan)
        diff_a = r.get("team_a_difficulty", np.nan)

        # home perspective
        rows.append(
            {
                "gw": gw_num,
                "team_id": team_h,
                "opponent_team": team_a,
                "was_home": 1,
                "fixture_difficulty": diff_h,
            }
        )
        # away perspective
        rows.append(
            {
                "gw": gw_num,
                "team_id": team_a,
                "opponent_team": team_h,
                "was_home": 0,
                "fixture_difficulty": diff_a,
            }
        )

    return pd.DataFrame(rows)


def build_latest_history_features(
    players: pd.DataFrame,
    target_gw: int,  # NEW: we only use history BEFORE this GW
) -> pd.DataFrame:
    """
    For each player, fetch their history and compute fetaures using only gameweeks < target_gw.:0
    Goal is to build feature vectors that match the *training* semantics for GW = target_gw:

        - raw stats (minutes, goals, assists, etc.) correspond to last-completed GW (t-1)
        - lag / rolling 3-GW stats (same as in build_training_table) use last w gameweeks up to and including (t-1).

        1) keep history strictly before target_gw,
        2) compute rolling windows without shift()
        3) take last available GW row (< target_gw) for each player as the feature row for target_gw.

    Yields numerically the same features as in training for GW t.

    Returns a DataFrame with at most one row per player.
    """
    # -----------------------------------------------------------------
    # SPECIAL CASE: Predicting Gameweek 1 (no historical matches exist)
    # -----------------------------------------------------------------
    if target_gw == 1:
        print("No previous GWs exist. Creating GW1 baseline features.")

        df = players.copy()

        # Create dummy gw = 0 (never used except for grouping consistency)
        df["gw"] = 0

        # All history-based stats = 0
        zero_cols = [
            "minutes", "goals_scored", "assists", "clean_sheets",
            "yellow_cards", "red_cards", "penalties_missed", "own_goals",
            "bonus", "saves",
            "prev_points", "prev_minutes",
            "roll_pts_3", "roll_min_3",
            "roll_pts_5", "roll_min_5",
            "roll_pts_8", "roll_min_8",
            "roll_pts_per90_3", "roll_pts_per90_5", "roll_pts_per90_8",
            "roll_goals_3", "roll_assists_3", "roll_yellow_3",
            "roll_red_3", "roll_pen_miss_3", "roll_og_3",
            "roll_bonus_3", "roll_saves_3",
            "games_played_5", "minutes_share_5",
            "transfers_in_event", "transfers_out_event"
        ]

        for c in zero_cols:
            df[c] = 0.0

        return df
        
    all_rows: List[pd.DataFrame] = []

    for idx, prow in players.iterrows():
        pid = int(prow["player_id"])

        try:
            hist_df = fetch_player_history(pid)
        except requests.HTTPError as e:
            print(f"[WARN] Could not fetch history for player {pid}: {e}")
            continue

        if hist_df.empty:
            continue

        #align with training table naming
        hist_df = hist_df.rename(columns={"round": "gw"})

        #keep only history stricly before target GW 
        hist_df = hist_df[hist_df["gw"] < target_gw].copy()
        if hist_df.empty:
            continue

        hist_df["player_id"] = pid
        hist_df["team_id"] = prow.get("team_id", np.nan)
        hist_df["position"] = prow.get("position", np.nan)
        hist_df["price"] = prow.get("price", np.nan)
        hist_df["status"] = prow.get("status", "a")

        all_rows.append(hist_df)

        if (idx + 1) % 50 == 0:
            print(f"Processed history for {idx + 1} players...")

    if not all_rows:
        raise RuntimeError("No player history could be fetched for predictions.")

    df = pd.concat(all_rows, ignore_index=True)

    # sort for consistant rolling calculations
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)

    # Groupby for lag + rolling features
    g = df.groupby("player_id", group_keys=False)

    df["prev_points"] = g["total_points"].shift(1)
    df["prev_minutes"] = g["minutes"].shift(1)

    # Rolling windows for points/minutes (3, 5, 8)
    for w in [3, 5, 8]:
        df[f"roll_pts_{w}"] = (
            g["total_points"].shift(1).rolling(window=w, min_periods=1).sum()
        )
        df[f"roll_min_{w}"] = (
            g["minutes"].shift(1).rolling(window=w, min_periods=1).sum()
        )
        minutes_col = df[f"roll_min_{w}"].replace(0, np.nan)
        df[f"roll_pts_per90_{w}"] = (
            df[f"roll_pts_{w}"] / (minutes_col / 90.0)
        )
        df[f"roll_pts_per90_{w}"] = df[f"roll_pts_per90_{w}"].fillna(0.0)

    # same 3-GW rolling for other stats
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

    # "Nailedness" features over last 5 GWs  
    minutes_last5 = g["minutes"].shift(1).rolling(window=5, min_periods=1).sum()
    games_last5 = (
        (g["minutes"].shift(1) > 0).astype(float).rolling(window=5, min_periods=1).sum()
    )

    df["games_played_5"] = games_last5
    df["minutes_share_5"] = minutes_last5 / (5 * 90.0)

    lag_roll_cols = [
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
    ]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0.0)

    # Keep only the latest GW BEFORE target_gw per player
    idx_latest = df.groupby("player_id")["gw"].idxmax()
    latest_df = df.loc[idx_latest].copy().reset_index(drop=True)

    return latest_df


def prepare_prediction_features(
    df_latest: pd.DataFrame,
    fixtures_gw: pd.DataFrame,
    feature_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine latest history features with fixtures for the target GW,
    then build a feature matrix X aligned with feature_cols.

    Returns:
        df_pred: full player info + features + metadata
        X      : feature matrix ready for model.predict
    """
    # Merge to attach fixture info for the target GW
    df = df_latest.merge(
        fixtures_gw,
        on="team_id",
        how="left",
        suffixes=("", "_fix"),
    )

    # Use fixture's opponent + difficulty + was_home for the target GW
    if "opponent_team_fix" in df.columns:
        df["opponent_team"] = df["opponent_team_fix"]
    if "fixture_difficulty_fix" in df.columns:
        df["fixture_difficulty"] = df["fixture_difficulty_fix"]
    if "was_home_fix" in df.columns:
        df["was_home"] = df["was_home_fix"]

    df.drop(columns=[c for c in df.columns if c.endswith("_fix")], inplace=True)

    # Fill missing fixture info (e.g. blanks) with neutral values
    df["opponent_team"] = df["opponent_team"].fillna(0).astype(int)
    df["fixture_difficulty"] = df["fixture_difficulty"].fillna(3.0)
    df["was_home"] = df["was_home"].fillna(0).astype(int)

    # encode position
    pos_map = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    df["position_encoded"] = df["position"].map(pos_map).fillna(-1).astype(int)

    # encode status availability
    df["status_available"] = df["status"].isin(["a", "d", "s"]).astype(int)

    # ensure all numeric columns used in training exist; if missing, create as 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # reorder and cast
    X = df[feature_cols].astype(float)

    return df, X


# ---------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------

def predict_for_gw(target_gw: int) -> Path:
    """
    Predict FPL points for a given gameweek `target_gw`,
    using only information from gameweeks < target_gw.
    Saves a CSV in results/ml_predictions and returns its path.
    """
    # 1) load trained model + feature list
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Model expects {len(feature_cols)} features.")
    print(f"\n=== Predicting for GW {target_gw} ===")

    # 2) Load players (same base as training) and filter by status
    players = load_players()
    players = players[players["status"].isin(["a", "d", "s"])].copy()
    print(f"Building predictions for {len(players)} players with acceptable status.")

    # 3) Latest history features BEFORE target_gw (no leakage)
    df_latest = build_latest_history_features(players, target_gw)

    # 4) Fixtures for target_gw
    fixtures_gw = get_gw_fixtures(target_gw)
    if fixtures_gw.empty:
        raise RuntimeError(f"No fixtures found for GW {target_gw} in API")

    # 5) Build prediction feature matrix
    df_pred, X_pred = prepare_prediction_features(df_latest, fixtures_gw, feature_cols)

    # 6) Predict points
    preds = model.predict(X_pred)
    df_pred["predicted_points"] = preds
    df_pred["gw"] = target_gw

    # 7) Attach player metadata (name, team_name, position, price)
    players_meta = pd.read_csv(PLAYERS_CSV)
    df_out = df_pred.merge(
        players_meta[["player_id", "name", "team_name", "position", "price", "status"]],
        on=["player_id", "position"],
        how="left",
        suffixes=("", "_meta"),
    )

    # 8) Save to CSV
    output_path = PREDICTIONS_DIR / f"predictions_gw{target_gw}.csv"
    cols_to_save = [
        "gw",
        "player_id",
        "name",
        "team_name",
        "position",
        "predicted_points",
        "price",
        "status",
    ]
    extra_cols = [
        "team_id",
        "opponent_team",
        "fixture_difficulty",
        "prev_points",
        "roll_pts_3",
        "roll_goals_3",
        "roll_assists_3",
    ]
    cols_to_save = cols_to_save + [c for c in extra_cols if c in df_out.columns]

    df_out[cols_to_save].sort_values(
        "predicted_points", ascending=False
    ).to_csv(output_path, index=False)

    print(f"Saved predictions for GW {target_gw} to {output_path}")
    return output_path


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gw",
        type=int,
        default=None,
        help="Gameweek to predict (e.g. 7). If omitted, use the next GW from the API.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gw is not None:
        target_gw = args.gw
        print(f"Using user-specified gameweek: GW {target_gw}")
    else:
        user_input = input(
            "No gameweek provided.\n"
            "7Enter the gameweek you want to predict (or press Enter to use next GW): "
        ).strip()

        if user_input:
            target_gw = int(user_input)
            print(f"Using user-input gameweek: GW {target_gw}")
        else:
            target_gw = get_next_gameweek()
            print(f"Using next gameweek from API: GW {target_gw}")

    predict_for_gw(target_gw)


if __name__ == "__main__":
    main()