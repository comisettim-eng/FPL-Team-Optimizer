import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import pulp as pl

# ---------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Your existing LP model
from src.optimize.LP_team_optimizer_11P import build_team_optimizer

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SOLUTIONS_DIR = PROJECT_ROOT / "results" / "ml_solutions_from_training"
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

# This is the file created by your train_model() script above
PREDICTIONS_CSV = DATA_DIR / "player_gw_predictions_2025_26.csv"

# Current-season players file from fetch_fpl_seasons_gw.py
CURRENT_SEASON = "2025-26"
PLAYERS_CUR_CSV = DATA_DIR / f"{CURRENT_SEASON}_players_clean.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_players_current_season() -> pd.DataFrame:
    """
    Load current-season players with meta info (name, team, position, price, status).
    """
    if not PLAYERS_CUR_CSV.exists():
        raise FileNotFoundError(
            f"Current-season players file not found: {PLAYERS_CUR_CSV}\n"
            "Run: python -m src.data.fetch_fpl_seasons_gw first."
        )

    df = pd.read_csv(PLAYERS_CUR_CSV)

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

    # status is optional but very useful
    if "status" not in df.columns:
        df["status"] = "a"  # assume available if not present

    # Core types
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


def load_predictions_table() -> pd.DataFrame:
    """
    Load the table produced by train_model(), with columns:
        season, gameweek, player_id, predicted_next_points
    """
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_CSV}\n"
            "Run your training script first:\n"
            "  python -m src.ml.<your_train_module>"
        )

    df = pd.read_csv(PREDICTIONS_CSV)
    df["season"] = df["season"].astype(str)
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")

    if "predicted_next_points" not in df.columns:
        raise ValueError(
            "Column 'predicted_next_points' missing in player_gw_predictions_2025_26.csv"
        )

    return df


def build_predictions_for_gw(gw: int) -> pd.DataFrame:
    """
    Build a player table for a given gameweek with a 'predicted_points' column,
    ready to be passed into build_team_optimizer.

    We interpret 'predicted_next_points' as the prediction for this gameweek
    itself (same gameweek index). If you want strict 'next GW' semantics, you
    can call this with gw-1 instead.
    """
    players = load_players_current_season()
    preds_all = load_predictions_table()

    # Filter to current season & selected GW
    preds_gw = preds_all[
        (preds_all["season"] == CURRENT_SEASON) & (preds_all["gameweek"] == gw)
    ].copy()

    if preds_gw.empty:
        raise ValueError(
            f"No predictions found for season={CURRENT_SEASON}, gameweek={gw} "
            f"in {PREDICTIONS_CSV}"
        )

    # Merge predictions onto current-season players using player_id
    merged = players.merge(
        preds_gw[["player_id", "predicted_next_points"]],
        on="player_id",
        how="inner",
    )

    # Rename to the name expected by the optimizer
    merged = merged.rename(columns={"predicted_next_points": "predicted_points"})

    # Drop rows with invalid prices or predictions
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged["predicted_points"] = pd.to_numeric(
        merged["predicted_points"], errors="coerce"
    )
    merged = merged.dropna(subset=["price", "predicted_points"]).copy()

    if merged.empty:
        raise ValueError(
            f"After merging and cleaning, no valid players remained for GW{gw}.\n"
            "Check that player_id values align between players_clean and predictions."
        )

    # -----------------------------------------------------------------
    # Ensure each player can only appear once
    # -----------------------------------------------------------------
    # If, for some reason, there are multiple rows with the same player_id
    # (e.g. duplicate predictions), we keep the row with the highest
    # predicted_points for that player.
    merged = (
        merged.sort_values("predicted_points", ascending=False)
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )

    return merged


def optimize_team_from_predictions(
    preds_df: pd.DataFrame,
    budget: float = 100.0,
) -> Tuple[pd.DataFrame, pl.LpProblem]:
    """
    Use existing LP optimizer to build the best 15-man squad for a single GW,
    maximizing ML predicted_points.

    Each player_id appears at most once in preds_df, so the LP
    cannot select the same player more than once.
    """
    best_team, model = build_team_optimizer(
        players=preds_df,
        budget=budget,
        objective_col="predicted_points",
    )
    return best_team, model


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(gw: int, budget: float = 100.0):
    print(f"\n=== ML-based LP optimization for GW {gw} (from training table) ===")

    # 1) Build prediction-augmented player table for this GW
    preds_df = build_predictions_for_gw(gw)
    print(f"Loaded {len(preds_df)} unique players with predictions for GW {gw}.")

    # 2) Run LP optimizer with predicted_points as objective
    best_team, model = optimize_team_from_predictions(preds_df, budget=budget)

    # 3) Show summary
    print("LP status:", pl.LpStatus[model.status])
    total_pred_points = (
        best_team["predicted_points"]
        * best_team["is_starter"].astype(int)
        * (1 + best_team["is_captain"].astype(int))
    ).sum()

    print(
        f"Total predicted points for starting XI (with captain): "
        f"{total_pred_points:.2f}\n"
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
    print("Optimal ML squad for this GW:")
    print(best_team[cols_to_show].to_string(index=False))

    # 4) Save to CSV
    out_path = SOLUTIONS_DIR / f"ml_team_gw{gw}.csv"
    best_team.to_csv(out_path, index=False)
    print(f"\nSaved optimized ML squad to: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gw",
        type=int,
        default=None,
        help=(
            "Gameweek for which to build optimized team (e.g. 3).\n"
            "Must exist in data/processed/player_gw_predictions_2025_26.csv."
        ),
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Total budget in millions (default 100.0).",
    )

    args = parser.parse_args()

    # Interactive fallback for GW
    if args.gw is None:
        gw_input = input("Which gameweek do you want to generate a team for? ").strip()
        args.gw = int(gw_input)

    # Interactive fallback for budget
    if args.budget is None:
        budget_input = input("Budget available? (press Enter for 100.0) ").strip()
        args.budget = float(budget_input) if budget_input else 100.0

    main(gw=args.gw, budget=args.budget)
