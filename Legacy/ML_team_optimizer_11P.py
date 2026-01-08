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

from src.optimize.LP_team_optimizer_11P import build_team_optimizer  # existing LP model

DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "ml_predictions"
SOLUTIONS_DIR = PROJECT_ROOT / "results" / "ml_solutions"

SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_predictions_for_gw(gw: int) -> pd.DataFrame:
    """
    Load ML predictions for a given gameweek.
    Expects a file results/ml_predictions/predictions_gw<gw>.csv
    with at least:
      - player_id, name, team_name, team_id, position, price, status
      - predicted_points
    """
    path = PREDICTIONS_DIR / f"predictions_gw{gw}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}\n"
            f"Run prediction script first, e.g.: "
            f"`python -m src.ml.predict_next_gameweek --gw {gw}`"
        )

    df = pd.read_csv(path)

    # Basic sanity checks
    required_cols = [
        "player_id",
        "name",
        "team_name",
        "team_id",
        "position",
        "price",
        "status",
        "predicted_points",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in predictions file: {missing}")

    #Make sure price & predicted_points are numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["predicted_points"] = pd.to_numeric(df["predicted_points"], errors="coerce")

    #drop players without valid prediction or price
    df = df.dropna(subset=["price","predicted_points"]).copy()

    return df

def optimize_team_from_predictions(
    preds_df: pd.DataFrame,
    budget: float = 100.0,
) -> Tuple[pd.DataFrame, pl.LpProblem]:
    """
    Use existing LP optimizer to build the best 15-man squad possible for a single gameweek, maximizing ML predicted_points.
    """
    # Can directly use preds_df as the "players" input, as long as it has at least: position, team_name, price, and predicted_points.
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
    print(f"\n=== ML-based LP optimization for GW {gw} ===")

    #1) Load predictions
    preds_df = load_predictions_for_gw(gw)
    print(f"Loaded predictions for {len(preds_df)} players.")
    
    #2) Run LP optimizer with predicted_points as the objective
    best_team, model = optimize_team_from_predictions(preds_df, budget=budget)

    #3) show summary
    print("LP status:", pl.LpStatus[model.status])
    total_pred_points = (best_team["predicted_points"] * best_team["is_starter"].astype(int) * (1 + best_team["is_captain"].astype(int))).sum()

    print(f"Total predicted points for starting XI (with captain): {total_pred_points:.2f}\n")

    #format output
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

    #4) Save to CSV
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
        help="Gameweek for which to build optimized team (e.g. 12)."
            "Must already have predictions_gw<gw>.csv in results/ml_prediction."

    )
    parser.add_argument(
        "--budget", #might be helpful for transfers
        type=float,
        default=None,
        help="Total budget in millions (default 100.0)"
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

    #python -m src.optimize.ML_team_optimizer_11P --gw 12 --budget 100