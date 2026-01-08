import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import pandas as pd
import pulp as pl

# ---------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# existing LP model with prev_squad_ids + transfers_per_gw support
from src.optimize.LP_team_optimizer_11P import build_team_optimizer

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SOLUTIONS_DIR = PROJECT_ROOT / "results" / "ml_season_solutions"
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

# File produced by train_model() script
PREDICTIONS_CSV = DATA_DIR / "player_gw_predictions_2025_26.csv"

# Current-season players file from fetch_fpl_seasons_gw.py
CURRENT_SEASON = "2025-26"
PLAYERS_CUR_CSV = DATA_DIR / f"{CURRENT_SEASON}_players_clean.csv"


# ---------------------------------------------------------------------
# Helpers: load players & predictions
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
    Load the table produced by train_model() script, with columns:
        season, gameweek, player_id, predicted_next_points
    """
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_CSV}\n"
            "Run training script first."
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


def build_predictions_for_gw(
    gw: int,
    players: pd.DataFrame,
    preds_all: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a player table for a given gameweek with a 'predicted_points' column,
    ready to be passed into build_team_optimizer.

    IMPORTANT:
    - In the training set, the target is 'target_next_points' = points in the
      *next* gameweek.
    - Therefore, a row with gameweek = t in player_gw_predictions_2025_26.csv
      is a prediction for GW t+1.

    Here we assume we start optimizing at GW2, so:
      - For GW2: use predictions with gameweek = 1.
      - For GW3: use predictions with gameweek = 2.
      - ...
      - For GWs beyond the last prediction week, we reuse the last available one.
    """
    season_mask = preds_all["season"] == CURRENT_SEASON
    preds_season = preds_all[season_mask].copy()

    if preds_season.empty:
        raise ValueError(f"No predictions rows for season={CURRENT_SEASON}.")

    max_lookup_gw = int(preds_season["gameweek"].max())

    if gw <= 1:
        raise ValueError(
            "build_predictions_for_gw was called with gw <= 1, "
            "but this script is designed to start at GW2."
        )

    # We want predictions for GW gw, so we look up gameweek = gw-1
    lookup_gw = gw - 1
    if lookup_gw > max_lookup_gw:
        # For GWs beyond what we have predictions for,
        # reuse the last available prediction GW.
        lookup_gw = max_lookup_gw

    preds_gw = preds_season[preds_season["gameweek"] == lookup_gw].copy()

    if preds_gw.empty:
        raise ValueError(
            f"No predictions found for season={CURRENT_SEASON}, "
            f"lookup_gw={lookup_gw} (requested GW={gw}) in {PREDICTIONS_CSV}"
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

    if merged.empty:
        raise ValueError(
            f"After merging and cleaning, no valid players remained for GW{gw}.\n"
            "Check that player_id values align between players_clean and predictions."
        )

    # Ensure each player_id appears only once
    merged = (
        merged.sort_values("predicted_points", ascending=False)
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )

    return merged


# ---------------------------------------------------------------------
# Core: season-long optimization
# ---------------------------------------------------------------------
def optimize_season(
    start_gw: int = 2,
    end_gw: int = 15,
    budget: float = 100.0,
) -> Dict[str, pd.DataFrame]:
    """
    Season-long ML+LP optimizer:

      - GW2: pick an optimal FPL squad from scratch (no previous squad),
              using ML predictions for GW2 (lookup gameweek=1).
      - GW3..end_gw: keep exactly 14 of the previous 15 players
        => only 1 transfer per GW, using ML predictions for that GW.

    At each GW, we:
      - Use the best available ML estimate of points for that GW,
      - Use build_team_optimizer to enforce all FPL constraints,
      - Apply the 1-transfer rule from GW3 onward,
      - Save each squad and starting XI to CSV.

    Returns
    -------
    all_squads : dict mapping 'GWxx' -> DataFrame of chosen squad.
    """
    if start_gw < 2:
        raise ValueError("This optimizer is designed to start at GW2 or later.")

    players_cur = load_players_current_season()
    preds_all = load_predictions_table()

    prev_squad_ids: Optional[List[int]] = None
    all_squads: Dict[str, pd.DataFrame] = {}
    total_season_pred_points = 0.0

    for gw in range(start_gw, end_gw + 1):
        print(f"\n=== ML-based LP optimization for GW {gw} ===")

        preds_df = build_predictions_for_gw(gw, players_cur, preds_all)
        print(f"Loaded {len(preds_df)} unique players with predictions for GW {gw}.")

        # GW2 (i.e. first GW in this loop): no transfer constraint
        # GW > start_gw: exactly 1 transfer per week
        if prev_squad_ids is None:
            transfers_per_gw = 0
        else:
            transfers_per_gw = 1

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
        out_path = SOLUTIONS_DIR / f"ml_team_{gw_tag}.csv"
        best_team.to_csv(out_path, index=False)
        print(f"Saved optimized ML squad for {gw_tag} to: {out_path}")

        # Save starting XI separately
        xi_path = SOLUTIONS_DIR / f"starting_xi_{gw_tag}.csv"
        best_team[best_team["is_starter"]].to_csv(xi_path, index=False)
        print(f"Saved starting XI for {gw_tag} to: {xi_path}")

        # Update previous squad for next GW (this enforces the 1-transfer rule)
        prev_squad_ids = best_team["player_id"].tolist()
        all_squads[gw_tag] = best_team

    print(
        f"\n=== Season summary (GW{start_gw}–GW{end_gw}) ===\n"
        f"Total predicted points over season "
        f"(sum of GW starting XI + captain): {total_season_pred_points:.2f}"
    )

    return all_squads


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Season-long ML+LP FPL optimizer: \n"
            "GW2: pick initial squad (ML predictions); "
            "GW>2: exactly 1 transfer per gameweek using RF predictions. "
            "Objective: maximize predicted points under FPL rules."
        )
    )

    parser.add_argument(
        "--start_gw",
        type=int,
        default=2,
        help="First gameweek to optimize (default 2). Must be >= 2.",
    )
    parser.add_argument(
        "--end_gw",
        type=int,
        default=15,
        help="Last gameweek to optimize (default 15).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Total FPL budget in millions (default 100.0).",
    )

    args = parser.parse_args()

    optimize_season(
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        budget=args.budget,
    )


if __name__ == "__main__":
    main()