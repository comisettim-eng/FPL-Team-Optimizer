import sys
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import pulp as pl

# ---------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Your existing LP model with prev_squad_ids + transfers_per_gw support
from src.optimize.LP_team_optimizer_11P import build_team_optimizer

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SOLUTIONS_DIR = PROJECT_ROOT / "results" / "ml_season_solutions"
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

# File produced by your train_model() script
PREDICTIONS_CSV = DATA_DIR / "player_gw_predictions_2025_26.csv"

# Current-season players file (from append_current_season.py / similar)
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
            "Run your data-building pipeline first."
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
    Load the table produced by your train_model() script, with columns:
        season, gameweek, player_id, predicted_next_points
    """
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_CSV}\n"
            "Run your training script first."
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
    gw: int, players: pd.DataFrame, preds_all: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a player table for a given gameweek with a 'predicted_points' column,
    ready to be passed into build_team_optimizer.

    We use 'predicted_next_points' from row gameweek == gw-1, because that
    label was trained to predict points in GW == gw.
    """
    if gw <= 1:
        raise ValueError(
            "build_predictions_for_gw(gw, ...) is only defined for gw >= 2 "
            "when using gw-1 indexing."
        )

    preds_gw = preds_all[
        (preds_all["season"] == CURRENT_SEASON) & (preds_all["gameweek"] == gw - 1)
    ].copy()

    if preds_gw.empty:
        raise ValueError(
            f"No predictions found for season={CURRENT_SEASON}, gameweek={gw - 1} "
            f"in {PREDICTIONS_CSV} (needed to predict GW{gw})."
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
# Core: season-long optimization with 1 transfer per GW
# ---------------------------------------------------------------------
def optimize_season(
    start_gw: int = 1,
    end_gw: int = 9,
    budget: float = 100.0,
) -> Dict[str, pd.DataFrame]:
    """
    Season-long ML+LP optimizer with GW1 baseline:

      - GW1: build initial squad using a static baseline objective
        (points_per_game) without ML predictions.
      - GW2..end_gw: keep exactly 14 of the previous 15 players
        => only 1 transfer per GW,
        and use RandomForest predicted points (with gw-1 indexing) as objective.

    Returns
    -------
    all_squads : dict mapping 'GWxx' -> DataFrame of chosen squad.
    """
    players_cur = load_players_current_season()
    preds_all = load_predictions_table()

    prev_squad_ids: Optional[List[int]] = None
    all_squads: Dict[str, pd.DataFrame] = {}
    total_season_pred_points = 0.0

    for gw in range(start_gw, end_gw + 1):
        print(f"\n=== Optimization for GW {gw} ===")

        # ------------------------------
        # 1) Build per-GW player table
        # ------------------------------
        if gw == 1:
            # Baseline GW1: no ML predictions, use a simple interpretable metric
            if "points_per_game" not in players_cur.columns:
                raise ValueError(
                    "Column 'points_per_game' not found in current-season players; "
                    "cannot use baseline GW1 objective."
                )

            preds_df = players_cur.copy()
            preds_df["predicted_points"] = pd.to_numeric(
                preds_df["points_per_game"], errors="coerce"
            ).fillna(0.0)

            print(
                "GW1 baseline objective: using 'points_per_game' as predicted_points."
            )
        else:
            # From GW2 onward: use RF predictions with gw-1 indexing
            preds_df = build_predictions_for_gw(gw, players_cur, preds_all)
            print(
                f"Loaded {len(preds_df)} unique players with ML predictions for GW {gw} "
                f"(from gameweek={gw-1} rows)."
            )

        # ------------------------------
        # 2) Transfers per GW
        # ------------------------------
        if gw == start_gw:
            # First optimized GW: no previous squad to compare against
            transfers_per_gw = 0
        else:
            # Subsequent GWs: exactly 1 transfer per week
            transfers_per_gw = 1

        # ------------------------------
        # 3) Run LP optimizer
        # ------------------------------
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
        print("\nOptimal squad for this GW:")
        print(best_team[cols_to_show].to_string(index=False))

        # ------------------------------
        # 4) Save outputs
        # ------------------------------
        gw_tag = f"GW{gw:02d}"
        out_path = SOLUTIONS_DIR / f"ml_team_{gw_tag}.csv"
        best_team.to_csv(out_path, index=False)
        print(f"Saved optimized squad for {gw_tag} to: {out_path}")

        xi_path = SOLUTIONS_DIR / f"starting_xi_{gw_tag}.csv"
        best_team[best_team["is_starter"]].to_csv(xi_path, index=False)
        print(f"Saved starting XI for {gw_tag} to: {xi_path}")

        # Update previous squad for next GW
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
            "Season-long ML+LP FPL optimizer:\n"
            "- GW1: baseline squad using points_per_game.\n"
            "- GW>1: exactly 1 transfer per gameweek with RF-based predictions.\n"
            "Objective: maximize predicted points each GW under FPL rules."
        )
    )

    parser.add_argument(
        "--start_gw",
        type=int,
        default=1,
        help="First gameweek to optimize (default 1).",
    )
    parser.add_argument(
        "--end_gw",
        type=int,
        default=9,
        help="Last gameweek to optimize (default 9).",
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