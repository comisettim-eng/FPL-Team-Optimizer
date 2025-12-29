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

# existing LP model with prev_squad_ids + transfers_per_gw support
from src.optimize.LP_team_optimizer_11P import build_team_optimizer

DATA_DIR = PROJECT_ROOT / "data" / "processed"
BASE_SOLUTIONS_DIR = PROJECT_ROOT / "results" / "ml_season_solutions"
BASE_SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Must match VAL_SEASON in train_compare_models.py
CURRENT_SEASON = "2025-26"

# Must match VAL_MAX_GW in train_compare_models.py
PREDICTION_START_GW = 1
PREDICTION_END_GW = 9

PLAYERS_CUR_CSV = DATA_DIR / f"{CURRENT_SEASON}_players_clean.csv"

# Per-model predictions created by train_compare_models.py:
#   player_gw_predictions_2025-26_gw1_9_rf.csv
#   player_gw_predictions_2025-26_gw1_9_lgbm.csv
#   player_gw_predictions_2025-26_gw1_9_xgb.csv
PREDICTIONS_TEMPLATE = (
    "player_gw_predictions_{season}_gw1_{max_gw}_{model}.csv"
)

# Readable names for logs
MODEL_NAME = {
    "rf": "RandomForest",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
}


# ---------------------------------------------------------------------
# Helpers: load players & predictions
# ---------------------------------------------------------------------
def load_players_current_season() -> pd.DataFrame:
    """
    Load current-season players with meta info:
    [player_id, name, team_name, team_id, position, price, status].
    """
    if not PLAYERS_CUR_CSV.exists():
        raise FileNotFoundError(
            f"Current-season players file not found: {PLAYERS_CUR_CSV}\n"
            "Run: python -m src.data.append_current_season first."
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

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


def load_predictions_table(pred_path: Path) -> pd.DataFrame:
    """
    Load per-model prediction table with columns:
        season, gameweek, player_id, predicted_next_points
    (created by train_compare_models.py)
    """
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}\n"
            "Make sure you ran the training script for this model."
        )

    df = pd.read_csv(pred_path)
    df["season"] = df["season"].astype(str)
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")

    if "predicted_next_points" not in df.columns:
        raise ValueError(
            f"Column 'predicted_next_points' missing in {pred_path.name}"
        )

    return df


def build_predictions_for_gw(
    gw: int,
    players: pd.DataFrame,
    preds_all: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a player table for a given GW with a 'predicted_points' column,
    ready for build_team_optimizer.

    Training target = target_next_gw_points (points in NEXT gameweek).
    So a row with gameweek=t in predictions is a prediction for GW t+1.

    Start at GW2, so:
      - For GW2, use rows with gameweek=1
      - For GW3, use rows with gameweek=2
      - ...
    If gw-1 > max available, reuse the last prediction GW.
    """
    preds_season = preds_all[preds_all["season"] == CURRENT_SEASON].copy()
    if preds_season.empty:
        raise ValueError(f"No predictions rows for season={CURRENT_SEASON}.")

    max_lookup_gw = int(preds_season["gameweek"].max())

    if gw <= 1:
        raise ValueError(
            "build_predictions_for_gw was called with gw <= 1, "
            "but this optimizer is designed to start at GW2."
        )

    lookup_gw = gw - 1
    if lookup_gw > max_lookup_gw:
        lookup_gw = max_lookup_gw  # reuse last known predictions

    preds_gw = preds_season[preds_season["gameweek"] == lookup_gw].copy()
    if preds_gw.empty:
        raise ValueError(
            f"No predictions found for season={CURRENT_SEASON}, "
            f"lookup_gw={lookup_gw} (requested GW={gw})."
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

    # Ensure each player_id appears only once (keep best prediction if duplicated)
    merged = (
        merged.sort_values("predicted_points", ascending=False)
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )

    return merged


# ---------------------------------------------------------------------
# Season-long optimization for one model
# ---------------------------------------------------------------------
def optimize_season_for_model(
    model_key: str,
    start_gw: int = 2,
    end_gw: int = 15,
    budget: float = 100.0,
) -> Dict[str, pd.DataFrame]:
    """
    Run  season-long optimisation for a single ML model.

    - GW2: choose initial squad from scratch using predictions for GW2.
    - GW3..end_gw: exactly 1 transfer per GW (keep 14 of 15 players),
      still maximising predicted points each GW under FPL constraints.

    model_key: short name used in filenames, e.g. 'rf', 'lgbm', 'xgb'.
    """
    if start_gw < 2:
        raise ValueError("This optimizer is designed to start at GW2 or later.")

    pred_filename = PREDICTIONS_TEMPLATE.format(
        season=CURRENT_SEASON,
        max_gw=PREDICTION_END_GW,
        model=model_key,
    )
    pred_path = DATA_DIR / pred_filename

    solutions_dir = BASE_SOLUTIONS_DIR / model_key
    solutions_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=============================="
        f"\n Optimizing season for model: {model_key} "
        f"({MODEL_NAME.get(model_key, model_key)})"
        f"\n Using predictions file: {pred_path.name}"
        f"\n Solutions dir: {solutions_dir}"
        f"\n=============================="
    )

    players_cur = load_players_current_season()
    preds_all = load_predictions_table(pred_path)

    prev_squad_ids: Optional[List[int]] = None
    all_squads: Dict[str, pd.DataFrame] = {}
    total_season_pred_points = 0.0

    for gw in range(start_gw, end_gw + 1):
        print(f"\n=== {model_key} - ML-based LP optimization for GW {gw} ===")

        preds_df = build_predictions_for_gw(gw, players_cur, preds_all)
        print(f"Loaded {len(preds_df)} unique players with predictions for GW {gw}.")

        # GW2: initial squad, no transfer constraint.
        # GW>start_gw: exactly 1 transfer relative to previous squad.
        transfers_per_gw = 0 if prev_squad_ids is None else 1

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

        # Save full 15-man squad
        out_path = solutions_dir / f"{model_key}_team_{gw_tag}.csv"
        best_team.to_csv(out_path, index=False)
        print(f"Saved optimized ML squad for {gw_tag} to: {out_path}")

        # Save starting XI only
        xi_path = solutions_dir / f"{model_key}_starting_xi_{gw_tag}.csv"
        best_team[best_team["is_starter"]].to_csv(xi_path, index=False)
        print(f"Saved starting XI for {gw_tag} to: {xi_path}")

        # Update previous squad for next GW (enforces 1-transfer rule)
        prev_squad_ids = best_team["player_id"].tolist()
        all_squads[gw_tag] = best_team

    print(
        f"\n=== {model_key} season summary (GW{start_gw}–GW{end_gw}) ===\n"
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
            "Season-long ML+LP FPL optimizer for multiple models:\n"
            "  GW2: pick initial squad using ML predictions\n"
            "  GW>2: exactly 1 transfer per GW.\n"
            "Models: rf (RandomForest), lgbm (LightGBM), xgb (XGBoost)."
        )
    )

    parser.add_argument(
        "--models",
        type=str,
        default="rf,lgbm,xgb",
        help="Comma-separated list of models to run (rf,lgbm,xgb). "
             "Examples: 'rf', 'rf,lgbm', 'xgb'. Default: rf,lgbm,xgb.",
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

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    allowed = {"rf", "lgbm", "xgb"}
    for m in model_keys:
        if m not in allowed:
            raise ValueError(f"Unknown model key '{m}'. Use subset of {allowed}.")

    for m in model_keys:
        optimize_season_for_model(
            model_key=m,
            start_gw=args.start_gw,
            end_gw=args.end_gw,
            budget=args.budget,
        )


if __name__ == "__main__":
    main()