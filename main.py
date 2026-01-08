"""
Entrypoint for  FPL-Team-Optimizer project.

Pipeline :
  1. Build raw + processed season data (players & gw_history) for 2016-17..2025-26
  2. Build ML-ready feature tables (player_gw_training.csv, etc.)
  3. Train & compare ML models (RF, LightGBM, XGBoost) and save predictions
  4. Run season-long LP optimisation for each model using those predictions
  5. Compare actual season scores for each model vs FPL global average
  6. Print ONE final merged results summary table (regression + ranking + season scores)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Import pipeline steps
# ---------------------------------------------------------------------
from src.data.fetch_fpl_seasons_gw import main as build_season_data 
from src.data.build_training_table_seasons import main as build_feature_tables
from src.models.train_compare_models import main as train_ml_models

from src.optimize.ML_season_optimizer_transfers_multi import (
    optimize_season_for_model,
    CURRENT_SEASON as OPTIMIZER_SEASON,
)

from src.evaluate.compare_models_season_scores import compare_models_over_gws
from src.evaluate.make_report_artifacts import main as make_report_artifacts

# Default config
MODEL_KEYS = ("rf", "lgbm", "xgb")        
OPTIMIZER_START_GW = 2
OPTIMIZER_END_GW = 15
COMPARE_START_GW = 2
COMPARE_END_GW = 12


VAL_SEASON = "2025-26"
VAL_MAX_GW = 9


# ---------------------------------------------------------------------
# FINAL SUMMARY TABLE HELPERS
# ---------------------------------------------------------------------
def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_final_results_summary(
    project_root: Path,
    season_scores_df: pd.DataFrame,
    val_season: str,
    val_max_gw: int,
) -> pd.DataFrame:
    """
    Merge:
      - Regression metrics (MAE/RMSE/R2) from train_compare_models outputs
      - Ranking metrics (Spearman, topK overlap) from train_compare_models outputs
      - Actual season scores from compare_models_over_gws output

    Returns a single tidy table.
    """
    backtests_dir = project_root / "results" / "ml_backtests"

    # 1) Regression comparison table
    reg_path = backtests_dir / f"model_comparison_{val_season}_gw1_{val_max_gw}.csv"
    reg_df = _safe_read_csv(reg_path)

    # 2) Ranking comparison table
    rank_path = backtests_dir / f"model_ranking_comparison_{val_season}_gw1_{val_max_gw}.csv"
    rank_df = _safe_read_csv(rank_path)

    # Normalize model names to a stable key for merging with season scores
    model_key_map = {
        "RandomForest": "rf",
        "LightGBM": "lgbm",
        "XGBoost": "xgb",
        "rf": "rf",
        "lgbm": "lgbm",
        "xgb": "xgb",
    }

    # --- regression metrics -> model_key
    if not reg_df.empty and "model" in reg_df.columns:
        reg_df = reg_df.copy()
        reg_df["model_key"] = reg_df["model"].map(model_key_map).fillna(reg_df["model"])
    else:
        reg_df = pd.DataFrame(columns=["model_key"])

    # --- ranking metrics -> model_key
    if not rank_df.empty and "model" in rank_df.columns:
        rank_df = rank_df.copy()
        rank_df["model_key"] = rank_df["model"].map(model_key_map).fillna(rank_df["model"])
    else:
        rank_df = pd.DataFrame(columns=["model_key"])

    # 3) Season scores -> totals from compare_models_over_gws()
    # Expected cols: rf_score, lgbm_score, xgb_score, fpl_average_score (per GW)
    totals = {}
    if isinstance(season_scores_df, pd.DataFrame) and not season_scores_df.empty:
        for mk in ("rf", "lgbm", "xgb"):
            col = f"{mk}_score"
            if col in season_scores_df.columns:
                totals[mk] = float(season_scores_df[col].sum())

        if "fpl_average_score" in season_scores_df.columns:
            totals["fpl_average"] = float(season_scores_df["fpl_average_score"].sum(skipna=True))

    season_totals_df = pd.DataFrame(
        [{"model_key": mk, "season_total_points": v} for mk, v in totals.items() if mk in ("rf", "lgbm", "xgb")]
    )

    # Merge all
    out = pd.DataFrame({"model_key": ["rf", "lgbm", "xgb"]})

    # Keep only the important columns
    reg_keep = [
        "model_key",
        "valid_mae",
        "valid_rmse",
        "valid_r2",
        "train_mae",
        "train_rmse",
        "train_r2",
        "train_time_sec",
    ]
    reg_keep = [c for c in reg_keep if c in reg_df.columns]
    out = out.merge(reg_df[reg_keep], on="model_key", how="left")

    rank_keep = ["model_key"] + [c for c in rank_df.columns if c.startswith("spearman")]
    rank_keep = [c for c in rank_keep if c in rank_df.columns]
    out = out.merge(rank_df[rank_keep], on="model_key", how="left")

    out = out.merge(season_totals_df, on="model_key", how="left")

    # Add readable model names
    name_map = {"rf": "RandomForest", "lgbm": "LightGBM", "xgb": "XGBoost"}
    out.insert(0, "Model", out["model_key"].map(name_map).fillna(out["model_key"]))

    #Order + round
    preferred_order = [
        "Model",
        "model_key",
        "valid_mae",
        "valid_rmse",
        "valid_r2",
        "spearman_rho",
        "season_total_points",
        "train_time_sec",
    ]
    cols = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]
    out = out[cols]

    # Round numeric columns
    for c in out.columns:
        if c in ("Model", "model_key"):
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(3)

    # Sort: prioritize actual season performance if present, otherwise by valid_mae
    if "season_total_points" in out.columns and out["season_total_points"].notna().any():
        out = out.sort_values("season_total_points", ascending=False)
    elif "valid_mae" in out.columns:
        out = out.sort_values("valid_mae", ascending=True)

    out = out.reset_index(drop=True)
    return out


def print_final_results_summary(summary_df: pd.DataFrame, season_scores_df: pd.DataFrame) -> None:
    print("\n==============================")
    print("📊 FINAL RESULTS SUMMARY")
    print("==============================")

    if summary_df.empty:
        print("⚠️ No summary available (missing training outputs or scores).")
        return

    print("\nMerged model summary (validation metrics + ranking + actual season totals):\n")
    print(summary_df.to_string(index=False))

    # Print FPL average total if present
    if isinstance(season_scores_df, pd.DataFrame) and not season_scores_df.empty:
        if "fpl_average_score" in season_scores_df.columns:
            avg_total = float(season_scores_df["fpl_average_score"].sum(skipna=True))
            print(f"\nFPL global average total (GW range): {avg_total:.2f}")


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def run_pipeline(
    run_data: bool = True,
    run_features: bool = True,
    run_training: bool = True,
    run_optimizers: bool = True,
    run_comparison: bool = True,
    run_reporting: bool = True,
):
    t0 = time.time()
    print("\n==============================")
    print(" FPL TEAM OPTIMIZER PIPELINE ")
    print("==============================\n")

    # 1) Build data
    if run_data:
        print("\n[1/5] Building raw + processed season data ...")
        build_season_data()
    else:
        print("\n[1/5] Skipping data build step.")

    # 2) Feature tables
    if run_features:
        print("\n[2/5] Building ML feature tables ...")
        build_feature_tables()
    else:
        print("\n[2/5] Skipping feature-table build step.")

    # 3) Train ML models & prediction files
    if run_training:
        print("\n[3/5] Training ML models & creating predictions ...")
        train_ml_models()
    else:
        print("\n[3/5] Skipping ML training step.")

    # 4) Season-long optimization for each model
    if run_optimizers:
        print("\n[4/5] Running season-long optimisers ...")
        for mk in MODEL_KEYS:
            print(f"\n--- Optimising season {OPTIMIZER_SEASON} for model '{mk}' ---")
            optimize_season_for_model(
                model_key=mk,
                start_gw=OPTIMIZER_START_GW,
                end_gw=OPTIMIZER_END_GW,
                budget=100.0,
            )
    else:
        print("\n[4/5] Skipping LP optimisation step.")

    # 5) Compare model season scores vs actual FPL points & average
    season_scores_df = pd.DataFrame()
    if run_comparison:
        print("\n[5/5] Comparing model scores vs FPL global average ...")
        season_scores_df = compare_models_over_gws(
            start_gw=COMPARE_START_GW,
            end_gw=COMPARE_END_GW,
        )
    else:
        print("\n[5/5] Skipping comparison step.")

    # 6) Build report artifacts (tables + figures) from saved CSVs (no retraining)
    if run_reporting:
        print("\n[6/6] Generating report tables + figures (no retraining) ...")
        make_report_artifacts()
    else:
        print("\n[6/6] Skipping report artifacts step.")


    # -----------------------------------------------------------------
    # FINAL: print 1 merged summary table at the very end
    # -----------------------------------------------------------------
    summary_df = build_final_results_summary(
        project_root=PROJECT_ROOT,
        season_scores_df=season_scores_df,
        val_season=VAL_SEASON,
        val_max_gw=VAL_MAX_GW,
    )
    print_final_results_summary(summary_df, season_scores_df)

    total = time.time() - t0
    print(
        f"\n✅ Pipeline finished in {total:.1f} seconds "
        f"({total/60:.1f} minutes)."
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FPL-Team-Optimizer pipeline.")

    parser.add_argument("--skip-data", action="store_true", help="Skip rebuilding raw + processed season data.")
    parser.add_argument("--skip-features", action="store_true", help="Skip building ML feature tables.")
    parser.add_argument("--skip-training", action="store_true", help="Skip training ML models & generating predictions.")
    parser.add_argument("--skip-optimizers", action="store_true", help="Skip season-long LP optimisation.")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip comparing model scores vs FPL average.")
    parser.add_argument("--skip-reporting", action="store_true", help="Skip generating report tables + figures.")

    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        run_data=not args.skip_data,
        run_features=not args.skip_features,
        run_training=not args.skip_training,
        run_optimizers=not args.skip_optimizers,
        run_comparison=not args.skip_comparison,
        run_reporting=not args.skip_reporting,
    )


if __name__ == "__main__":
    main()