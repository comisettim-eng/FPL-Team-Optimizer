"""
train_compare_models.py

Train & compare multiple regression models to predict next-gameweek FPL points
(target_next_gw_points) using season-level feature tables.

Adds stronger, FPL-relevant evaluation:
- Regression: MAE, RMSE, R2
- Ranking: Spearman rank correlation (rho)
- Decision quality: Top-K overlap precision (K=11,15)
- Optional (secondary): "haul" classification metrics at a chosen threshold (>=6 points)

Outputs (results/ml_backtests):
- model_comparison_<season>_gw1_<max_gw>.csv
- model_comparison_<season>_gw1_<max_gw>_markdown.md
- model_ranking_comparison_<season>_gw1_<max_gw>.csv
- model_ranking_comparison_<season>_gw1_<max_gw>_markdown.md
- model_haul_metrics_<season>_gw1_<max_gw>_thr<thr>.csv
- model_haul_metrics_<season>_gw1_<max_gw>_thr<thr>_markdown.md
- predictions_<season>_gw1_<max_gw>_<model>.csv

Outputs (data/processed):
- player_gw_predictions_<season>_gw1_<max_gw>_<model>.csv   (LP-ready)

Outputs (models/ml_models):
- <model>_points_model_seasons.pkl
- <model>_points_model_seasons_metadata.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV = DATA_DIR / "player_gw_training.csv"

MODELS_DIR = PROJECT_ROOT / "models" / "ml_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_DIR = PROJECT_ROOT / "results" / "ml_backtests"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Validation spec
# ---------------------------------------------------------------------
VAL_SEASON = "2025-26"
VAL_MAX_GW = 9

# ---------------------------------------------------------------------
# Extra evaluation knobs
# ---------------------------------------------------------------------
TOPK_LIST = (11, 15)  # Top-K overlap precision: starting XI and full squad
HAUL_THRESHOLD = 6    


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
def load_training_data(path: Path = TRAIN_CSV) -> pd.DataFrame:
    """Load the ML feature table (no target yet)."""
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["season"] = df["season"].astype(str)
    return df


# ---------------------------------------------------------------------
# BUILD TARGET: NEXT GW POINTS
# ---------------------------------------------------------------------
def add_next_gw_target(
    df: pd.DataFrame, target_col: str = "target_next_gw_points"
) -> pd.DataFrame:
    """
    Add `target_next_gw_points` to the training table.

    For each (season, player_id, gameweek) row, the target is that
    player's `total_points` in the *next* row within the same season/player,
    i.e., next appearance.

    Rows that have no next appearance get NaN target and are dropped.
    """
    if "total_points" not in df.columns:
        raise ValueError("Column 'total_points' is required to build the target.")
    if "gameweek" not in df.columns:
        raise ValueError("Column 'gameweek' missing from training data.")

    df = df.copy()
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df["player_id"] = pd.to_numeric(df.get("player_id"), errors="coerce")

    df = df.sort_values(["season", "player_id", "gameweek"])

    df[target_col] = df.groupby(["season", "player_id"])["total_points"].shift(-1)

    df = df[~df[target_col].isna()].copy()
    return df


# ---------------------------------------------------------------------
# FEATURE PREP
# ---------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame,
    target_col: str = "target_next_gw_points",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare X, y, feature_cols from a dataframe that already contains
    `target_next_gw_points`.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' missing. "
            "Make sure add_next_gw_target() has been applied."
        )

    drop_cols = {
        "season",
        "gameweek",
        "player_id",
        "name",
        "team_id",
        "team_name",
        "position",
        "element_type",
        target_col,
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    return X, y, feature_cols


# ---------------------------------------------------------------------
# EXTRA METRICS (RANKING + DECISION QUALITY + OPTIONAL HAUL METRICS)
# ---------------------------------------------------------------------
def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation computed via rank transform + Pearson corr.
    Avoids scipy dependency for this single metric.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # stable ranks (ties get average rank)
    rt = pd.Series(y_true).rank(method="average").to_numpy()
    rp = pd.Series(y_pred).rank(method="average").to_numpy()

    # handle constant vectors
    if np.nanstd(rt) == 0 or np.nanstd(rp) == 0:
        return float("nan")

    rho = np.corrcoef(rt, rp)[0, 1]
    return float(rho)


def topk_overlap_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
) -> float:
    """
    Top-K overlap precision:
      |TopK(pred)| intersect |TopK(true)|  divided by K

    Interprets "decision quality": do we pick the right best players?
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n == 0:
        return float("nan")
    k_eff = int(min(k, n))
    if k_eff <= 0:
        return float("nan")

    top_true = set(np.argsort(-y_true)[:k_eff])
    top_pred = set(np.argsort(-y_pred)[:k_eff])

    return float(len(top_true & top_pred) / k_eff)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Optional: convert regression into a secondary binary task:
    "haul" if points >= threshold.
    Returns accuracy/precision/recall/f1 (as floats).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate_true": float(y_true_bin.mean()) if total else float("nan"),
        "positive_rate_pred": float(y_pred_bin.mean()) if total else float("nan"),
    }


# ---------------------------------------------------------------------
# TRAIN + EVAL WITH TIMING
# ---------------------------------------------------------------------
def train_and_evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Dict:
    """
    Train model, then compute train vs validation metrics to detect overfitting.
    Returns metrics + predictions on the validation set.
    """
    print(f"\n=== Training {name} ===")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"⏱ {name} training time: {train_time:.2f} seconds")

    # -------- TRAIN METRICS --------
    y_pred_train = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    train_r2 = float(r2_score(y_train, y_pred_train))

    # -------- VALIDATION METRICS --------
    y_pred_valid = model.predict(X_valid)
    valid_mae = mean_absolute_error(y_valid, y_pred_valid)
    valid_rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
    valid_r2 = float(r2_score(y_valid, y_pred_valid))

    # -------- GAP (overfitting signal) --------
    mae_gap = train_mae - valid_mae
    rmse_gap = train_rmse - valid_rmse

    print(f"{name} - TRAIN MAE : {train_mae:.3f}")
    print(f"{name} - VALID MAE : {valid_mae:.3f}")
    print(f"{name} - MAE GAP   : {mae_gap:.3f}")

    print(f"{name} - TRAIN RMSE: {train_rmse:.3f}")
    print(f"{name} - VALID RMSE: {valid_rmse:.3f}")
    print(f"{name} - RMSE GAP  : {rmse_gap:.3f}")

    print(f"{name} - TRAIN R2  : {train_r2:.3f}")
    print(f"{name} - VALID R2  : {valid_r2:.3f}")

    if mae_gap < -0.5 or rmse_gap < -0.5:
        print("⚠️  Model may be UNDERfitting (validation better than train).")
    elif mae_gap > 1.0 or rmse_gap > 1.0:
        print("⚠️  Possible OVERFITTING: train much better than validation.")

    return {
        "name": name,
        "model": model,
        "train_mae": float(train_mae),
        "valid_mae": float(valid_mae),
        "train_rmse": float(train_rmse),
        "valid_rmse": float(valid_rmse),
        "train_r2": float(train_r2),
        "valid_r2": float(valid_r2),
        "mae_gap": float(mae_gap),
        "rmse_gap": float(rmse_gap),
        "train_time": float(train_time),
        "y_pred_valid": np.asarray(y_pred_valid, dtype=float),
    }


def _save_markdown_table(df: pd.DataFrame, out_path: Path, float_cols: List[str]) -> None:
    """Save a simple GitHub-flavored markdown table (no extra dependencies)."""
    md_df = df.copy()
    for c in float_cols:
        if c in md_df.columns:
            md_df[c] = md_df[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")

    header = "| " + " | ".join(md_df.columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(md_df.columns)) + " |\n"
    lines = [header + sep]
    for _, row in md_df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in md_df.columns) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    total_start = time.time()
    np.random.seed(42)

    print(f"Loading training data from {TRAIN_CSV}")
    df = load_training_data()

    # -----------------------------------------------------------------
    # Build target column: next GW's points for each (season, player_id)
    # -----------------------------------------------------------------
    print("Adding target_next_gw_points ...")
    df = add_next_gw_target(df, target_col="target_next_gw_points")

    # Ensure gameweek numeric
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")

    # -----------------------------------------------------------------
    # SPLIT:
    #   Training   = all seasons EXCEPT VAL_SEASON
    #   Validation = VAL_SEASON, gameweeks <= VAL_MAX_GW
    # -----------------------------------------------------------------
    valid_df = df[(df["season"] == VAL_SEASON) & (df["gameweek"] <= VAL_MAX_GW)].copy()
    train_df = df[df["season"] != VAL_SEASON].copy()

    if valid_df.empty:
        raise ValueError(
            f"Validation set is empty after building target! "
            f"Check season={VAL_SEASON}, gameweek <= {VAL_MAX_GW} in {TRAIN_CSV.name}."
        )

    print(f"Train rows (with target): {len(train_df)}")
    print(f"Valid rows (with target): {len(valid_df)}")

    X_train, y_train, feature_cols = prepare_features(train_df, target_col="target_next_gw_points")
    X_valid, y_valid, _ = prepare_features(valid_df, target_col="target_next_gw_points")

    # Meta for backtest output
    valid_meta = valid_df[["season", "gameweek", "player_id"]].reset_index(drop=True)
    y_valid_array = y_valid.to_numpy(dtype=float)

    # -----------------------------------------------------------------
    # THE MODELS
    # -----------------------------------------------------------------
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        ),
    }

    short_key = {"RandomForest": "rf", "LightGBM": "lgbm", "XGBoost": "xgb"}

    # -----------------------------------------------------------------
    # TRAIN + EVAL
    # -----------------------------------------------------------------
    per_model: Dict[str, Dict] = {}
    rows_core: List[Dict] = []
    rows_rank: List[Dict] = []
    rows_haul: List[Dict] = []

    for name, model in models.items():
        res = train_and_evaluate_model(name, model, X_train, y_train, X_valid, y_valid)
        per_model[name] = res

        y_pred_valid = res["y_pred_valid"]

        # Ranking/decision metrics computed on validation set
        rho = spearman_rho(y_valid_array, y_pred_valid)
        rank_metrics = {
            "model": name,
            "spearman_rho": rho,
        }
        for k in TOPK_LIST:
            rank_metrics[f"top{k}_overlap_precision"] = topk_overlap_precision(
                y_valid_array, y_pred_valid, k=k
            )
        rows_rank.append(rank_metrics)

        # Optional haul metrics (secondary, threshold-based)
        haul = binary_classification_metrics(
            y_true=y_valid_array, y_pred=y_pred_valid, threshold=HAUL_THRESHOLD
        )
        rows_haul.append(
            {
                "model": name,
                "threshold": HAUL_THRESHOLD,
                "accuracy": haul["accuracy"],
                "precision": haul["precision"],
                "recall": haul["recall"],
                "f1": haul["f1"],
                "positive_rate_true": haul["positive_rate_true"],
                "positive_rate_pred": haul["positive_rate_pred"],
            }
        )

        # Core regression metrics
        rows_core.append(
            {
                "model": name,
                "train_mae": res["train_mae"],
                "valid_mae": res["valid_mae"],
                "mae_gap": res["mae_gap"],
                "train_rmse": res["train_rmse"],
                "valid_rmse": res["valid_rmse"],
                "rmse_gap": res["rmse_gap"],
                "train_r2": res["train_r2"],
                "valid_r2": res["valid_r2"],
                "train_time_sec": res["train_time"],
            }
        )

    # -----------------------------------------------------------------
    # SAVE TABLES
    # -----------------------------------------------------------------
    core_df = pd.DataFrame(rows_core).sort_values("valid_mae").reset_index(drop=True)
    core_csv = BACKTEST_DIR / f"model_comparison_{VAL_SEASON}_gw1_{VAL_MAX_GW}.csv"
    core_md = BACKTEST_DIR / f"model_comparison_{VAL_SEASON}_gw1_{VAL_MAX_GW}_markdown.md"
    core_df.to_csv(core_csv, index=False)
    _save_markdown_table(
        core_df,
        out_path=core_md,
        float_cols=[
            "train_mae", "valid_mae", "mae_gap",
            "train_rmse", "valid_rmse", "rmse_gap",
            "train_r2", "valid_r2",
            "train_time_sec",
        ],
    )

    print(f"\nSaved regression comparison table to: {core_csv}")
    print(core_df.to_string(index=False))
    print(f"\nSaved regression Markdown table to: {core_md}")

    rank_df = pd.DataFrame(rows_rank).sort_values("spearman_rho", ascending=False).reset_index(drop=True)
    rank_csv = BACKTEST_DIR / f"model_ranking_comparison_{VAL_SEASON}_gw1_{VAL_MAX_GW}.csv"
    rank_md = BACKTEST_DIR / f"model_ranking_comparison_{VAL_SEASON}_gw1_{VAL_MAX_GW}_markdown.md"
    rank_df.to_csv(rank_csv, index=False)

    rank_float_cols = ["spearman_rho"] + [f"top{k}_overlap_precision" for k in TOPK_LIST]
    _save_markdown_table(rank_df, out_path=rank_md, float_cols=rank_float_cols)

    print(f"\nSaved ranking/decision comparison table to: {rank_csv}")
    print(rank_df.to_string(index=False))
    print(f"\nSaved ranking/decision Markdown table to: {rank_md}")

    haul_df = pd.DataFrame(rows_haul).sort_values("f1", ascending=False).reset_index(drop=True)
    haul_csv = BACKTEST_DIR / f"model_haul_metrics_{VAL_SEASON}_gw1_{VAL_MAX_GW}_thr{HAUL_THRESHOLD}.csv"
    haul_md = BACKTEST_DIR / f"model_haul_metrics_{VAL_SEASON}_gw1_{VAL_MAX_GW}_thr{HAUL_THRESHOLD}_markdown.md"
    haul_df.to_csv(haul_csv, index=False)
    _save_markdown_table(
        haul_df,
        out_path=haul_md,
        float_cols=["accuracy", "precision", "recall", "f1", "positive_rate_true", "positive_rate_pred"],
    )

    print(f"\nSaved optional haul-metrics table to: {haul_csv}")
    print(haul_df.to_string(index=False))
    print(f"\nSaved optional haul-metrics Markdown table to: {haul_md}")

    # -----------------------------------------------------------------
    # SAVE MODELS + METADATA + PREDICTIONS
    # -----------------------------------------------------------------
    import joblib

    for name, res in per_model.items():
        sk = short_key[name]

        # model artifacts
        model_path = MODELS_DIR / f"{sk}_points_model_seasons.pkl"
        meta_path = MODELS_DIR / f"{sk}_points_model_seasons_metadata.json"

        joblib.dump({"model": res["model"], "feature_cols": feature_cols}, model_path)

        # attach extra eval summaries
        rank_row = next((r for r in rows_rank if r["model"] == name), {})
        haul_row = next((r for r in rows_haul if r["model"] == name), {})

        metadata = {
            "type": name,
            "short_key": sk,
            "train_seasons": sorted(train_df["season"].unique().tolist()),
            "validation_season": VAL_SEASON,
            "validation_gameweeks": f"1-{VAL_MAX_GW}",
            "feature_cols": feature_cols,
            "regression_metrics": {
                "train_mae": float(res["train_mae"]),
                "valid_mae": float(res["valid_mae"]),
                "train_rmse": float(res["train_rmse"]),
                "valid_rmse": float(res["valid_rmse"]),
                "train_r2": float(res["train_r2"]),
                "valid_r2": float(res["valid_r2"]),
                "train_time_sec": float(res["train_time"]),
            },
            "ranking_metrics_validation": {
                "spearman_rho": float(rank_row.get("spearman_rho", float("nan"))),
                **{
                    f"top{k}_overlap_precision": float(rank_row.get(f"top{k}_overlap_precision", float("nan")))
                    for k in TOPK_LIST
                },
            },
            "optional_haul_metrics_validation": {
                "threshold": HAUL_THRESHOLD,
                "accuracy": float(haul_row.get("accuracy", float("nan"))),
                "precision": float(haul_row.get("precision", float("nan"))),
                "recall": float(haul_row.get("recall", float("nan"))),
                "f1": float(haul_row.get("f1", float("nan"))),
                "positive_rate_true": float(haul_row.get("positive_rate_true", float("nan"))),
                "positive_rate_pred": float(haul_row.get("positive_rate_pred", float("nan"))),
            },
        }

        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {name} model + metadata (key={sk}).")

        # backtest predictions for validation window
        y_pred = np.asarray(res["y_pred_valid"], dtype=float)
        preds_df = valid_meta.copy()
        preds_df["y_true"] = y_valid_array
        preds_df["y_pred"] = y_pred

        preds_backtest_path = BACKTEST_DIR / f"predictions_{VAL_SEASON}_gw1_{VAL_MAX_GW}_{sk}.csv"
        preds_df.to_csv(preds_backtest_path, index=False)
        print(f"Saved detailed predictions for {name} to: {preds_backtest_path}")

        # LP-ready predictions file
        lp_df = preds_df[["season", "gameweek", "player_id", "y_pred"]].rename(
            columns={"y_pred": "predicted_next_points"}
        )
        lp_path = DATA_DIR / f"player_gw_predictions_{VAL_SEASON}_gw1_{VAL_MAX_GW}_{sk}.csv"
        lp_df.to_csv(lp_path, index=False)
        print(f"Saved LP-ready predictions for {name} to: {lp_path}")

    total_time = time.time() - total_start
    print(f"\n⏱ Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()