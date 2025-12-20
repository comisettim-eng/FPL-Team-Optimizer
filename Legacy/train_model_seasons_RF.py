from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

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

CURRENT_SEASON = "2025-26"

# Validation on CURRENT_SEASON GW1–8
VALID_MAX_GW = 8

# For predictions: we want to predict up to GW 15.
# Row with gameweek = t predicts points in GW t+1,
# so we need feature rows with gameweek <= 14.
PREDICT_UP_TO_GW = 15
CURRENT_MAX_FEATURE_GW = PREDICT_UP_TO_GW - 1  # 14


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
def load_training_data(path: Path = TRAIN_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------
# FEATURE PREP
# ---------------------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    """
    Build X, y from the player_gw_training schema.

    Uses all numeric / engineered features except IDs, labels, etc.
    Target = target_next_points (points in NEXT gameweek).
    """
    if "target_next_points" not in df.columns:
        raise ValueError("Column 'target_next_points' missing in training data.")

    # Columns we do NOT want as features
    drop_cols = {
        "season",
        "gameweek",
        "player_id",
        "name",
        "team_id",
        "team_name",
        "position",
        "element_type",
        "target_next_points",
    }

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Ensure everything is numeric
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df["target_next_points"], errors="coerce").fillna(0.0)

    return X, y, feature_cols


# ---------------------------------------------------------------------
# TRAIN ON PAST SEASONS, VALIDATE ON 2025-26 GW1–8, PREDICT 2025-26 UP TO GW15
# ---------------------------------------------------------------------
def train_model():
    print(f"Loading training data from {TRAIN_CSV}")
    df = load_training_data()

    if "season" not in df.columns:
        raise ValueError("Column 'season' missing in training data.")
    if "gameweek" not in df.columns:
        raise ValueError("Column 'gameweek' missing in training data.")

    # -----------------------------------------------------------------
    # 1) Build train / validation splits
    # -----------------------------------------------------------------
    # TRAIN: all seasons except CURRENT_SEASON
    train_df = df[df["season"] != CURRENT_SEASON].copy()

    # VALIDATION: CURRENT_SEASON, GW 1..VALID_MAX_GW
    valid_df = df[
        (df["season"] == CURRENT_SEASON) & (df["gameweek"] <= VALID_MAX_GW)
    ].copy()

    if valid_df.empty:
        raise ValueError(
            f"No rows found for CURRENT_SEASON={CURRENT_SEASON} with "
            f"gameweek <= {VALID_MAX_GW} in player_gw_training.csv"
        )

    print(
        "Training on seasons (excluding current):",
        sorted(train_df["season"].unique().tolist()),
    )
    print(
        f"Validation on CURRENT_SEASON={CURRENT_SEASON}, "
        f"gameweeks 1..{VALID_MAX_GW}"
    )
    print(f"Train rows: {len(train_df)}, Valid rows: {len(valid_df)}")

    # Features for train / validation
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_valid, y_valid, _ = prepare_features(valid_df)

    # -----------------------------------------------------------------
    # 2) Train model
    # -----------------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # 3) Evaluate on CURRENT_SEASON (GW1–8)
    # -----------------------------------------------------------------
    y_valid_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_valid_pred)
    mse = mean_squared_error(y_valid, y_valid_pred)
    rmse = mse ** 0.5

    print(f"\n=== Validation performance on {CURRENT_SEASON} GW1–{VALID_MAX_GW} ===")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    # Save detailed validation predictions (for analysis)
    if {"gameweek", "player_id"}.issubset(valid_df.columns):
        val_preds_df = valid_df[["season", "gameweek", "player_id"]].copy()
        val_preds_df["y_true"] = y_valid.values
        val_preds_df["y_pred"] = y_valid_pred

        val_backtest_path = BACKTEST_DIR / f"predictions_{CURRENT_SEASON}_GW1_{VALID_MAX_GW}.csv"
        val_preds_df.to_csv(val_backtest_path, index=False)
        print(f"Saved detailed validation predictions to: {val_backtest_path}")

    # -----------------------------------------------------------------
    # 4) Build predictions for CURRENT_SEASON up to GW15
    # -----------------------------------------------------------------
    # Need rows with gameweek <= CURRENT_MAX_FEATURE_GW (e.g. 1..14),
    # because each row with gameweek = t predicts points in GW t+1.
    current_df = df[
        (df["season"] == CURRENT_SEASON)
        & (df["gameweek"] <= CURRENT_MAX_FEATURE_GW)
    ].copy()

    if current_df.empty:
        raise ValueError(
            f"No rows found for CURRENT_SEASON={CURRENT_SEASON} with "
            f"gameweek <= {CURRENT_MAX_FEATURE_GW} in player_gw_training.csv"
        )

    # Use same feature_cols as training; ignore their target_next_points here
    X_current = current_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_current_pred = model.predict(X_current)

    # Save full predictions for CURRENT_SEASON (for analysis)
    current_preds_df = current_df[["season", "gameweek", "player_id"]].copy()
    current_preds_df["y_pred"] = y_current_pred

    preds_backtest_path = BACKTEST_DIR / f"predictions_{CURRENT_SEASON}_up_to_GW{PREDICT_UP_TO_GW}.csv"
    current_preds_df.to_csv(preds_backtest_path, index=False)
    print(f"Saved {CURRENT_SEASON} predictions (up to GW{PREDICT_UP_TO_GW}) to: {preds_backtest_path}")

    # LP-ready table for optimizer:
    # rows with gameweek = t contain predicted_next_points FOR GW t+1,
    # where t ranges 1..CURRENT_MAX_FEATURE_GW (so predictions for GW 2..PREDICT_UP_TO_GW).
    lp_preds = current_preds_df.rename(columns={"y_pred": "predicted_next_points"})
    lp_path = DATA_DIR / "player_gw_predictions_2025_26.csv"
    lp_preds.to_csv(lp_path, index=False)
    print(
        f"Saved LP-ready predictions to: {lp_path} "
        f"(gameweek 1..{CURRENT_MAX_FEATURE_GW} -> predicting GW 2..{PREDICT_UP_TO_GW})"
    )

    # -----------------------------------------------------------------
    # 5) Save model + metadata
    # -----------------------------------------------------------------
    model_path = MODELS_DIR / "rf_points_model_seasons.pkl"
    meta_path = MODELS_DIR / "rf_points_model_seasons_metadata.json"

    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"\nSaved trained model to: {model_path}")

    metadata = {
        "type": "RandomForestRegressor",
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 5,
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "validation_season": CURRENT_SEASON,
        "validation_gw_range": [1, VALID_MAX_GW],
        "prediction_season": CURRENT_SEASON,
        "prediction_feature_gw_max": int(CURRENT_MAX_FEATURE_GW),
        "prediction_target_gw_max": int(PREDICT_UP_TO_GW),
        "mae_valid": float(mae),
        "rmse_valid": float(rmse),
        "feature_cols": feature_cols,
        "notes": (
            f"Trained on all seasons except {CURRENT_SEASON}, "
            f"validated on {CURRENT_SEASON} GW1–{VALID_MAX_GW}, "
            f"and produced predictions for {CURRENT_SEASON} rows with gameweek "
            f"<= {CURRENT_MAX_FEATURE_GW} (i.e. targets GW2–{PREDICT_UP_TO_GW})."
        ),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")


def main():
    train_model()


if __name__ == "__main__":
    main()