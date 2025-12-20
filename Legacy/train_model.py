import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

#---------------------------------------------------------------------
#Paths
#---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV = DATA_DIR / "player_gw_training.csv"

MODELS_DIR = PROJECT_ROOT / "models" / "ml_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_DIR = PROJECT_ROOT / "results" / "ml_backtests"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

#---------------------------------------------------------------------
#Load Data
#---------------------------------------------------------------------
def load_training_data(path: Path = TRAIN_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_csv(path)
    return df


#---------------------------------------------------------------------
#Preprocessing: feature engineering for ML
#---------------------------------------------------------------------

def prepare_features(df: pd.DataFrame): 
    """
    Take the player_gw_training table and build:
        X : feature matrix
        Y : target (points in that GW)

        pre-match stats column already shifted by 1 gameweek in build_training_table.py -> now represent previous gameweek stats
    """

    #Make sure required columns exist
    required = [
        "target_points",
        "price",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "yellow_cards",
        "red_cards",
        "penalties_missed",
        "own_goals",
        "bonus",
        "saves",
        "was_home",
        "opponent_team",
        "team_id",
        "position",
        "status",
        "fixture_difficulty",
        "transfers_in_event",
        "transfers_out_event",
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
        "roll_og_3",
        "roll_pen_miss_3",
        "roll_bonus_3",
        "roll_saves_3",
        "games_played_5",     
        "minutes_share_5",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    #encode position as numeric category
    pos_map = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    df["position_encoded"] = df["position"].map(pos_map).fillna(-1).astype(int)

    # 2) Encode status as availability flag (1 = available, 0 = not available)
    df["status_available"] = df["status"].isin(["a", "d", "s"]).astype(int)

#features to use - deliberately do not use previous week points to avoid it being too heavily influences by one performance - all pre match info

    feature_cols = [
        "price",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "yellow_cards",
        "red_cards",
        "penalties_missed",
        "own_goals",
        "bonus",
        "saves",
        "was_home",
        "opponent_team",
        "team_id",
        "fixture_difficulty",
        "transfers_in_event",
        "transfers_out_event",
        "position_encoded",
        "status_available",
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

    X = df[feature_cols].astype(float)
    y = df["target_points"].astype(float)

    return X, y, feature_cols

#---------------------------------------------------------------------
#Train / validation split (time-aware)
#---------------------------------------------------------------------
def time_based_split(df: pd.DataFrame, test_ratio: float = 0.1): #30% of the past gameweeks will be used for validation
    """
    Split by gameweek to avoid future leakage:
    use earlier gameweeks for training, and later gameweeks for validaion.
    """
    if "gw" not in df.columns:
        raise ValueError("Column 'gw' not foud in training data.")

    unique_gws = sorted(df["gw"].unique())
    if len(unique_gws) < 3:
        raise ValueError("Not enough different gameweeks for a time-based split.")

    split_index = int(len(unique_gws) * (1 - test_ratio))
    split_gw = unique_gws[split_index]

    train_df = df[df["gw"] <= split_gw].copy()
    valid_df = df[df["gw"] > split_gw].copy()

    return train_df, valid_df, split_gw

#---------------------------------------------------------------------
# Walk-forward backtest
#---------------------------------------------------------------------

def walk_forward_backtest() -> pd.DataFrame:
    """
    Walk-forward backtest:
        Train on GWs ≤ t
        Predict on GW t+1

    Saves results to results/ml_backtests/walk_forward_results.csv
    """
    print("Running walk-forward backtest on player_gw_training...")

    df = load_training_data()

    if "gw" not in df.columns:
        raise ValueError("Column 'gw' missing.")

    unique_gws = sorted(df["gw"].unique())
    if len(unique_gws) < 3:
        raise ValueError("Not enough distinct gameweeks for walk-forward backtest.")

    results = []

    for i in range(1, len(unique_gws)):
        train_gws = unique_gws[:i]
        test_gw = unique_gws[i]

        train_df = df[df["gw"].isin(train_gws)].copy()
        test_df = df[df["gw"] == test_gw].copy()

        if len(test_df) == 0:
            continue

        X_train, y_train, feature_cols = prepare_features(train_df)
        X_test, y_test, _ = prepare_features(test_df)

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            "train_up_to_gw": int(max(train_gws)),
            "test_gw": int(test_gw),
            "mae": float(mae),
            "avg_pred": float(y_pred.mean()),
            "avg_actual": float(y_test.mean()),
            "n_samples": int(len(test_df)),
        })

        print(f"GW{test_gw}: MAE={mae:.3f}")

    results_df = pd.DataFrame(results)
    out_path = BACKTEST_DIR / "walk_forward_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved walk-forward backtest to {out_path}")

    return results_df

#---------------------------------------------------------------------
#Train model
#---------------------------------------------------------------------
def train_model():
    print(f"Loading training data from {TRAIN_CSV}")
    df = load_training_data()

    #Time-based split
    train_df, valid_df, split_gw = time_based_split(df, test_ratio=0.1)
    print(f"Using Gws <= {split_gw} for training, > {split_gw} for validation.")

    #Prepare features
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_valid, y_valid, _ = prepare_features(valid_df)

    #define model (tune later)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    #Evaluate on validation set:
    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    mse = mean_squared_error(y_valid, y_pred)
    rmse = mse ** 0.5

    # ---------------------------------------------
    # Plot Actual vs Predicted
    # ---------------------------------------------
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,6))
    plt.scatter(y_valid, y_pred, alpha=0.3)
    plt.xlabel("Actual FPL Points")
    plt.ylabel("Predicted FPL Points")
    plt.title("Actual vs Predicted FPL Points")
    plt.grid(True)
    plt.savefig(MODELS_DIR / "actual_vs_predicted.png")
    print("Saved validation plot to models/ml_models/actual_vs_predicted.png")
    """

    print("\n=== Validation performance===")
    print(f"MAE : {mae:.3f}")
    print(F"RMSE: {rmse:.3f}")

    #save model + metadata
    model_path = MODELS_DIR / "rf_points_model.pkl"
    meta_path = MODELS_DIR / "rf_points_model_metadata.json"

    joblib.dump(
        {"model": model,
        "feature_cols": feature_cols,
        },
        model_path,
    )
    print(f"\nSaved trained model to: {model_path}")

    #Save a small JSON metadata file
    import json 
    
    metadata = {
        "type": "RandomForestRegressor",
        "n_estimators": 300,
        "max_depth": 10,
        "split_gw": int(split_gw),
        "mae_valid": float(mae),
        "rmse_valid": float(rmse),
        "feature_cols": feature_cols,
        "notes": (
            "Per-match stats features (minutes, goals, etc.) and lagged by 1 GW"
            "in the training table; the model therefore uses only information available before the target gameweek."
        ),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "both"],
        default="train",
        help="train = train model (default), backtest = walk-forward backtest, both = train then backtest."
    )

    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train_model()

    if args.mode in ("backtest", "both"):
        walk_forward_backtest()


if __name__ == "__main__":
    main()

    #python -m src.ml.train_model --mode backtest