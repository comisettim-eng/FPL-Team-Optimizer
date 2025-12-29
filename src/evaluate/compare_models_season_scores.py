import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

# ---------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "ml_season_solutions"
BACKTEST_DIR = PROJECT_ROOT / "results" / "ml_backtests"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

FPL_API_BASE = "https://fantasy.premierleague.com/api"

# Match pipeline
CURRENT_SEASON = "2025-26"

# the three models
MODELS = ["rf", "lgbm", "xgb"]
MODEL_NAME = {
    "rf": "RandomForest",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
}

# caches to avoid repeated HTTP calls
_live_cache: Dict[int, pd.DataFrame] = {}
_avg_cache: Dict[int, float] = {}


# ---------------------------------------------------------------------
# FPL API HELPERS
# ---------------------------------------------------------------------
def fetch_gw_live_points(gw: int) -> pd.DataFrame:
    """
    Fetch actual FPL points for all players for a given GW from
    /event/{gw}/live.
    """
    if gw in _live_cache:
        return _live_cache[gw].copy()

    url = f"{FPL_API_BASE}/event/{gw}/live/"
    resp = requests.get(url, timeout=15)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch FPL live data for GW {gw}: HTTP {resp.status_code}"
        )

    data = resp.json()
    elements = data.get("elements", [])

    rows = []
    for el in elements:
        stats = el.get("stats", {}) or {}
        rows.append(
            {
                "player_id": el.get("id"),
                "total_points": stats.get("total_points", 0),
                "minutes": stats.get("minutes", 0),
                "goals_scored": stats.get("goals_scored", 0),
                "assists": stats.get("assists", 0),
                "clean_sheets": stats.get("clean_sheets", 0),
                "goals_conceded": stats.get("goals_conceded", 0),
                "saves": stats.get("saves", 0),
                "bonus": stats.get("bonus", 0),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        df["total_points"] = (
            pd.to_numeric(df["total_points"], errors="coerce").fillna(0.0)
        )

    _live_cache[gw] = df.copy()
    return df


def fetch_fpl_average_scores_from_bootstrap() -> Dict[int, float]:
    """
    Fetch mapping gameweek -> average_entry_score from /bootstrap-static.
    """
    if _avg_cache:
        return _avg_cache

    url = f"{FPL_API_BASE}/bootstrap-static/"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    events = data.get("events", [])

    for ev in events:
        gw = ev.get("id")
        avg = ev.get("average_entry_score")
        if gw is not None and avg is not None:
            _avg_cache[int(gw)] = float(avg)

    return _avg_cache


# ---------------------------------------------------------------------
# HELPER: load starting XI for a model & GW
# ---------------------------------------------------------------------
def load_starting_xi(model_key: str, gw: int) -> pd.DataFrame:
    """
    Load starting XI for a given model & GW.
    """
    gw_tag = f"GW{gw:02d}"
    xi_path = RESULTS_DIR / model_key / f"{model_key}_starting_xi_{gw_tag}.csv"

    if not xi_path.exists():
        raise FileNotFoundError(
            f"Starting XI file not found for model={model_key}, GW={gw}: {xi_path}\n"
            f"Run the season optimizer with end_gw >= {gw}."
        )

    df = pd.read_csv(xi_path)

    if "player_id" not in df.columns:
        raise ValueError(f"Column 'player_id' missing in {xi_path}")
    if "is_captain" not in df.columns:
        raise ValueError(f"Column 'is_captain' missing in {xi_path}")

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")

    # boolean parsing
    df["is_captain"] = (
        df["is_captain"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    )

    return df[["player_id", "is_captain"]].copy()


# ---------------------------------------------------------------------
# HELPER: compute actual score for a model's team in a GW
# ---------------------------------------------------------------------
def compute_model_score_for_gw(model_key: str, gw: int) -> float:
    """
    Compute model's actual points (real FPL points from API).
    """
    xi = load_starting_xi(model_key, gw)
    live = fetch_gw_live_points(gw)

    merged = xi.merge(
        live[["player_id", "total_points"]],
        on="player_id",
        how="left"
    )

    merged["total_points"] = merged["total_points"].fillna(0.0)
    merged["multiplier"] = 1 + merged["is_captain"].astype(int)
    merged["effective_points"] = merged["total_points"] * merged["multiplier"]

    return float(merged["effective_points"].sum())


# ---------------------------------------------------------------------
# CORE: comparison over a range of GWs
# ---------------------------------------------------------------------
def compare_models_over_gws(start_gw: int = 2, end_gw: int = 12) -> pd.DataFrame:
    """
    Compare actual season performance for all ML models vs FPL average.
    """
    if start_gw < 2:
        raise ValueError("start_gw must be >= 2, because ML optimizer starts GW2.")

    # fetch FPL averages once
    try:
        avg_map = fetch_fpl_average_scores_from_bootstrap()
    except Exception as e:
        print(f"[WARNING] Could not fetch FPL averages: {e}")
        avg_map = {}

    rows = []
    totals = {m: 0.0 for m in MODELS}
    total_avg = 0.0

    for gw in range(start_gw, end_gw + 1):
        print(f"\n=== Computing scores for GW {gw} ===")

        row = {"gameweek": gw}

        # actual points for each model
        for model_key in MODELS:
            score = compute_model_score_for_gw(model_key, gw)
            row[f"{model_key}_score"] = score
            totals[model_key] += score

            print(f"  {MODEL_NAME[model_key]}: {score:.2f}")

        # FPL global average
        if gw in avg_map:
            avg = avg_map[gw]
            row["fpl_average_score"] = avg
            total_avg += avg
            print(f"  FPL average: {avg:.2f}")
        else:
            row["fpl_average_score"] = float("nan")
            print("  FPL average: N/A")

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("gameweek").reset_index(drop=True)

    # summary
    print("\n=== TOTAL SCORES ===")
    for m in MODELS:
        print(f"{MODEL_NAME[m]} total: {totals[m]:.2f}")

    if avg_map:
        print(f"FPL average total: {total_avg:.2f}")
    else:
        print("FPL average total: (missing)")

    out_path = BACKTEST_DIR / f"model_season_scores_api_gw{start_gw}_{end_gw}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved comparison to: {out_path}")
    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare ML model season scores vs real FPL points & FPL average."
    )

    parser.add_argument("--start_gw", type=int, default=2)
    parser.add_argument("--end_gw", type=int, default=12)

    args = parser.parse_args()
    compare_models_over_gws(args.start_gw, args.end_gw)


if __name__ == "__main__":
    main()