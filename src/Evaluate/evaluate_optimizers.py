import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

# ---------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.optimize.LP_team_optimizer_11P import (
    load_players,
    build_team_optimizer,
)

from src.optimize.ML_team_optimizer_11P import (
    load_predictions_for_gw,
    optimize_team_from_predictions,
)

BASE_URL = "https://fantasy.premierleague.com/api"

# ---------------------------------------------------------------------
# FPL helpers
# ---------------------------------------------------------------------

def get_events():
    """Fetch event metadata (gameweeks) from bootstrap-static."""
    resp = requests.get(f"{BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    data = resp.json()
    return data["events"]


def get_finished_and_future_gameweeks():
    """Return (finished_gws, future_gws) as list of GW ids."""
    events = get_events()
    finished = [e["id"] for e in events if e.get("finished")]
    future = [e["id"] for e in events if not e.get("finished")]
    finished.sort()
    future.sort()
    return finished, future


def get_average_scores_by_gw(gw_ids: List[int]) -> Dict[int, int]:
    """Return dict: gw_id -> average_entry_score for the specified gameweeks."""
    events = get_events()
    event_by_id = {e["id"]: e for e in events}

    avg_by_gw: Dict[int, int] = {}
    for gw in gw_ids:
        event = event_by_id.get(gw)
        if event is not None:
            avg_by_gw[gw] = event.get("average_entry_score", 0)
        else:
            avg_by_gw[gw] = 0
    return avg_by_gw


# ---------------------------------------------------------------------
# Player history helpers (with caching)
# ---------------------------------------------------------------------

_history_cache: Dict[int, Dict[int, int]] = {}


def get_player_history(player_id: int) -> Dict[int, int]:
    """
    Get per-gameweek history for a player.
    Return dict: gw_id -> total_points in that GW.
    Uses a simple cache to avoid repeated API calls.
    """
    if player_id in _history_cache:
        return _history_cache[player_id]

    resp = requests.get(f"{BASE_URL}/element-summary/{player_id}/")
    resp.raise_for_status()
    data = resp.json()
    history = data.get("history", [])
    hist_dict = {entry["round"]: entry["total_points"] for entry in history}
    _history_cache[player_id] = hist_dict
    return hist_dict


def compute_team_points_for_gw(team: pd.DataFrame, gw: int) -> float:
    """
    Compute realized points for a single gameweek for a given team:
    - use is_starter / is_captain columns
    - captain gets double points
    - assumes no subs / autosubs / chips
    """
    starters = team[team["is_starter"]].copy()
    captain_row = starters[starters["is_captain"]]
    if len(captain_row) != 1:
        raise ValueError("Expected exactly one captain in team")

    captain_player_id = int(captain_row["player_id"].iloc[0])

    gw_total = 0.0
    for _, row in starters.iterrows():
        pid = int(row["player_id"])
        player_history = get_player_history(pid)
        pts = player_history.get(gw, 0)

        if pid == captain_player_id:
            gw_total += 2 * pts
        else:
            gw_total += pts

    return gw_total


# ---------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------

def main():
    # --------------------------------------------------------------
    # 1) Build LP baseline team (static for the whole season)
    # --------------------------------------------------------------
    players = load_players()
    lp_team, _ = build_team_optimizer(
        players,
        budget=100.0,
        objective_col="points_per_game",
    )

    finished_gws, _ = get_finished_and_future_gameweeks()
    if not finished_gws:
        print("No finished gameweeks found in API.")
        return

    avg_by_gw = get_average_scores_by_gw(finished_gws)

    # --------------------------------------------------------------
    # 2) For each finished GW:
    #    - compute LP team realized score
    #    - build ML-optimized team for that GW and compute its score
    #    - record FPL average score
    # --------------------------------------------------------------
    rows = []

    for gw in finished_gws:
        print(f"\n=== Evaluating GW {gw} ===")

        # LP team: static squad, same XI & captain all season
        lp_points = compute_team_points_for_gw(lp_team, gw)

        # ML team: optimized fresh for this GW from predictions_gw<gw>.csv
        try:
            preds_df = load_predictions_for_gw(gw)
            ml_team, _ = optimize_team_from_predictions(preds_df, budget=100.0)
            ml_points = compute_team_points_for_gw(ml_team, gw)
        except FileNotFoundError as e:
            print(f"  [WARN] No predictions file for GW {gw}: {e}")
            ml_points = float("nan")

        avg_points = float(avg_by_gw.get(gw, 0.0))

        rows.append(
            {
                "gw": gw,
                "lp_points": lp_points,
                "ml_points": ml_points,
                "average_points": avg_points,
                "lp_minus_avg": lp_points - avg_points,
                "ml_minus_avg": ml_points - avg_points
                if not pd.isna(ml_points)
                else float("nan"),
                "ml_minus_lp": ml_points - lp_points
                if not pd.isna(ml_points)
                else float("nan"),
            }
        )

    df = pd.DataFrame(rows).sort_values("gw").reset_index(drop=True)

    # Cumulative sums
    df["cum_lp_points"] = df["lp_points"].cumsum()
    df["cum_ml_points"] = df["ml_points"].cumsum()
    df["cum_average_points"] = df["average_points"].cumsum()
    df["cum_lp_minus_avg"] = df["cum_lp_points"] - df["cum_average_points"]
    df["cum_ml_minus_avg"] = df["cum_ml_points"] - df["cum_average_points"]

    print("\n=== Week-by-week comparison: LP vs ML vs FPL average ===")
    print(df.to_string(index=False))

    # Overall summary
    total_lp = df["lp_points"].sum()
    total_ml = df["ml_points"].sum()
    total_avg = df["average_points"].sum()

    print("\n=== Overall totals over finished gameweeks ===")
    print(f"Gameweeks evaluated:               {len(df)}")
    print(f"LP team total points:              {total_lp:.1f}")
    print(f"ML team total points:              {total_ml:.1f}")
    print(f"Official FPL average total points: {total_avg:.1f}")
    print(f"LP - average:                      {total_lp - total_avg:.1f}")
    print(f"ML - average:                      {total_ml - total_avg:.1f}")
    print(f"ML - LP:                           {total_ml - total_lp:.1f}")

    # Optional: save to CSV
    eval_dir = PROJECT_ROOT / "results" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "lp_vs_ml_vs_average_by_gw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed comparison to: {out_path}")


if __name__ == "__main__":
    main()
