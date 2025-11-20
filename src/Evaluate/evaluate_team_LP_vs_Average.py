import sys
from pathlib import Path

import pandas as pd
import requests



#import existing functions
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.optimize.LP_team_optimizer_11P import (
    load_players,
    build_team_optimizer,
)

BASE_URL = "https://fantasy.premierleague.com/api"

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

def get_average_scores_by_gw(gw_ids):
    """Return dict: gw_id -> average_entry_score for the specified gameweeks.
    
    Uses the 'events' list from bootstrap-static, which already contains average_entry_score for each gameweek.
    """

    events = get_events()
    # Build quick lookup: id -> event dict
    event_by_id = {e["id"]: e for e in events}

    avg_by_gw = {}
    for gw in gw_ids:
        event = event_by_id.get(gw)
        if event is not None:
            avg_by_gw[gw] = event.get("average_entry_score", 0)
        else: #If for some reason the GW is not present -> retrun 0
            avg_by_gw[gw] = 0
    return avg_by_gw


def get_player_history(player_id: int):
    """
    Get per-gameweek history for a player.
    Return dict: gw_id -> total_points in that GW.
    """
    resp = requests.get(f"{BASE_URL}/element-summary/{player_id}/")
    resp.raise_for_status()
    data = resp.json()
    history = data.get("history", [])
    return {entry["round"]: entry["total_points"]for entry in history}


#TEAM POINTS CALCULATIONS

def compute_team_points_by_gw(best_team: pd.DataFrame, finished_gws):
    """
    Compute GW-by-GW realized points for optimized team, assuming:
    - Same starting XI every GW (is_starter == True)
    - Same Captain every GW (is_captain == True)
    - use per-player GW history from FPL API.
    Returns dict: gw_id -> team_points_in_that_gameweek.
    """
    starters = best_team[best_team["is_starter"]].copy()
    captain_row = starters[starters["is_captain"]]
    if len(captain_row) != 1:
        raise ValueError("Expected exactly one captain in best_team")

    captain_player_id = int(captain_row["player_id"].iloc[0])

    #Cach per-player histories to avoid repeated API calls
    history_cache: dict[int, dict[int, int]] = {}

    team_points_by_gw: dict[int,float] = {}

    for gw in finished_gws:
        gw_total = 0.0
        for _, row in starters.iterrows():
            pid = int(row["player_id"])
            if pid not in history_cache:
                history_cache[pid] = get_player_history(pid)

            player_history = history_cache[pid]
            pts = player_history.get(gw,0)

            # Double points if this is the captain 

            if pid == captain_player_id:
                gw_total += 2 * pts
            else:
                gw_total += pts

        team_points_by_gw[gw] = gw_total

    return team_points_by_gw

def compute_overall_team_points (team_points_by_gw: dict) -> float:
    """Sum GW-by-GW points to get season total"""
    return float(sum(team_points_by_gw.values()))

def compute_projected_team_points_by_gw (best_team: pd.DataFrame, future_gws):
    """
    Very simple projection for future gameweeks:
    - Use points_per_game as expected GW points.
    - Starters score their points_per_game
    - Captain gets double.
    Returns dict: gw_id -> projected_team_points_in_that_gw.
    """
    starters = best_team[best_team["is_starter"]].copy()
    captain_row = starters[starters["is_captain"]]
    if len(captain_row) != 1:
        raise ValueError("Expected exactly one captain in best_team")
        
    captain_player_id = int(captain_row["player_id"].iloc[0])

    team_proj_by_gw: dict[int, float] = {}

    for gw in future_gws:
        gw_total = 0.0
        for _, row in starters.iterrows():
            base = float(row["points_per_game"])
            if int(row["player_id"]) == captain_player_id:
                gw_total += 2 * base
            else:
                gw_total += base
        team_proj_by_gw[gw] = gw_total

    return team_proj_by_gw


# MAIN EVALUATION LOGIC 

def main():
    # 1. Build optimized team using existing optimizer
    players = load_players()
    best_team, model = build_team_optimizer(
        players,
        budget=100.0,
        objective_col="points_per_game",
    )

    finished_gws, future_gws = get_finished_and_future_gameweeks()

    # 2. REALIZED : team vs official FPL average so far 
    print ("Fetching realized performance...")
    team_points_by_gw = compute_team_points_by_gw(best_team, finished_gws)
    team_total_realized = compute_overall_team_points(team_points_by_gw)

    avg_by_gw = get_average_scores_by_gw(finished_gws)
    avg_total_realized = float(sum(avg_by_gw.values()))

    diff_total = team_total_realized - avg_total_realized
    perc_total = (
        diff_total / avg_total_realized * 100 if avg_total_realized > 0 else float("nan")
        )

    print("\n=== Overall Comparison (realized so far) ===")
    print(f"Finished gameweeks:                         {len(finished_gws)}")
    print(f"Optimized team total points (with captain): {team_total_realized:.1f} ")
    print(f"Official FPL average total points:          {avg_total_realized:.1f}")
    print(f"Difference:                                 {diff_total:.1f}")
    print(f"Relative improvement:                       {perc_total:.1f}%")

#week-by-week table 
    df_gw = pd.DataFrame(
        {
            "gw":finished_gws,
            "team_points": [team_points_by_gw.get(gw, 0.0) for gw in finished_gws],
            "average_points": [avg_by_gw.get(gw, 0.0) for gw in finished_gws],
        }
    )
    df_gw["difference"] = df_gw["team_points"] - df_gw["average_points"]

    print("=== Week-by-week comparison (realized) ===")
    print(df_gw.to_string(index=False))

    #Future Projection

    if future_gws:
        print("\nProjecting future gameweeks...")

        team_proj_by_gw = compute_projected_team_points_by_gw(best_team, future_gws)

        #simple baseline for future average: mean of past average scores 
        avg_past_mean = df_gw["average_points"].mean() if not df_gw.empty else 0.0
        avg_future_by_gw = {
            gw: avg_past_mean for gw in future_gws
        } #Flat projection for FPL average

        df_future = pd.DataFrame(
            {
                "gw": future_gws,
                "team_proj_points":[team_proj_by_gw.get(gw, 0.0) for gw in future_gws],
                "avg_proj_points": [avg_future_by_gw.get(gw, 0.0) for gw in future_gws],
            }
        )
        df_future["difference"] = (
            df_future["team_proj_points"] - df_future["avg_proj_points"]
        )

        print("\n=== Future gameweeks projection ===")
        print(df_future.to_string(index=False))

        future_team_total = df_future["team_proj_points"].sum()
        future_avg_total = df_future["avg_proj_points"].sum()
        future_diff = future_team_total - future_avg_total
        future_perc = (
            future_diff / future_avg_total * 100 if future_avg_total > 0 else float ("nan")
        )

        print("\n=== Overall Comparison (Projection) ===")
        print(f"Number of future gameweeks:                 {len(future_gws)}")
        print(f"Projected team total points (with captain): {future_team_total:.1f} ")
        print(f"Projected Official FPL average total points:{future_avg_total:.1f}")
        print(f"Difference:                                 {future_diff:.1f}")
        print(f"Relative improvement:                       {future_perc:.1f}%")

        #full season table with realized + future slots + projections 
        #Realized part: add empty projection columns 
        df_real = df_gw.copy()
        df_real["team_proj_points"] = pd.NA
        df_real["avg_proj_points"] = pd.NA
        df_real["proj_difference"] = pd.NA

        #Future slots: mp realized data yet, but projections filled
        df_future_slots = df_future.copy()
        df_future_slots = df_future_slots.rename(columns={"difference": "proj_difference"})
        df_future_slots = pd.DataFrame(
            {
                "gw":df_future_slots["gw"],
                "team_points": [pd.NA] * len(df_future_slots),
                "average_points": [pd.NA] * len(df_future_slots),
                "difference": [pd.NA] * len(df_future_slots),
                "team_proj_points": df_future_slots["team_proj_points"].tolist(),
                "avg_proj_points": df_future_slots["avg_proj_points"].tolist(),
                "proj_difference": df_future_slots["proj_difference"].tolist(),
            }
        )

        df_full = pd.concat([df_real, df_future_slots], ignore_index=True)

        #cumulative realized gap (future stays flat until real data arrives)
        df_full["cum_team_points"] = df_full["team_points"].fillna(0).cumsum()
        df_full["cum_average_points"] = df_full["average_points"].fillna(0).cumsum()
        df_full["cum_difference"] = (
            df_full["cum_team_points"] - df_full["cum_average_points"]
        )

        print("\n=== Full season table (realized + future slots + projections) ===")
        print(df_full.to_string(index=False))

    else:
        print("\nNo future gameweeks found in API (season may be finished).")

if __name__ == "__main__":
    main()


#EVALUATION

#World Position after gameweek 11 of optimized team = 5
#does not take into account chips (Free hit, Triple Captain, or transfers (one transfer allowed per week)
#Not surpising it is better than most players over the past gameweeks as that is the data it was based off of, but it will be interesting to see how this team performs over the future gameweeks