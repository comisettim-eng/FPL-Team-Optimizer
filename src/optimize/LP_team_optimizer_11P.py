#Here, we add the fact that out of the 15 players in our squad, only 11 can play and earn points, that includes only one GK, and a Captain (whose points are dooubled)
#Before gameweek 12
#Add a constraint so that the optimizer does not choose one game wonders or players who no longer play in the league (such as Matt O'Riley )

from pathlib import Path
from typing import Tuple

import pandas as pd
import pulp as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PLAYERS_CSV = DATA_DIR / "players_clean.csv"

def load_players(
    path: Path = PLAYERS_CSV,
    min_minutes: int = 270,
    only_available: bool = True,
) -> pd.DataFrame:
    """load the cleaned players dataset and apply basic filters so that the optimizer doesn't pick players who never / no longer play.
    
    Parameters
    ----------
    min_minutes : int 
        minimum total minutes played this season to keep a player (=270 -> 3 full games)
    Only available : bool
        If True and a 'status' column exists, keep only players with status == 'a' (available in FPL API)
    
    """
    df = pd.read_csv(path)

    #Basic cleaning for the optimizer
    df = df[df["minutes"] >= min_minutes].copy() #remove players with less than 270 minuites played

    if only_available and "status" in df.columns:
        df = df[df["status"].isin(["a", "d", "s"])].copy() #Optionally drop players who are no longer available (transferred / long term injury, etc.)

    return df

def build_team_optimizer(
    players: pd.DataFrame,
    budget: float = 100.0,
    objective_col: str = "points_per_game",
    min_minutes_season: int = 270,
    min_minutes_share_5: float = 0.40,
    min_games_played_5: float = 2.0,
    only_available: bool = True,
) -> Tuple[pd.DataFrame, pl.LpProblem]:
    """
    Build a single gameweek optimization problem.

    PARAMETERS
    ----------
    players : DataFrame
        Must contain columns: [player_id, name, team_id, team_name, position, price]
        plus a column used as objective (e.g. 'points_per_game' or 'predicted_points').
        Optionally may contain:
        - 'minutes' (total season minutes)
        - 'minutes_share_5', 'games_played_5' (recent "nailedness" features)
        - 'status' (for availability: a/d/s etc.)
    
    Returns
    ---------
    selected: Dataframe
        Subset of players representing the optimal 15 man squad
    model: pulp.LpProblem
        The solved optimization model (for inspection / debugging)
        """
    df = players.copy()

    #Basic fetaures to remove non-nailed / unavailable players

    # 1) Filter by season minutes if available
    if min_minutes_season > 0 and "minutes" in df.columns:
        df = df[df["minutes"] >= min_minutes_season].copy()

    # 2) Filter by recent minutes_share_5 (nailedness over last 5 GWs)
    if min_minutes_share_5 > 0 and "minutes_share_5" in df.columns:
        df = df[df["minutes_share_5"] >= min_minutes_share_5].copy()

    # 3) Filter by recent games_played_5
    if min_games_played_5 > 0 and "games_played_5" in df.columns:
        df = df[df["games_played_5"] >= min_games_played_5].copy()

    # 4) Filter by availability status
    if only_available and "status" in df.columns:
        df = df[df["status"].isin(["a", "d", "s"])].copy()

    #check that chosen variable exists 
    if objective_col not in df.columns:
        raise ValueError(f"objective column '{objective_col}' not found in DataFrame")

    #use index as internal ID 
    df = df.reset_index(drop=True)
    indices = df.index.tolist()

    # Decision Variables: x_i = 1 if player is selected, 0 otherwise)
    x = pl.LpVariable.dicts("select", indices, lowBound=0, upBound=1, cat="Binary") #in squad
    y = pl.LpVariable.dicts("start", indices, lowBound=0, upBound=1, cat="Binary") # in starting XI 
    c = pl.LpVariable.dicts("captain", indices, lowBound=0, upBound=1, cat="Binary") # Captain

    #Define the problem: maximizing total projected points
    model = pl.LpProblem("FPL_Team_Selection", pl.LpMaximize)

    # Objective function: Maximize points of Starters Only
    model += pl.lpSum(df.loc[i, objective_col] * (y[i] + c[i]) for i in indices), "Total_Projected_Points"

    #Constraint: Budget of full squad
    model += pl.lpSum(df.loc[i, "price"] * x[i] for i in indices) <= budget, "Budget"

    #Constraint: Exactly 15 players in the full squad
    model += pl.lpSum(x[i] for i in indices) == 15, "Squad_Size"

    #Positional Constraints 
    for pos, required in {"GK": 2, "DEF": 5, "MID": 5, "FWD" : 3 }. items():
        model += (
            pl.lpSum(x[i] for i in indices if df.loc[i, "position"] == pos) == required,
            f"{pos}_squad_count",
        )

    #Only 11 players start and earn points
    model += pl.lpSum(y[i] for i in indices) == 11, "Starting_XI_Size"

    #Exactly 1 GK in the starting 11
    model += (
        pl.lpSum(y[i] for i in indices if df.loc[i, "position"] == "GK") == 1,
        "Starting_GK_Count"
    )

    # Minimum positional requirements in the Starting XI
    model += (
        pl.lpSum(y[i] for i in indices if df.loc[i, "position"] == "DEF") >= 3,
        "Min_Starting_DEF"
    )
    model += (
        pl.lpSum(y[i] for i in indices if df.loc[i, "position"] == "MID") >= 2,
        "Min_Starting_MID"
    )
    model += (
        pl.lpSum(y[i] for i in indices if df.loc[i, "position"] == "FWD") >= 1,
        "Min_Starting_FWD"
    )

    #A player can only start if he is in the squad
    for i in indices:
        model += y[i] <= x[i], f"start_only_if_selected_{i}"

    #Exactly 1 Captain
    model += pl.lpSum(c[i] for i in indices) == 1, "Captain_Count"

    #Captain must be a starter
    for i in indices:
        model += c[i] <= y[i], f"captain_must_start_{i}"

    #max 3 players per real-life team (ex. Max 3 players from FC Liverpool)
    for team in df["team_name"].unique():
        model += (
            pl.lpSum(x[i] for i in indices if df.loc[i,"team_name"] == team) <= 3,
            f"Max_3_from{team}",
        )
    # Solve the Model
    model.solve(pl.PULP_CBC_CMD(msg=False))

    #Collect selected players
    selected_idx = [i for i in indices if x[i].value() == 1]
    selected = df.loc[selected_idx].copy()
    selected["is_starter"] = False
    selected["is_captain"] = False

    #Mark starters
    starter_idx = [i for i in indices if y[i].value() == 1]
    selected.loc[selected.index.isin(starter_idx), "is_starter"] = True

    #Mark Captain
    captain_idx = [i for i in indices if c[i].value() == 1]
    selected.loc[selected.index.isin(captain_idx), "is_captain"] = True

    #sort nicely for display
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    selected["pos_sort"] = selected["position"].map(pos_order)
    selected.sort_values(["is_starter", "pos_sort"], ascending=[False, True], inplace=True)

    selected = selected.drop(columns=["pos_sort"])

    #add objective contribution for inspection
    selected["objective_value"] = selected[objective_col] * selected["is_starter"].astype(int) * (1 + selected["is_captain"].astype(int))

    return selected, model

def main():
    players = load_players()
    best_team, model = build_team_optimizer(
        players, 
        budget=100.0, 
        objective_col="points_per_game",
        min_minutes_season=270,      # avoid extreme one-game wonders
        min_minutes_share_5=0.40,     # set >0 if column exists
        min_games_played_5=2,      # set >0 if column exists
        only_available=True,
    )
    print("Status:", pl.LpStatus[model.status])
    print("\nTotal projected points:", pl.value(model.objective))
    print("\nOptimal squad:")
    cols_to_show = [
        "is_starter",
        "is_captain",
        "position",
        "name",
        "team_name",
        "price",
        "points_per_game",
        "points_per_90",
        "points_per_million",
    ]
    print (best_team[cols_to_show].to_string(index=False))

    # === Save optimized team to results/lp_solutions/ ===
    OUTPUT_DIR = PROJECT_ROOT / "results" / "lp_solutions"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "best_team_lp_GW11.csv"
    best_team.to_csv(output_path, index=False)
    print(f"\nSaved optimized team to: {output_path}")

    #=== Save starting 11 only ===
    starting_xi = best_team[best_team["is_starter"] == True].copy()
    xi_path = OUTPUT_DIR / "starting_xi_lp_GW11.csv"
    starting_xi.to_csv(xi_path, index=False)
    print(f"Saved starting XI to: {xi_path}")

if __name__ == "__main__":
    main()