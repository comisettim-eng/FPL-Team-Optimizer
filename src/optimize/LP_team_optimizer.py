from pathlib import Path
from typing import Tuple

import pandas as pd
import pulp as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PLAYERS_CSV = DATA_DIR / "players_clean.csv"

def load_players(path: Path = PLAYERS_CSV) -> pd.DataFrame:
    """load the cleaned players dataset"""
    df = pd.read_csv(path)

    #Basic cleaning for the optimizer
    df = df[df["minutes"] > 0].copy() #remove players with 0 minuites played

    return df

def build_team_optimizer(
    players: pd.DataFrame,
    budget: float = 100.0,
    objective_col: str = "points_per_game",
) -> Tuple[pd.DataFrame, pl.LpProblem]:
    """
    Build a single gameweek optimization problem.

    PARAMETERS
    ----------
    players : Dataframe 
        Must contain columns: [Player id, name, team_id, team_name, position, price, points_per_game, points_per_90, points_per_million]
    Budget: FLoat
        Total budget in millions (default = 100.0).
    Objective_col : str 
        Column used as objective (points_per_game)
    
    Returns
    ---------
    selected: Dataframe
        Subset of players representing the optimal 15 man squad
    model: pulp.LpProblem
        The solved optimization model (for inspection / debugging)
        """
    df = players.copy()

    #check that chosen variable exists 
    if objective_col not in df.columns:
        raise ValueError(f"objective column '{objective_col}' not found in DataFrame")

    #use index as internal ID 
    df = df.reset_index(drop=True)
    indices = df.index.tolist()

    # Decision Variables: x_i = 1 if player is selected, 0 otherwise)
    x = pl.LpVariable.dicts("select", indices, lowBound=0, upBound=1, cat="Binary")

    #Define the problem: maximizing total projected points
    model = pl.LpProblem("FPL_Team_Selection", pl.LpMaximize)

    # Objective function
    model += pl.lpSum(df.loc[i, objective_col] * x[i] for i in indices), "Total_Projected_Points"

    #Constraint: Total cost within budget
    model += pl.lpSum(df.loc[i, "price"] * x[i] for i in indices) <= budget, "Budget"

    #Constraint: Exactly 15 players
    model += pl.lpSum(x[i] for i in indices) == 15, "Squad_Size"

    #Positional Constraints 
    for pos, required in {"GK": 2, "DEF": 5, "MID": 5, "FWD" : 3 }. items():
        model += (
            pl.lpSum(x[i] for i in indices if df.loc[i, "position"] == pos) == required,
            f"{pos}_count",
        )
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

    #sort nicely for display
    selected.sort_values(["position", "team_name", "name"], inplace=True)

    #add objective contribution for inspection
    selected["objective_value"] = selected[objective_col]

    return selected, model

def main():
    players = load_players()
    best_team, model = build_team_optimizer(players, budget=100.0, objective_col="points_per_game")
    print("Status:", pl.LpStatus[model.status])
    print("\nTotal projected points:", pl.value(model.objective))
    print("\nOptimal squad:")
    cols_to_show = [
        "position",
        "name",
        "team_name",
        "price",
        "points_per_game",
        "points_per_90",
        "points_per_million",
    ]
    print (best_team[cols_to_show].to_string(index=False))

if __name__ == "__main__":
    main()