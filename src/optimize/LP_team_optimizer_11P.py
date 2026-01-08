"""
LP_team_optimizer_11P.py

Build an optimal FPL squad (15 players, 11 starters, 1 captain) with
standard FPL constraints, and optionally enforce a limited number of
transfers compared to a previous squad.

- Can be run standalone (uses players_clean.csv and points_per_game)
- Can be imported and used from ML season optimizers with
  prev_squad_ids + transfers_per_gw to do season-long optimization.
"""

from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd
import pulp as pl

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PLAYERS_CSV = DATA_DIR / "players_clean.csv"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_players(
    path: Path = PLAYERS_CSV,
    min_minutes: int = 0,
    only_available: bool = True,
) -> pd.DataFrame:
    """
    Load the cleaned players dataset and apply basic filters so that the
    optimizer doesn't pick players who never / no longer play.

    Parameters
    ----------
    min_minutes : int
        Minimum total minutes played this season to keep a player (=0)
    only_available : bool
        If True and a 'status' column exists, keep only players with
        status in ['a', 'd', 's'] (available / doubtful / suspended).
    """
    df = pd.read_csv(path)

    # basic cleaning for the optimizer
    if "minutes" in df.columns:
        df = df[df["minutes"] >= min_minutes].copy()

    if only_available and "status" in df.columns:
        df = df[df["status"].isin(["a", "d", "s"])].copy()

    return df


# ---------------------------------------------------------------------
# Core LP optimizer
# ---------------------------------------------------------------------
def build_team_optimizer(
    players: pd.DataFrame,
    budget: float = 100.0,
    objective_col: str = "points_per_game",
    min_minutes_season: int = 0,
    min_minutes_share_5: float = 0.40,
    min_games_played_5: float = 2.0,
    only_available: bool = True,
    prev_squad_ids: Optional[List[int]] = None,
    transfers_per_gw: int = 0,
) -> Tuple[pd.DataFrame, pl.LpProblem]:
    
    df = players.copy()

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

    # Check that chosen objective exists
    if objective_col not in df.columns:
        raise ValueError(f"objective column '{objective_col}' not found in DataFrame")

    # Use index as internal ID
    df = df.reset_index(drop=True)
    indices = df.index.tolist()

    # -----------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------
    x = pl.LpVariable.dicts("select", indices, lowBound=0, upBound=1, cat="Binary")   # in squad
    y = pl.LpVariable.dicts("start", indices, lowBound=0, upBound=1, cat="Binary")    # starting XI
    c = pl.LpVariable.dicts("captain", indices, lowBound=0, upBound=1, cat="Binary")  # captain

    # Define problem
    model = pl.LpProblem("FPL_Team_Selection", pl.LpMaximize)

    # Objective: maximize projected points of starters + captain (double)
    model += pl.lpSum(
        df.loc[i, objective_col] * (y[i] + c[i]) for i in indices
    ), "Total_Projected_Points"

    # -----------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------

    # Budget
    model += pl.lpSum(df.loc[i, "price"] * x[i] for i in indices) <= budget, "Budget"

    # Exactly 15 players in squad
    model += pl.lpSum(x[i] for i in indices) == 15, "Squad_Size"

    # Positional squad constraints
    for pos, required in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
        model += (
            pl.lpSum(x[i] for i in indices if df.loc[i, "position"] == pos) == required,
            f"{pos}_squad_count",
        )

    # Starting XI = 11
    model += pl.lpSum(y[i] for i in indices) == 11, "Starting_XI_Size"

    # Exactly 1 GK in starting XI
    model += (
        pl.lpSum(y[i] for i in indices if df.loc[i, "position"] == "GK") == 1,
        "Starting_GK_Count",
    )

    # Minimum positional requirements in starting XI
    model += (
        pl.lpSum(y[i] for i in indices if df.loc[i, "position"] == "DEF") >= 3,
        "Min_Starting_DEF",
    )
    model += (
        pl.lpSum(y[i] for i in indices if df.loc[i, "position"] == "MID") >= 2,
        "Min_Starting_MID",
    )
    model += (
        pl.lpSum(y[i] for i in indices if df.loc[i, "position"] == "FWD") >= 1,
        "Min_Starting_FWD",
    )

    # A player can only start if he is in the squad
    for i in indices:
        model += y[i] <= x[i], f"start_only_if_selected_{i}"

    # Exactly 1 captain
    model += pl.lpSum(c[i] for i in indices) == 1, "Captain_Count"

    # Captain must be a starter
    for i in indices:
        model += c[i] <= y[i], f"captain_must_start_{i}"

    # Max 3 players per real-life team
    if "team_name" in df.columns:
        for team in df["team_name"].unique():
            model += (
                pl.lpSum(x[i] for i in indices if df.loc[i, "team_name"] == team) <= 3,
                f"Max_3_from_{team}",
            )

    # -----------------------------------------------------------------
    # Transfer constraint
    # -----------------------------------------------------------------
    if prev_squad_ids is not None and transfers_per_gw > 0:
        # map previous squad ids to current index set
        prev_indices = [i for i in indices if df.loc[i, "player_id"] in prev_squad_ids]
        if prev_indices:
            required_kept = max(len(prev_indices) - transfers_per_gw, 0)
            model += (
                pl.lpSum(x[i] for i in prev_indices) == required_kept,
                "Transfer_Limit_From_Previous_Squad",
            )

    # -----------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------
    if pl.COIN_CMD().available():
        solver = pl.COIN_CMD(msg=False)
    else:
        solver = pl.PULP_CBC_CMD(msg=False)

    model.solve(solver)


    # Collect selected players
    selected_idx = [i for i in indices if x[i].value() == 1]
    selected = df.loc[selected_idx].copy()
    selected["is_starter"] = False
    selected["is_captain"] = False

    # Mark starters
    starter_idx = [i for i in indices if y[i].value() == 1]
    selected.loc[selected.index.isin(starter_idx), "is_starter"] = True

    # Mark captain
    captain_idx = [i for i in indices if c[i].value() == 1]
    selected.loc[selected.index.isin(captain_idx), "is_captain"] = True

    # Sort nicely
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    selected["pos_sort"] = selected["position"].map(pos_order)
    selected.sort_values(["is_starter", "pos_sort"], ascending=[False, True], inplace=True)
    selected = selected.drop(columns=["pos_sort"])

    # Add objective contribution for inspection
    selected["objective_value"] = (
        selected[objective_col]
        * selected["is_starter"].astype(int)
        * (1 + selected["is_captain"].astype(int))
    )

    return selected, model


# ---------------------------------------------------------------------
# Standalone main (single-GW optimization using points_per_game)
# ---------------------------------------------------------------------
def main():
    players = load_players()
    best_team, model = build_team_optimizer(
        players,
        budget=100.0,
        objective_col="points_per_game",
        min_minutes_season=0,
        min_minutes_share_5=0.40,
        min_games_played_5=2.0,
        only_available=True,
        prev_squad_ids=None,
        transfers_per_gw=0,
    )

    print("Status:", pl.LpStatus[model.status])
    print("\nTotal projected points:", pl.value(model.objective))

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
    print("\nOptimal squad (single GW demo):")
    print(best_team[cols_to_show].to_string(index=False))

    output_dir = PROJECT_ROOT / "results" / "lp_solutions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "best_team_lp_demo.csv"
    best_team.to_csv(output_path, index=False)
    print(f"\nSaved optimized team to: {output_path}")

    xi_path = output_dir / "starting_xi_lp_demo.csv"
    best_team[best_team["is_starter"]].to_csv(xi_path, index=False)
    print(f"Saved starting XI to: {xi_path}")


if __name__ == "__main__":
    main()