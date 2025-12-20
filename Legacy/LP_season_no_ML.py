from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import pulp as pl

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PLAYERS_CSV = DATA_DIR / "players_clean.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "lp_solutions_no_ml"


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------
def load_players(
    path: Path = PLAYERS_CSV,
    min_minutes: int = 270,
    only_available: bool = True,
) -> pd.DataFrame:
    """
    Load the cleaned players dataset and apply basic filters so that the
    optimizer doesn't pick players who never / no longer play.

    Parameters
    ----------
    min_minutes : int
        Minimum total minutes played this season to keep a player (=270 -> 3 full games).
    only_available : bool
        If True and a 'status' column exists, keep only players with status in
        ['a', 'd', 's'] (available, doubtful, suspended).
    """
    df = pd.read_csv(path)

    # Basic cleaning for the optimizer
    df = df[df["minutes"] >= min_minutes].copy()

    if only_available and "status" in df.columns:
        df = df[df["status"].isin(["a", "d", "s"])].copy()

    return df


# -------------------------------------------------------------------
# CORE LP MODEL
# -------------------------------------------------------------------
def build_team_optimizer(
    players: pd.DataFrame,
    budget: float = 100.0,
    objective_col: str = "points_per_game",
    min_minutes_season: int = 270,
    min_minutes_share_5: float = 0.40,
    min_games_played_5: float = 2.0,
    only_available: bool = True,
    prev_squad_ids: Optional[List[int]] = None,
    transfers_per_gw: int = 0,
) -> Tuple[pd.DataFrame, pl.LpProblem]:
    """
    Build a single-gameweek optimization problem.

    Parameters
    ----------
    players : DataFrame
        Must contain columns: [player_id, name, team_id, team_name, position, price]
        plus a column used as objective (e.g. 'points_per_game').
        Optionally may contain:
        - 'minutes' (total season minutes)
        - 'minutes_share_5', 'games_played_5' (recent "nailedness" features)
        - 'status' (for availability: a/d/s etc.)

    prev_squad_ids : list[int] or None
        If provided and transfers_per_gw > 0, we enforce that exactly
        len(prev_squad_ids) - transfers_per_gw of these players remain
        in the new 15-man squad (i.e. exactly transfers_per_gw players are changed).

    Returns
    -------
    selected : DataFrame
        Subset of players representing the optimal 15-man squad.
    model : pulp.LpProblem
        The solved optimization model.
    """
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

    # Check that chosen variable exists
    if objective_col not in df.columns:
        raise ValueError(f"objective column '{objective_col}' not found in DataFrame")

    # Use index as internal ID
    df = df.reset_index(drop=True)
    indices = df.index.tolist()

    # Decision Variables
    x = pl.LpVariable.dicts("select", indices, lowBound=0, upBound=1, cat="Binary")   # in squad
    y = pl.LpVariable.dicts("start", indices, lowBound=0, upBound=1, cat="Binary")    # starting XI
    c = pl.LpVariable.dicts("captain", indices, lowBound=0, upBound=1, cat="Binary")  # captain

    # Define the problem
    model = pl.LpProblem("FPL_Team_Selection", pl.LpMaximize)

    # Objective function: maximize points of starters + captain (double)
    model += pl.lpSum(
        df.loc[i, objective_col] * (y[i] + c[i]) for i in indices
    ), "Total_Projected_Points"

    # Budget constraint
    model += pl.lpSum(df.loc[i, "price"] * x[i] for i in indices) <= budget, "Budget"

    # Exactly 15 players in full squad
    model += pl.lpSum(x[i] for i in indices) == 15, "Squad_Size"

    # Positional constraints in squad
    for pos, required in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
        model += (
            pl.lpSum(x[i] for i in indices if df.loc[i, "position"] == pos) == required,
            f"{pos}_squad_count",
        )

    # Only 11 players start
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

    # A player can only start if in squad
    for i in indices:
        model += y[i] <= x[i], f"start_only_if_selected_{i}"

    # Exactly 1 captain
    model += pl.lpSum(c[i] for i in indices) == 1, "Captain_Count"

    # Captain must be a starter
    for i in indices:
        model += c[i] <= y[i], f"captain_must_start_{i}"

    # Max 3 players per real-life team
    for team in df["team_name"].unique():
        model += (
            pl.lpSum(x[i] for i in indices if df.loc[i, "team_name"] == team) <= 3,
            f"Max_3_from_{team}",
        )

    # === ONE-TRANSFER CONSTRAINT (if previous squad is provided) ===
    if prev_squad_ids is not None and len(prev_squad_ids) > 0 and transfers_per_gw > 0:
        # map prev_squad_ids to current indices (players may be filtered out)
        prev_indices = [i for i in indices if df.loc[i, "player_id"] in prev_squad_ids]
        if prev_indices:
            kept_required = max(len(prev_indices) - transfers_per_gw, 0)
            model += (
                pl.lpSum(x[i] for i in prev_indices) == kept_required,
                "Transfer_Limit_From_Previous_Squad",
            )

    # Solve
    model.solve(pl.PULP_CBC_CMD(msg=False))

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

    # Sort nicely for display
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


# -------------------------------------------------------------------
# SEASON LOOP (NO ML, CONSTANT OBJECTIVE)
# -------------------------------------------------------------------
def optimize_season_no_ml(num_gws: int = 9):
    """
    Pure LP season optimizer with one transfer per GW and a fixed objective
    (e.g. points_per_game). No ML / predictions.

    For GW1: build best squad from scratch.
    For GW2..N: same objective, but exactly 1 player changed each GW.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    players = load_players()

    prev_squad_ids: Optional[List[int]] = None
    all_squads = {}

    for gw in range(1, num_gws + 1):
        print(f"\n=== Optimizing GW{gw} (no ML, objective = points_per_game) ===")

        if gw == 1:
            transfers = 0  # first squad, no previous team
        else:
            transfers = 1  # exactly 1 transfer per GW

        best_team, model = build_team_optimizer(
            players,
            budget=100.0,
            objective_col="points_per_game",
            min_minutes_season=270,
            min_minutes_share_5=0.40,
            min_games_played_5=2.0,
            only_available=True,
            prev_squad_ids=prev_squad_ids,
            transfers_per_gw=transfers,
        )

        print("Status:", pl.LpStatus[model.status])
        print("Total projected points:", pl.value(model.objective))

        # Save full squad and XI
        gw_tag = f"GW{gw:02d}"
        out_path = RESULTS_DIR / f"best_team_lp_{gw_tag}.csv"
        best_team.to_csv(out_path, index=False)

        xi = best_team[best_team["is_starter"]].copy()
        xi_path = RESULTS_DIR / f"starting_xi_lp_{gw_tag}.csv"
        xi.to_csv(xi_path, index=False)

        print(f"Saved squad to: {out_path}")
        print(f"Saved starting XI to: {xi_path}")

        prev_squad_ids = best_team["player_id"].tolist()
        all_squads[gw_tag] = best_team

    return all_squads


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    optimize_season_no_ml(num_gws=9)


if __name__ == "__main__":
    main()