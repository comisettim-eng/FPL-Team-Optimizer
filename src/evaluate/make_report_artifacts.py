"""

Reads existing outputs created by:
- src/models/train_compare_models.py  -> results/ml_backtests/model_comparison_*.csv
- src/evaluate/compare_models_season_scores.py -> results/ml_backtests/model_season_scores_api_gw*_*.csv

Outputs:
- results/tables/
  - table_1_model_performance.csv / .md
  - table_2_generalization_gap.csv / .md
  - table_3_season_performance.csv / .md
  - final_results_table.csv / .md   (single “final” table to paste in report)

- results/figures/
  - fig_1_validation_error.png
  - fig_2_train_vs_valid_error.png
  - fig_3_cumulative_points.png
  - fig_4_weekly_points.png

Also prints ONE final results table at the end 
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_DIR = PROJECT_ROOT / "results" / "ml_backtests"

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
def _latest_file(pattern: str) -> Path:
    """
    Pick the most recently modified file matching a glob pattern.
    Example: pattern="model_comparison_*.csv"
    """
    matches = sorted(BACKTEST_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files found in {BACKTEST_DIR} matching: {pattern}")
    return matches[0]


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Small, dependency-free markdown table."""
    out = df.copy()
    # format floats nicely
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]) or pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    header = "| " + " | ".join(out.columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |\n"
    lines = [header + sep]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in out.columns) + " |")
    return "\n".join(lines) + "\n"


def _save_table(df: pd.DataFrame, stem: str) -> None:
    csv_path = TABLES_DIR / f"{stem}.csv"
    md_path = TABLES_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown_table(df), encoding="utf-8")


def _infer_gw_range_from_filename(path: Path) -> str:
    """
    Try to parse something like "..._gw2_12.csv" -> "GW2–GW12"
    """
    m = re.search(r"_gw(\d+)_(\d+)", path.name)
    if not m:
        return "GW range (unknown)"
    return f"GW{int(m.group(1))}–GW{int(m.group(2))}"


# -------------------------
# Load sources
# -------------------------
def load_model_comparison() -> pd.DataFrame:
    # produced by train_compare_models.py
    p = _latest_file("model_comparison_*.csv")
    df = pd.read_csv(p)

    # normalize column names if needed
    df.columns = [c.strip() for c in df.columns]
    if "model" not in df.columns:
        raise ValueError(f"'model' column not found in {p}")

    return df, p


def load_season_scores() -> pd.DataFrame:
    # produced by compare_models_season_scores.py
    p = _latest_file("model_season_scores_api_gw*_*.csv")
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if "gameweek" not in df.columns:
        raise ValueError(f"'gameweek' column not found in {p}")
    return df, p


# -------------------------
# Build Tables
# -------------------------
def build_table_1(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: Model performance comparison (validation & decision quality)
    Expected columns in model_df (depending on your implementation):
      - valid_mae, valid_rmse, valid_r2
      - spearman_rho (optional)
      - top11_overlap_precision, top15_overlap_precision (optional)
    """
    cols = ["model"]

    # core regression
    for c in ["valid_mae", "valid_rmse", "valid_r2"]:
        if c in model_df.columns:
            cols.append(c)

    # ranking/decision if present
    for c in ["spearman_rho", "top11_overlap_precision", "top15_overlap_precision"]:
        if c in model_df.columns:
            cols.append(c)

    t1 = model_df[cols].copy()
    # sort by valid_mae if exists
    if "valid_mae" in t1.columns:
        t1 = t1.sort_values("valid_mae", ascending=True).reset_index(drop=True)
    return t1


def build_table_2(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: Generalization gap (overfitting analysis)
    Uses:
      - train_mae, valid_mae, mae_gap
      - train_r2, valid_r2
    """
    needed = ["model"]
    for c in ["train_mae", "valid_mae", "mae_gap", "train_r2", "valid_r2"]:
        if c in model_df.columns:
            needed.append(c)

    t2 = model_df[needed].copy()

    # if mae_gap doesn't exist but train/valid exist, compute it
    if "mae_gap" not in t2.columns and {"train_mae", "valid_mae"}.issubset(t2.columns):
        t2["mae_gap"] = t2["train_mae"] - t2["valid_mae"]

    if "valid_mae" in t2.columns:
        t2 = t2.sort_values("valid_mae", ascending=True).reset_index(drop=True)
    return t2


def build_table_3(season_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: Season-level FPL performance:
      - Total points per model
      - Avg points per GW
      - Delta vs FPL average (if present)
    Assumes columns like: rf_score, lgbm_score, xgb_score, fpl_average_score
    """
    df = season_scores_df.copy()
    gw_count = df["gameweek"].nunique()

    score_cols = [c for c in df.columns if c.endswith("_score") and c != "fpl_average_score"]
    if not score_cols:
        raise ValueError("No model score columns found (expected '*_score').")

    rows = []
    avg_total = df["fpl_average_score"].sum(skipna=True) if "fpl_average_score" in df.columns else np.nan
    avg_mean = df["fpl_average_score"].mean(skipna=True) if "fpl_average_score" in df.columns else np.nan

    for c in score_cols:
        model_key = c.replace("_score", "")
        total = float(df[c].sum(skipna=True))
        mean = float(df[c].mean(skipna=True))
        delta_total = float(total - avg_total) if np.isfinite(avg_total) else np.nan
        rows.append(
            {
                "model": model_key,
                "total_points": total,
                "avg_points_per_gw": mean,
                "delta_total_vs_fpl_avg": delta_total,
                "num_gameweeks": int(gw_count),
            }
        )

    t3 = pd.DataFrame(rows).sort_values("total_points", ascending=False).reset_index(drop=True)

    # add the average line as reference (not a “model”)
    if np.isfinite(avg_total):
        t3 = pd.concat(
            [
                t3,
                pd.DataFrame(
                    [{
                        "model": "fpl_average",
                        "total_points": float(avg_total),
                        "avg_points_per_gw": float(avg_mean),
                        "delta_total_vs_fpl_avg": 0.0,
                        "num_gameweeks": int(gw_count),
                    }]
                ),
            ],
            ignore_index=True,
        )
    return t3


def build_final_results_table(table_1: pd.DataFrame, table_3: pd.DataFrame) -> pd.DataFrame:
    """
    One final table (printed ONCE at end of project), combining:
    - validation metrics (from Table 1)
    - season totals (from Table 3)
    """
    t1 = table_1.copy()
    t3 = table_3.copy()

    # Map season table model keys to full names if needed
    # If your Table 1 uses full names and Table 3 uses rf/lgbm/xgb,
    # we keep both by merging on a normalized key.
    def norm_model(s: str) -> str:
        s = str(s).lower()
        if "random" in s or s in ["rf"]:
            return "rf"
        if "light" in s or s in ["lgbm", "lgb"]:
            return "lgbm"
        if "xgb" in s or "xgboost" in s:
            return "xgb"
        if s == "fpl_average":
            return "fpl_average"
        return s

    t1["model_key"] = t1["model"].map(norm_model)
    t3["model_key"] = t3["model"].map(norm_model)

    merged = t1.merge(
        t3[["model_key", "total_points", "avg_points_per_gw", "delta_total_vs_fpl_avg"]],
        on="model_key",
        how="left",
    )

    # keep a clean final set of columns
    cols = ["model"]
    for c in ["valid_mae", "valid_rmse", "valid_r2", "spearman_rho", "top11_overlap_precision", "top15_overlap_precision"]:
        if c in merged.columns:
            cols.append(c)
    cols += ["total_points", "avg_points_per_gw", "delta_total_vs_fpl_avg"]

    final_df = merged[cols].copy()

    # sort by total_points if present, else by valid_mae
    if "total_points" in final_df.columns and final_df["total_points"].notna().any():
        final_df = final_df.sort_values("total_points", ascending=False).reset_index(drop=True)
    elif "valid_mae" in final_df.columns:
        final_df = final_df.sort_values("valid_mae", ascending=True).reset_index(drop=True)

    return final_df


# -------------------------
# Figures
# -------------------------
def fig_1_validation_error(model_df: pd.DataFrame) -> None:
    # bar chart for valid MAE (fallback to RMSE if missing)
    df = model_df.copy()
    metric = "valid_mae" if "valid_mae" in df.columns else ("valid_rmse" if "valid_rmse" in df.columns else None)
    if metric is None:
        print("[FIG1] Skipped (no valid_mae/valid_rmse in model_comparison).")
        return

    df = df.sort_values(metric, ascending=True)

    plt.figure()
    plt.bar(df["model"].astype(str), df[metric].astype(float))
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(f"Figure 1: Validation error by model ({metric})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_1_validation_error.png", dpi=200)
    plt.close()


def fig_2_train_vs_valid_error(model_df: pd.DataFrame) -> None:
    if not {"train_mae", "valid_mae"}.issubset(model_df.columns):
        print("[FIG2] Skipped (need train_mae and valid_mae).")
        return

    df = model_df.copy().sort_values("valid_mae", ascending=True)
    x = np.arange(len(df))
    width = 0.4

    plt.figure()
    plt.bar(x - width / 2, df["train_mae"].astype(float), width, label="Train MAE")
    plt.bar(x + width / 2, df["valid_mae"].astype(float), width, label="Valid MAE")
    plt.ylabel("MAE")
    plt.xlabel("Model")
    plt.title("Figure 2: Train vs Validation MAE (generalization gap)")
    plt.xticks(x, df["model"].astype(str), rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_2_train_vs_valid_error.png", dpi=200)
    plt.close()


def fig_3_cumulative_points(season_scores_df: pd.DataFrame, gw_label: str) -> None:
    df = season_scores_df.copy().sort_values("gameweek")
    score_cols = [c for c in df.columns if c.endswith("_score")]

    plt.figure()
    for c in score_cols:
        plt.plot(df["gameweek"], df[c].fillna(0).cumsum(), label=c.replace("_score", ""))
    plt.xlabel("Gameweek")
    plt.ylabel("Cumulative points")
    plt.title(f"Figure 3: Cumulative season points ({gw_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_3_cumulative_points.png", dpi=200)
    plt.close()


def fig_4_weekly_points(season_scores_df: pd.DataFrame, gw_label: str) -> None:
    df = season_scores_df.copy().sort_values("gameweek")
    score_cols = [c for c in df.columns if c.endswith("_score")]

    plt.figure()
    for c in score_cols:
        plt.plot(df["gameweek"], df[c].fillna(0), label=c.replace("_score", ""))
    plt.xlabel("Gameweek")
    plt.ylabel("Weekly points (starting XI + captain)")
    plt.title(f"Figure 4: Weekly team points over time ({gw_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_4_weekly_points.png", dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def main() -> None:
    print("\n[Reporting] Building report artifacts from saved CSV outputs...")

    model_df, model_path = load_model_comparison()
    season_df, season_path = load_season_scores()

    gw_label = _infer_gw_range_from_filename(season_path)

    # Tables
    t1 = build_table_1(model_df)
    t2 = build_table_2(model_df)
    t3 = build_table_3(season_df)

    _save_table(t1, "table_1_model_performance")
    _save_table(t2, "table_2_generalization_gap")
    _save_table(t3, "table_3_season_performance")

    final_tbl = build_final_results_table(t1, t3)
    _save_table(final_tbl, "final_results_table")

    # Figures
    fig_1_validation_error(model_df)
    fig_2_train_vs_valid_error(model_df)
    fig_3_cumulative_points(season_df, gw_label)
    fig_4_weekly_points(season_df, gw_label)

    # Print exactly ONE final table (grader-friendly)
    print("\n==============================")
    print("FINAL RESULTS TABLE (print once)")
    print("==============================")
    print(final_tbl.to_string(index=False))

    print("\n[Reporting] Saved tables to:", TABLES_DIR)
    print("[Reporting] Saved figures to:", FIGURES_DIR)


if __name__ == "__main__":
    main()