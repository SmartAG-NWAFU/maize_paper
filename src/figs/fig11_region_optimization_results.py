from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_INPUT = ROOT / "outputs" / "management_yield_region_optimized" / "management_region_baseline_vs_optimal.csv"
DEFAULT_OUTPUT = ROOT / "fig" / "fig11_region_optimization_results.png"
DEFAULT_SUMMARY_OUTPUT = ROOT / "fig" / "fig11_region_optimization_summary.csv"

SCENARIO_LABELS = {
    "Baseline median": "Baseline",
    "Optimal yield": "Max-yield strategy",
}

SCENARIO_COLORS = {
    "Baseline": "#A5B1BD",
    "Max-yield strategy": "#3E6B89",
}

REGION_ORDER = ["Hebei", "Shandong"]

BASE_FONT_SIZE = 17
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
X_TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 13
PANEL_FONT_SIZE = 19

BAR_SPECS = [
    {
        "column": "yield_value",
        "label": "Grain yield (kg ha$^{-1}$)",
        "summary_prefix": "yield",
        "tag": "(a)",
    },
    {
        "column": "profit_value",
        "label": "Net profit (CNY ha$^{-1}$)",
        "summary_prefix": "profit",
        "tag": "(b)",
    },
    {
        "column": "input_cost",
        "label": "Production input cost (CNY ha$^{-1}$)",
        "summary_prefix": "input_cost",
        "tag": "(c)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regional optimization results for baseline, optimal yield, and irrigation-mode candidates."
    )
    parser.add_argument(
        "--baseline-input",
        type=Path,
        default=DEFAULT_BASELINE_INPUT,
        help="CSV file containing region-level baseline and optimal results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Output CSV path for region-level optimization summary.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.titlesize": LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.linewidth": 1.1,
            "savefig.bbox": "tight",
        }
    )


def load_baseline_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path).copy()
    df = df.loc[df["result_type"].isin(SCENARIO_LABELS)].copy()
    numeric_columns = ["yield_value", "profit_value", "input_cost", "objective_value", "distance_penalty"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["scenario"] = df["result_type"].map(SCENARIO_LABELS)
    df["region_name"] = df["region_name"].astype(str)
    df["region_order"] = pd.Categorical(df["region_name"], categories=REGION_ORDER, ordered=True)
    df = df.dropna(subset=["region_name", "scenario", "yield_value", "profit_value", "input_cost"]).copy()
    return df.sort_values(["region_order", "scenario"]).reset_index(drop=True)


def build_summary_table(baseline_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({"region": REGION_ORDER})
    baseline_pivot = baseline_df.pivot(index="region_name", columns="scenario")

    for spec in BAR_SPECS:
        column = spec["column"]
        prefix = spec["summary_prefix"]
        baseline_values = baseline_pivot[column]["Baseline"].reindex(REGION_ORDER)
        optimal_values = baseline_pivot[column]["Max-yield strategy"].reindex(REGION_ORDER)
        diff_values = optimal_values - baseline_values
        pct_values = np.where(baseline_values != 0, diff_values / baseline_values * 100.0, np.nan)
        summary[f"{prefix}_baseline"] = baseline_values.to_numpy(dtype=float)
        summary[f"{prefix}_optimal"] = optimal_values.to_numpy(dtype=float)
        summary[f"{prefix}_change"] = diff_values.to_numpy(dtype=float)
        summary[f"{prefix}_change_pct"] = pct_values

    return summary.round(2)


def add_panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        0.02,
        0.97,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle=(0, (2, 2)), alpha=0.28, color="#6C757D")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", length=4.5, width=1.0, pad=6)


def format_bar_value(value: float) -> str:
    return f"{value:.0f}"


def plot_grouped_bars(ax: plt.Axes, df: pd.DataFrame, column: str, y_label: str, tag: str) -> None:
    labels = REGION_ORDER
    x = np.arange(len(labels))
    width = 0.28

    baseline_values = (
        df.loc[df["scenario"] == "Baseline", ["region_name", column]]
        .set_index("region_name")[column]
        .reindex(labels)
        .to_numpy(dtype=float)
    )
    optimal_values = (
        df.loc[df["scenario"] == "Max-yield strategy", ["region_name", column]]
        .set_index("region_name")[column]
        .reindex(labels)
        .to_numpy(dtype=float)
    )

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_values,
        width=width,
        color=SCENARIO_COLORS["Baseline"],
        edgecolor="black",
        linewidth=0.8,
        label="Baseline",
    )
    bars_optimal = ax.bar(
        x + width / 2,
        optimal_values,
        width=width,
        color=SCENARIO_COLORS["Max-yield strategy"],
        edgecolor="black",
        linewidth=0.8,
        label="Max-yield strategy",
    )

    y_max = float(np.nanmax(np.concatenate([baseline_values, optimal_values])))
    offset = max(1.0, y_max * 0.02)
    for bars in (bars_baseline, bars_optimal):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                format_bar_value(value),
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#2F2F2F",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=X_TICK_FONT_SIZE)
    ax.set_ylabel(y_label, labelpad=8)
    ax.set_ylim(0, y_max * 1.32 if y_max > 0 else 1.0)
    style_axis(ax)
    add_panel_tag(ax, tag)


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    baseline_df = load_baseline_data(args.baseline_input)
    summary = build_summary_table(baseline_df)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8), constrained_layout=False)
    for ax, spec in zip(np.atleast_1d(axes), BAR_SPECS):
        plot_grouped_bars(ax, baseline_df, spec["column"], spec["label"], spec["tag"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
        columnspacing=1.8,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.80, wspace=0.48)
    fig.savefig(args.output)
    plt.close(fig)

    print(f"Saved figure to: {args.output}")
    print(f"Saved summary to: {args.summary_output}")


if __name__ == "__main__":
    main()
