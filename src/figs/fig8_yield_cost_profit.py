from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "model_vars.csv"
DEFAULT_COST_INPUT = ROOT / "data" / "cost_required_ha.csv"
DEFAULT_OUTPUT = ROOT / "fig" / "fig8_region_yield_cost_profit_boxplots.png"
DEFAULT_SUMMARY_OUTPUT = ROOT / "fig" / "fig8_region_yield_cost_profit_summary.csv"
DEFAULT_GRAIN_PRICE = 2.4
MU_TO_HA = 15.0

REGION_LABELS = {
    0: "Hebei",
    1: "Shandong",
}

REGION_COLORS = {
    "Hebei": "#9DC3E6",
    "Shandong": "#F4A582",
}

BASE_FONT_SIZE = 19
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 19
BOX_X_TICK_FONT_SIZE = 25
ANNOTATION_FONT_SIZE = 19
PANEL_FONT_SIZE = 22
SCATTER_LEGEND_FONT_SIZE = 14

VARIABLE_SPECS = [
    {
        "column": "yield_ha",
        "label": "Yield (kg/ha)",
        "summary_prefix": "yield",
        "tag": "(a)",
    },
    {
        "column": "cost_ha",
        "label": "Cost (CNY/ha)",
        "summary_prefix": "cost",
        "tag": "(b)",
    },
    {
        "column": "profit_ha",
        "label": "Profit (CNY/ha)",
        "summary_prefix": "profit",
        "tag": "(c)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot yield, cost, and profit boxplots by region."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cost-input", type=Path, default=DEFAULT_COST_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--grain-price", type=float, default=DEFAULT_GRAIN_PRICE)
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
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.linewidth": 1.1,
            "savefig.bbox": "tight",
        }
    )


def load_data(input_path: Path, cost_input_path: Path, grain_price: float) -> pd.DataFrame:
    model_df = pd.read_csv(input_path, usecols=["uuid", "Region", "Yield"]).copy()
    cost_df = pd.read_csv(cost_input_path, usecols=["uuid", "total_cost_yuan_per_ha"], encoding="utf-8-sig").copy()

    df = model_df.merge(cost_df, on="uuid", how="inner")
    df = df.rename(columns={"Region": "region_code", "Yield": "yield_ha", "total_cost_yuan_per_ha": "cost_ha"})
    df["region"] = df["region_code"].map(REGION_LABELS)
    df["yield_ha"] = pd.to_numeric(df["yield_ha"], errors="coerce")
    df["cost_ha"] = pd.to_numeric(df["cost_ha"], errors="coerce")
    df["profit_ha"] = df["yield_ha"] * grain_price - df["cost_ha"]
    return df.dropna(subset=["region", "yield_ha", "cost_ha", "profit_ha"]).copy()


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    agg_columns = [spec["column"] for spec in VARIABLE_SPECS]
    summary = (
        df.groupby("region", as_index=False)[agg_columns]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )
    renamed_columns = ["region"]
    for spec in VARIABLE_SPECS:
        prefix = spec["summary_prefix"]
        renamed_columns.extend(
            [f"{prefix}_count", f"{prefix}_mean", f"{prefix}_median", f"{prefix}_min", f"{prefix}_max"]
        )
    summary.columns = renamed_columns
    return summary.reset_index(drop=True)


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


def plot_boxplot(ax: plt.Axes, df: pd.DataFrame, column: str, y_label: str, tag: str) -> None:
    regions = ["Hebei", "Shandong"]
    positions = np.arange(len(regions))
    values = [df.loc[df["region"] == region, column].to_numpy(dtype=float) for region in regions]

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 1.8},
        whiskerprops={"color": "#4F4F4F", "linewidth": 1.3},
        capprops={"color": "#4F4F4F", "linewidth": 1.3},
        boxprops={"edgecolor": "black", "linewidth": 1.1},
    )
    for patch, region in zip(bp["boxes"], regions):
        patch.set_facecolor(REGION_COLORS[region])
        patch.set_alpha(0.6)

    rng = np.random.default_rng(66)
    for pos, region, arr in zip(positions, regions, values):
        jitter = rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(
            np.full(len(arr), pos) + jitter,
            arr,
            s=34,
            color=REGION_COLORS[region],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        mean_value = float(np.mean(arr))
        ax.text(
            0.02,
            0.84 - pos * 0.07,
            f"{region}: {mean_value:.0f}",
            fontsize=ANNOTATION_FONT_SIZE,
            color="#1f1f1f",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
            transform=ax.transAxes,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(regions, fontsize=BOX_X_TICK_FONT_SIZE)
    ax.set_ylabel(y_label)
    col_min = min(float(arr.min()) for arr in values)
    col_max = max(float(arr.max()) for arr in values)
    span = col_max - col_min
    pad = span * 0.16 if span > 0 else max(abs(col_max) * 0.16, 1.0)
    ax.set_ylim(col_min - pad, col_max + pad)
    style_axis(ax)
    add_panel_tag(ax, tag)


def plot_yield_profit_scatter(ax: plt.Axes, df: pd.DataFrame) -> None:
    scatter_df = df[["region", "cost_ha", "yield_ha"]].copy()
    x_col = "cost_ha"
    y_col = "yield_ha"

    x_med = float(scatter_df[x_col].median())
    y_med = float(scatter_df[y_col].median())
    x_low, x_up = x_med * 0.9, x_med * 1.1
    y_low, y_up = y_med * 0.9, y_med * 1.1

    def assign_zone(row: pd.Series) -> str:
        x_val = row[x_col]
        y_val = row[y_col]
        if x_low <= x_val <= x_up and y_low <= y_val <= y_up:
            return "M"
        if x_val >= x_med and y_val >= y_med:
            return "H-H"
        if x_val < x_med and y_val >= y_med:
            return "L-H"
        if x_val < x_med and y_val < y_med:
            return "L-L"
        return "H-L"

    scatter_df["zone"] = scatter_df.apply(assign_zone, axis=1)

    zone_markers = {
        "H-H": "^",
        "L-H": "o",
        "M": "s",
        "L-L": "v",
        "H-L": "X",
    }
    zone_order = ["H-H", "L-H", "M", "L-L", "H-L"]

    for zone in zone_order:
        zone_df = scatter_df[scatter_df["zone"] == zone]
        if zone_df.empty:
            continue
        for region in REGION_LABELS.values():
            region_df = zone_df[zone_df["region"] == region]
            if region_df.empty:
                continue
            ax.scatter(
                region_df[x_col],
                region_df[y_col],
                s=92,
                marker=zone_markers[zone],
                color=REGION_COLORS[region],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.45,
                zorder=3,
            )

    x_min = float(scatter_df[x_col].min())
    x_max = float(scatter_df[x_col].max())
    y_min = float(scatter_df[y_col].min())
    y_max = float(scatter_df[y_col].max())
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = x_span * 0.12 if x_span > 0 else max(abs(x_max) * 0.12, 1.0)
    y_pad = y_span * 0.18 if y_span > 0 else max(abs(y_max) * 0.18, 1.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    current_x_min, current_x_max = ax.get_xlim()
    current_y_min, current_y_max = ax.get_ylim()
    line_style = {"colors": "black", "linestyles": "--", "lw": 1.0, "zorder": 2}
    ax.vlines([x_low, x_up], y_low, y_up, **line_style)
    ax.hlines([y_low, y_up], x_low, x_up, **line_style)
    ax.vlines(x_med, current_y_min, y_low, **line_style)
    ax.vlines(x_med, y_up, current_y_max, **line_style)
    ax.hlines(y_med, current_x_min, x_low, **line_style)
    ax.hlines(y_med, x_up, current_x_max, **line_style)

    style_axis(ax)
    add_panel_tag(ax, "(d)")
    ax.set_xlabel("Cost (CNY/ha)")
    ax.set_ylabel("Yield (kg/ha)")

    region_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=REGION_COLORS[region],
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=12,
            label=region,
        )
        for region in REGION_LABELS.values()
    ]
    zone_handles = [
        Line2D(
            [0],
            [0],
            marker=zone_markers[zone],
            linestyle="None",
            color="#6789CB",
            # markerfacecolor="#6B7280",
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=10,
            label=zone,
        )
        for zone in zone_order
        if zone in scatter_df["zone"].values
    ]

    first_legend = ax.legend(
        handles=region_handles,
        loc=(0.64, 0.8),
        frameon=False,
        fontsize=SCATTER_LEGEND_FONT_SIZE + 7,
        handletextpad=0.2,
        labelspacing=0.2,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=zone_handles,
        loc=(-0.02, 0.52),
        frameon=False,
        fontsize=SCATTER_LEGEND_FONT_SIZE + 7,
        handletextpad=0.2,
        labelspacing=0.2,
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input, args.cost_input, args.grain_price)
    summary = build_summary_table(df)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.2), constrained_layout=True)
    axes_flat = axes.flatten()
    for ax, spec in zip(axes_flat, VARIABLE_SPECS):
        plot_boxplot(ax, df, spec["column"], spec["label"], spec["tag"])
    plot_yield_profit_scatter(axes_flat[-1], df)

    fig.savefig(args.output)
    plt.close(fig)

    print(f"Saved figure to: {args.output}")
    print(f"Saved summary to: {args.summary_output}")


if __name__ == "__main__":
    main()
