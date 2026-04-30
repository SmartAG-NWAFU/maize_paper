from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "model_vars.csv"
DEFAULT_OUTPUT = ROOT / "fig" / "fig6_region_irrigation_boxplots.png"
DEFAULT_SUMMARY_OUTPUT = ROOT / "fig" / "fig6_region_irrigation_summary.csv"

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
LEGEND_FONT_SIZE = 19

VARIABLE_SPECS = [
    {
        "column": "Irr_Count",
        "label": "Irrigation frequency",
        "summary_prefix": "irr_count",
        "tag": "(a)",
    },
    {
        "column": "Irr_Elec",
        "label": "Irrigation electricity usage (kWh ha$^{-1}$)",
        "summary_prefix": "irr_elec",
        "tag": "(b)",
    },
]

FACILITY_SPECS = [
    ("Irr_None", "Rainfed"),
    ("Irr_Flood", "Flood"),
    ("Irr_Sprinkler", "Sprinkler"),
    ("Irr_Drip", "Drip"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot irrigation frequency and electricity boxplots by region."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file containing regional irrigation observations.",
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
        help="Output CSV path for grouped summary statistics.",
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


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        input_path,
        usecols=["Region", "Irr_Count", "Irr_Elec", "Irr_None", "Irr_Flood", "Irr_Sprinkler", "Irr_Drip"],
    ).copy()
    df = df.rename(columns={"Region": "region_code"})
    df["region"] = df["region_code"].map(REGION_LABELS)
    for column in ["Irr_Count", "Irr_Elec", "Irr_None", "Irr_Flood", "Irr_Sprinkler", "Irr_Drip"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(
        subset=["region", "Irr_Count", "Irr_Elec", "Irr_None", "Irr_Flood", "Irr_Sprinkler", "Irr_Drip"]
    ).copy()


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    agg_columns = [spec["column"] for spec in VARIABLE_SPECS]
    summary = (
        df.groupby("region", as_index=False)[agg_columns]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )
    summary.columns = [
        "region",
        "irr_count_count",
        "irr_count_mean",
        "irr_count_median",
        "irr_count_min",
        "irr_count_max",
        "irr_elec_count",
        "irr_elec_mean",
        "irr_elec_median",
        "irr_elec_min",
        "irr_elec_max",
    ]
    return summary.reset_index(drop=True)


def build_facility_summary(df: pd.DataFrame) -> pd.DataFrame:
    facility_columns = [column for column, _ in FACILITY_SPECS]
    summary = (
        df.groupby("region", as_index=False)[facility_columns]
        .sum()
        .round(0)
    )
    summary = summary.rename(columns=dict(FACILITY_SPECS))
    return summary


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
        display_value = f"{mean_value:.1f}" if column == "Irr_Count" else f"{mean_value:.0f}"
        ax.text(
            0.02,
            0.84 - pos * 0.07,
            f"{region}: {display_value}",
            fontsize=ANNOTATION_FONT_SIZE,
            color="#1f1f1f",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
            transform=ax.transAxes,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(regions, fontsize=BOX_X_TICK_FONT_SIZE)
    ax.set_ylabel(y_label)
    style_axis(ax)
    add_panel_tag(ax, tag)


def plot_facility_bar(ax: plt.Axes, facility_summary: pd.DataFrame, tag: str) -> None:
    facility_labels = [label for _, label in FACILITY_SPECS]
    regions = ["Hebei", "Shandong"]
    x = np.arange(len(facility_labels))
    width = 0.34

    ordered = facility_summary.set_index("region").reindex(regions)
    max_value = 0.0
    for idx, region in enumerate(regions):
        values = ordered.loc[region, facility_labels].to_numpy(dtype=float)
        max_value = max(max_value, float(values.max(initial=0.0)))
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            color=REGION_COLORS[region],
            alpha=0.88,
            edgecolor="black",
            linewidth=0.8,
            label=region,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(facility_labels, fontsize=TICK_FONT_SIZE+6)
    ax.set_ylabel("Count")
    ax.set_ylim(0, max_value * 1.28 if max_value > 0 else 1.0)
    style_axis(ax)
    ax.legend(frameon=False, loc=(0.01, 0.68),fontsize=22)
    add_panel_tag(ax, tag)


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    summary = build_summary_table(df)
    facility_summary = build_facility_summary(df)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    facility_summary.to_csv(
        args.summary_output.with_name(f"{args.summary_output.stem}_facility.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=(13.3, 10.2), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    facility_ax = fig.add_subplot(gs[1, :])

    for ax, spec in zip(axes, VARIABLE_SPECS):
        plot_boxplot(ax, df, spec["column"], spec["label"], spec["tag"])
    plot_facility_bar(facility_ax, facility_summary, "(c)")

    fig.savefig(args.output)
    plt.close(fig)

    print(f"Saved figure to: {args.output}")
    print(f"Saved summary to: {args.summary_output}")


if __name__ == "__main__":
    main()
