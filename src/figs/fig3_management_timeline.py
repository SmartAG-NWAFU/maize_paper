from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "management_time.csv"
DEFAULT_OUTPUT = ROOT / "fig" / "fig3_region_management_timeline.png"
DEFAULT_SUMMARY_OUTPUT = ROOT / "fig" / "fig3_region_management_timeline_summary.csv"

REGION_LABELS = {
    "hebei": "Hebei",
    "shandong": "Shandong",
}

REGION_ORDER = ["Hebei", "Shandong"]

REGION_COLORS = {
    "Hebei": "#9DC3E6",
    "Shandong": "#F4A582",
}

EVENT_LABELS = {
    "播种": "Sowing",
    "施肥": "Fertilization",
    "灌溉": "Irrigation",
    "收获": "Harvest",
    "杀菌剂": "Fungicide",
    "杀虫剂": "Insecticide",
    "除草剂": "Herbicide",
    "生长调节剂": "Growth regulator",
}

EVENT_ORDER = [
    "Sowing",
    "Fertilization",
    "Herbicide",
    "Insecticide",
    "Irrigation",
    "Fungicide",
    "Growth regulator",
    "Harvest",
]

BASE_FONT_SIZE = 12
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
PANEL_FONT_SIZE = 12
ROW_SPACING = 1.38
REGION_OFFSETS = {
    "Hebei": -0.18,
    "Shandong": 0.18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot management timeline by region from management_time.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file containing management dates and action types.",
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
        help="Output CSV path for aggregated management timeline records.",
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
        usecols=["uuid", "region", "实施日期", "措施类型", "措施细类"],
    ).copy()
    df["region_name"] = df["region"].astype(str).str.strip().str.lower().map(REGION_LABELS)
    df["event_raw"] = df["措施类型"].astype(str).str.strip()
    pesticide_mask = df["event_raw"].eq("打药")
    df.loc[pesticide_mask, "event_raw"] = df.loc[pesticide_mask, "措施细类"].astype(str).str.strip()
    df["event_label"] = df["event_raw"].map(EVENT_LABELS)
    df["event_date"] = pd.to_datetime(df["实施日期"], errors="coerce")
    df = df.dropna(subset=["region_name", "event_label", "event_date"]).copy()
    return df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = df[["region_name", "event_label", "event_date"]].copy()
    event_category = pd.Categorical(summary["event_label"], categories=EVENT_ORDER, ordered=True)
    summary = summary.assign(event_order=event_category)
    summary = summary.sort_values(["event_order", "event_date", "region_name"]).drop(columns=["event_order"])
    return summary.reset_index(drop=True)


def add_panel_tag(ax: plt.Axes, tag: str) -> None:
    if not tag:
        return
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
    spine_color = "#B8BFC7"
    ax.grid(True, axis="x", linestyle=(0, (2, 2)), alpha=0.22, color="#AEB6BF")
    ax.grid(True, axis="y", linestyle=(0, (2, 2)), alpha=0.28, color="#6C757D")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color(spine_color)
    ax.tick_params(axis="both", length=4.5, width=1.0, pad=6, color=spine_color, labelcolor="#1F1F1F")


def plot_combined_timeline(ax: plt.Axes, summary_df: pd.DataFrame, tag: str) -> None:
    y_positions = {event: idx * ROW_SPACING for idx, event in enumerate(EVENT_ORDER)}
    rng = np.random.default_rng(66)
    for region_name in REGION_ORDER:
        region_df = summary_df.loc[summary_df["region_name"] == region_name].copy()
        y_base = region_df["event_label"].map(y_positions).to_numpy(dtype=float)
        y_values = y_base + REGION_OFFSETS[region_name] + rng.uniform(-0.05, 0.05, size=len(region_df))
        ax.scatter(
            region_df["event_date"],
            y_values,
            s=30,
            color=REGION_COLORS[region_name],
            alpha=0.92,
            linewidths=0.0,
            zorder=3,
        )

    ax.set_yticks([y_positions[event] for event in EVENT_ORDER])
    ax.set_yticklabels(EVENT_ORDER)
    ax.set_ylim(-0.45, max(y_positions.values()) + 0.45)
    ax.invert_yaxis()
    ax.set_ylabel("Management type")
    style_axis(ax)
    add_panel_tag(ax, tag)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=REGION_COLORS[region],
            markeredgecolor="none",
            markersize=7,
            label=region,
        )
        for region in REGION_ORDER
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=LEGEND_FONT_SIZE)


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    summary = build_summary_table(df)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    year_value = int(summary["event_date"].dt.year.mode().iloc[0])
    min_date = pd.Timestamp(year=year_value, month=5, day=15)
    max_date = pd.Timestamp(year=year_value, month=11, day=15)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.2), constrained_layout=True)
    plot_combined_timeline(ax, summary, "")
    ax.set_xlim(min_date, max_date)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlabel("")

    fig.savefig(args.output)
    plt.close(fig)

    print(f"Saved figure to: {args.output}")
    print(f"Saved summary to: {args.summary_output}")


if __name__ == "__main__":
    main()
