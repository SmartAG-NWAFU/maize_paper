from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "model_vars.csv"
DEFAULT_OUTPUT = ROOT / "fig" / "fig4_region_sowing_density_boxplots.png"
DEFAULT_SUMMARY_OUTPUT = ROOT / "fig" / "fig4_region_management_difference_summary.csv"

REGION_LABELS = {
    0: "Hebei",
    1: "Shandong",
}

REGION_COLORS = {
    "Hebei": "#DD8C49",
    "Shandong": "#58A5CF",
}

BASE_FONT_SIZE = 19
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 19
BOX_X_TICK_FONT_SIZE = 25
LEGEND_FONT_SIZE = 21
ANNOTATION_FONT_SIZE = 19
PANEL_FONT_SIZE = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sowing date and plant density boxplots by region."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file containing regional sowing and density observations.",
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
        usecols=["Region", "Sow_DOY", "Density"],
    ).copy()
    df = df.rename(
        columns={
            "Region": "region_code",
            "Sow_DOY": "sowing_day_of_year",
            "Density": "measured_density",
        }
    )
    df["region"] = df["region_code"].map(REGION_LABELS)
    df["sowing_day_of_year"] = pd.to_numeric(df["sowing_day_of_year"], errors="coerce")
    df["measured_density"] = pd.to_numeric(df["measured_density"], errors="coerce")
    return df.dropna(subset=["region", "sowing_day_of_year", "measured_density"]).copy()


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("region", as_index=False)[["sowing_day_of_year", "measured_density"]]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )
    summary.columns = [
        "region",
        "sowing_count",
        "sowing_mean",
        "sowing_median",
        "sowing_min",
        "sowing_max",
        "density_count",
        "density_mean",
        "density_median",
        "density_min",
        "density_max",
    ]
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
    values = [df.loc[df["region"] == region, column].to_numpy() for region in regions]

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
            f"{region} mean: {mean_value:.0f}",
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


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    summary = build_summary_table(df)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.8), constrained_layout=True)
    plot_boxplot(axes[0], df, "sowing_day_of_year", "Sowing date (day of year)", "(a)")
    plot_boxplot(axes[1], df, "measured_density", "Plant density (plants ha$^{-1}$)", "(b)")
    fig.savefig(args.output)
    plt.close(fig)

    print(f"Saved figure to: {args.output}")
    print(f"Saved summary to: {args.summary_output}")


if __name__ == "__main__":
    main()
