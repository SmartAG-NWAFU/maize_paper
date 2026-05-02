from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "model_outputs_fixed"
DEFAULT_OUTPUT = ROOT / "fig" / "fig9_model_performance_comparison.png"

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "elastic_net": "Elastic Net",
    "linear_regression": "Linear Regression",
}

BASE_FONT_SIZE = 19
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 17
LEGEND_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 16
PANEL_FONT_SIZE = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model performance comparison from outputs/model_outputs_fixed."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "axes.linewidth": 1.1,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle=(0, (2, 2)), alpha=0.28, color="#6C757D")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", length=4.5, width=1.0, pad=6)


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


def load_summary(output_root: Path) -> pd.DataFrame:
    summary_path = output_root / "summary_leaderboard.csv"
    df = pd.read_csv(summary_path)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise ValueError(f"No successful model rows found in {summary_path}")
    return df.sort_values("valid_rmse").reset_index(drop=True)


def plot_model_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    ordered = summary_df.copy()
    ordered["model_label"] = ordered["model"].map(lambda x: MODEL_LABELS.get(x, x))
    x = np.arange(len(ordered))
    width = 0.32
    colors = {"Train": "#2B5C8A", "Validation": "#C97B63"}

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.4), constrained_layout=True)

    for ax, train_col, valid_col, y_label, tag in [
        (axes[0], "train_r2", "valid_r2", r"R²", "(a)"),
        (axes[1], "train_rmse", "valid_rmse", "RMSE (kg ha$^{-1}$)", "(b)"),
    ]:
        ax.bar(
            x - width / 2,
            ordered[train_col],
            width=width,
            color=colors["Train"],
            alpha=0.9,
            label="Train",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            x + width / 2,
            ordered[valid_col],
            width=width,
            color=colors["Validation"],
            alpha=0.82,
            label="Validation",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_ylabel(y_label)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered["model_label"], rotation=0, ha="center")
        ax.legend(frameon=False, loc=(0.66, 0.87))
        style_axis(ax)
        add_panel_tag(ax, tag)

    max_r2 = float(ordered[["train_r2", "valid_r2"]].to_numpy().max())
    axes[0].set_ylim(0, max(0.82, max_r2 * 1.2))

    max_rmse = float(ordered[["train_rmse", "valid_rmse"]].to_numpy().max())
    axes[1].set_ylim(0, max_rmse * 1.2)

    for idx, row in ordered.iterrows():
        axes[0].text(
            idx - width / 2,
            row["train_r2"] + 0.015,
            f"{row['train_r2']:.2f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )
        axes[0].text(
            idx + width / 2,
            row["valid_r2"] + 0.015,
            f"{row['valid_r2']:.2f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )
        rmse_offset = max_rmse * 0.015
        axes[1].text(
            idx - width / 2,
            row["train_rmse"] + rmse_offset,
            f"{row['train_rmse']:.0f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )
        axes[1].text(
            idx + width / 2,
            row["valid_rmse"] + rmse_offset,
            f"{row['valid_rmse']:.0f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    summary_df = load_summary(args.output_root)
    plot_model_comparison(summary_df, args.output)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
