from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "fig" / "fig2.png"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "savefig.bbox": "tight",
        }
    )


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    lines: list[str],
    facecolor: str,
    edgecolor: str,
    title_color: str = "#1F2933",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.3,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height - 0.045,
        title,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + 0.05,
        y + height - 0.115,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.22,
        color="#222222",
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        color=color,
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(arrow)


def plot_workflow(output: Path = DEFAULT_OUTPUT) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(15.4, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "green_fill": "#EAF3E4",
        "green_edge": "#6E9B5D",
        "blue_fill": "#E7F0F8",
        "blue_edge": "#5386B3",
        "orange_fill": "#F8E6D9",
        "orange_edge": "#C97B63",
        "gray_fill": "#F1F3F5",
        "gray_edge": "#6C757D",
        "gold_fill": "#F4ECD1",
        "gold_edge": "#AA8A3B",
    }

    add_box(
        ax,
        (0.04, 0.67),
        0.24,
        0.24,
        "Study fields",
        ["Hebei: 26 fields", "Shandong: 55 fields", "Summer maize season"],
        colors["green_fill"],
        colors["green_edge"],
    )
    add_box(
        ax,
        (0.38, 0.67),
        0.24,
        0.24,
        "Field data",
        ["Weather and field status", "Management and input costs", "Yield and profit"],
        colors["green_fill"],
        colors["green_edge"],
    )
    add_box(
        ax,
        (0.72, 0.67),
        0.24,
        0.24,
        "Predictors",
        ["Sowing date and density", "N, P, K and irrigation", "Pest control and lodging"],
        colors["green_fill"],
        colors["green_edge"],
    )

    add_arrow(ax, (0.29, 0.79), (0.37, 0.79), colors["green_edge"])
    add_arrow(ax, (0.63, 0.79), (0.71, 0.79), colors["green_edge"])

    add_box(
        ax,
        (0.08, 0.36),
        0.24,
        0.21,
        "Pre-processing",
        ["Numeric conversion", "Missing-value imputation", "Train/validation split"],
        colors["gray_fill"],
        colors["gray_edge"],
    )
    add_box(
        ax,
        (0.38, 0.36),
        0.24,
        0.21,
        "Yield modelling",
        ["OLS and Elastic Net", "Random forest", "XGBoost"],
        colors["blue_fill"],
        colors["blue_edge"],
    )
    add_box(
        ax,
        (0.68, 0.36),
        0.24,
        0.21,
        "Model evaluation",
        ["R² and RMSE", "Validation performance", "Predictor importance"],
        colors["blue_fill"],
        colors["blue_edge"],
    )

    add_arrow(ax, (0.49, 0.66), (0.49, 0.58), colors["gray_edge"])
    add_arrow(ax, (0.33, 0.47), (0.37, 0.47), colors["blue_edge"])
    add_arrow(ax, (0.63, 0.47), (0.67, 0.47), colors["blue_edge"])

    add_box(
        ax,
        (0.14, 0.08),
        0.28,
        0.19,
        "Constrained optimization",
        ["Empirical management bounds", "Differential evolution search", "Regional optimized scenario"],
        colors["orange_fill"],
        colors["orange_edge"],
    )
    add_box(
        ax,
        (0.58, 0.08),
        0.28,
        0.19,
        "Outcomes",
        ["Yield improvement potential", "Profit response", "Region-specific management levers"],
        colors["gold_fill"],
        colors["gold_edge"],
    )

    add_arrow(ax, (0.79, 0.35), (0.73, 0.28), colors["orange_edge"])
    add_arrow(ax, (0.43, 0.18), (0.57, 0.18), colors["orange_edge"])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> None:
    plot_workflow()
    print(f"Saved figure to: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
