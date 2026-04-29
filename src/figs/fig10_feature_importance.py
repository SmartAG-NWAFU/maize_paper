from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "model_outputs_fixed"
DEFAULT_OUTPUT = ROOT / "fig" / "fig10_xgboost_feature_importance.png"

FEATURE_LABELS = {
    "Irr_Drip": "Irrigation = Drip",
    "Irr_Sprinkler": "Irrigation = Sprinkler",
    "Irr_None": "Irrigation = None",
    "Irr_Flood": "Irrigation = Flood",
    "Irr_Method": "Irrigation method",
    "RH": "Relative humidity",
    "Irr_Elec": "Irrigation electricity",
    "Density": "Plant density",
    "Pest_Cost": "Pesticide cost",
    "Fer_K": "Total K",
    "Rad": "Radiation",
    "Precip": "Precipitation",
    "Fer_P": "Total P",
    "Fer_N": "Total N",
    "Sow_DOY": "Sowing day of year",
    "Region": "Region",
    "Irr_Count": "Irrigation count",
    "Pest_Count": "Pesticide count",
    "Wind": "Wind speed",
    "Lodging": "Lodging",
    "Fer_Count": "Fertilization count",
}

BASE_FONT_SIZE = 19
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot XGBoost feature importance from outputs/model_outputs_fixed."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=10)
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
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "axes.linewidth": 1.1,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", linestyle=(0, (2, 2)), alpha=0.28, color="#6C757D")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", length=4.5, width=1.0, pad=6)


def load_feature_importance(output_root: Path, top_n: int) -> pd.DataFrame:
    importance_path = output_root / "xgboost" / "feature_importance.csv"
    df = pd.read_csv(importance_path).copy()
    irrigation_mask = df["feature"].isin(["Irr_Drip", "Irr_Sprinkler", "Irr_None", "Irr_Flood"])
    if irrigation_mask.any():
        irrigation_total = float(df.loc[irrigation_mask, "importance"].sum())
        df = df.loc[~irrigation_mask].copy()
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{"feature": "Irr_Method", "importance": irrigation_total, "method": "aggregated"}]
                ),
            ],
            ignore_index=True,
        )
    df["feature_label"] = df["feature"].map(lambda x: FEATURE_LABELS.get(x, x.replace("_", " ")))
    return df.sort_values("importance", ascending=False).head(top_n).sort_values("importance").reset_index(drop=True)


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.8), constrained_layout=True)
    ax.barh(
        importance_df["feature_label"],
        importance_df["importance"],
        height=0.6,
        color="#5386B3",
        alpha=0.95,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    style_axis(ax)

    xmax = float(importance_df["importance"].max())
    ax.set_xlim(0, xmax * 1.18 if xmax > 0 else 1.0)
    for row in importance_df.itertuples(index=False):
        ax.text(
            row.importance + xmax * 0.03,
            row.feature_label,
            f"{row.importance:.3f}",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    importance_df = load_feature_importance(args.output_root, args.top_n)
    plot_feature_importance(importance_df, args.output)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
