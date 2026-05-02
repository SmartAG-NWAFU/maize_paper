from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "fig"
MODEL_OUTPUT_DIR = ROOT / "outputs" / "model_outputs_fixed"
OPT_OUTPUT = ROOT / "outputs" / "management_yield_region_optimized" / "management_region_baseline_vs_optimal.csv"

REGIONS = ["Hebei", "Shandong"]
REGION_LABELS = {0: "Hebei", 1: "Shandong"}
REGION_COLORS = {"Hebei": "#4E79A7", "Shandong": "#F28E2B"}
SCENARIO_COLORS = {"Baseline": "#A8B2BD", "Max-yield strategy": "#2F6F91"}

BASE_FONT_SIZE = 18
LABEL_FONT_SIZE = 21
TICK_FONT_SIZE = 17
LEGEND_FONT_SIZE = 16
PANEL_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild publication-style figures 3-11.")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
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
            "savefig.bbox": "tight",
            "axes.linewidth": 0.9,
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linestyle=(0, (2, 2)), linewidth=0.65, alpha=0.55, color="#C9D1D9")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#8E99A4")
    ax.tick_params(axis="both", length=3.8, width=0.8, pad=4, color="#8E99A4", labelcolor="#222222")


def add_panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        0.018,
        0.982,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
        color="#111111",
    )


def region_model_data(columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "model_vars.csv", usecols=["Region", *columns]).copy()
    df["region"] = df["Region"].map(REGION_LABELS)
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["region", *columns]).copy()


def plot_region_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    y_label: str,
    tag: str,
    annotation_decimals: int = 0,
) -> None:
    positions = np.arange(len(REGIONS))
    values = [df.loc[df["region"] == region, column].to_numpy(dtype=float) for region in REGIONS]
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F2933", "linewidth": 1.35},
        whiskerprops={"color": "#4B5563", "linewidth": 1.05},
        capprops={"color": "#4B5563", "linewidth": 1.05},
        boxprops={"edgecolor": "#2F343A", "linewidth": 0.9},
    )
    for patch, region in zip(bp["boxes"], REGIONS):
        patch.set_facecolor(REGION_COLORS[region])
        patch.set_alpha(0.34)

    rng = np.random.default_rng(66)
    for pos, region, arr in zip(positions, REGIONS, values):
        jitter = rng.uniform(-0.115, 0.115, size=len(arr))
        ax.scatter(
            np.full(len(arr), pos) + jitter,
            arr,
            s=23,
            color=REGION_COLORS[region],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        mean_value = float(np.nanmean(arr))
        formatted = f"{mean_value:.{annotation_decimals}f}"
        ax.text(
            0.04,
            0.86 - pos * 0.085,
            f"{region} mean: {formatted}",
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "#D7DDE3", "linewidth": 0.4, "alpha": 0.86, "pad": 2.0},
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(REGIONS)
    ax.set_ylabel(y_label)
    style_axis(ax)
    add_panel_tag(ax, tag)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def build_fig3(fig_dir: Path) -> None:
    event_labels = {
        "播种": "Sowing",
        "施肥": "Fertilization",
        "灌溉": "Irrigation",
        "收获": "Harvest",
        "杀菌剂": "Fungicide",
        "杀虫剂": "Insecticide",
        "除草剂": "Herbicide",
        "生长调节剂": "Growth regulator",
    }
    event_order = [
        "Sowing",
        "Fertilization",
        "Herbicide",
        "Insecticide",
        "Irrigation",
        "Fungicide",
        "Growth regulator",
        "Harvest",
    ]
    df = pd.read_csv(DATA_DIR / "management_time.csv", usecols=["region", "实施日期", "措施类型", "措施细类"]).copy()
    df["region"] = df["region"].str.strip().str.lower().map({"hebei": "Hebei", "shandong": "Shandong"})
    raw = df["措施类型"].astype(str).str.strip()
    pesticide = raw.eq("打药")
    raw.loc[pesticide] = df.loc[pesticide, "措施细类"].astype(str).str.strip()
    df["event"] = raw.map(event_labels)
    df["date"] = pd.to_datetime(df["实施日期"], errors="coerce")
    df = df.dropna(subset=["region", "event", "date"])
    df["event"] = pd.Categorical(df["event"], categories=event_order, ordered=True)
    df = df.sort_values(["event", "date", "region"])
    df[["region", "event", "date"]].to_csv(fig_dir / "fig3_region_management_timeline_summary.csv", index=False, encoding="utf-8-sig")

    y_positions = {event: i for i, event in enumerate(event_order)}
    offsets = {"Hebei": -0.16, "Shandong": 0.16}
    rng = np.random.default_rng(66)
    fig, ax = plt.subplots(figsize=(9.4, 5.0), constrained_layout=True)
    for region in REGIONS:
        sub = df[df["region"] == region]
        y = sub["event"].map(y_positions).astype(float).to_numpy() + offsets[region]
        y = y + rng.uniform(-0.04, 0.04, size=len(y))
        ax.scatter(
            sub["date"],
            y,
            s=32,
            color=REGION_COLORS[region],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            label=region,
            zorder=3,
        )
    year = int(df["date"].dt.year.mode().iloc[0])
    ax.set_xlim(pd.Timestamp(year=year, month=5, day=15), pd.Timestamp(year=year, month=11, day=15))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(event_order)
    ax.set_ylim(-0.55, len(event_order) - 0.45)
    ax.invert_yaxis()
    ax.set_ylabel("Management type")
    style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, loc="upper right")
    save(fig, fig_dir / "fig3_region_management_timeline.png")


def build_fig4(fig_dir: Path) -> None:
    df = region_model_data(["Sow_DOY", "Density"])
    summary = (
        df.groupby("region", as_index=False)[["Sow_DOY", "Density"]]
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
    summary.to_csv(fig_dir / "fig4_region_management_difference_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)
    plot_region_boxplot(axes[0], df, "Sow_DOY", "Sowing date (day of year)", "(a)")
    plot_region_boxplot(axes[1], df, "Density", "Plant density (plants ha$^{-1}$)", "(b)")
    save(fig, fig_dir / "fig4_region_sowing_density_boxplots.png")


def build_fig5(fig_dir: Path) -> None:
    df = region_model_data(["Fer_Count", "Fer_N", "Fer_P", "Fer_K"])
    specs = [
        ("Fer_Count", "Fertilization events (count)", "(a)", 1),
        ("Fer_N", "N application rate (kg ha$^{-1}$)", "(b)", 0),
        ("Fer_P", "P application rate (kg ha$^{-1}$)", "(c)", 0),
        ("Fer_K", "K application rate (kg ha$^{-1}$)", "(d)", 0),
    ]
    summary = df.groupby("region", as_index=False)[[s[0] for s in specs]].agg(["count", "mean", "median", "min", "max"]).round(2)
    summary.to_csv(fig_dir / "fig5_region_fertilization_summary.csv", index=True, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.4), constrained_layout=True)
    for ax, (column, label, tag, decimals) in zip(axes.ravel(), specs):
        plot_region_boxplot(ax, df, column, label, tag, decimals)
    save(fig, fig_dir / "fig5_region_fertilization_boxplots.png")


def build_fig6(fig_dir: Path) -> None:
    columns = ["Irr_Count", "Irr_Elec", "Irr_None", "Irr_Flood", "Irr_Sprinkler", "Irr_Drip"]
    df = region_model_data(columns)
    df.groupby("region", as_index=False)[["Irr_Count", "Irr_Elec"]].agg(["count", "mean", "median", "min", "max"]).round(2).to_csv(
        fig_dir / "fig6_region_irrigation_summary.csv", index=True, encoding="utf-8-sig"
    )
    facility = df.groupby("region", as_index=False)[["Irr_None", "Irr_Flood", "Irr_Sprinkler", "Irr_Drip"]].sum()
    facility = facility.rename(
        columns={"Irr_None": "Rainfed", "Irr_Flood": "Flood", "Irr_Sprinkler": "Sprinkler", "Irr_Drip": "Drip"}
    )
    facility.to_csv(fig_dir / "fig6_region_irrigation_summary_facility.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(12.4, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    plot_region_boxplot(ax_a, df, "Irr_Count", "Irrigation events (count)", "(a)", 1)
    plot_region_boxplot(ax_b, df, "Irr_Elec", "Irrigation electricity use (kWh ha$^{-1}$)", "(b)")

    x_labels = ["Rainfed", "Flood", "Sprinkler", "Drip"]
    x = np.arange(len(x_labels))
    width = 0.34
    ordered = facility.set_index("region").reindex(REGIONS)
    max_value = float(ordered[x_labels].to_numpy().max())
    for idx, region in enumerate(REGIONS):
        ax_c.bar(
            x + (idx - 0.5) * width,
            ordered.loc[region, x_labels].to_numpy(dtype=float),
            width=width,
            color=REGION_COLORS[region],
            alpha=0.82,
            edgecolor="#2F343A",
            linewidth=0.8,
            label=region,
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(x_labels)
    ax_c.set_ylabel("Number of fields")
    ax_c.set_ylim(0, max_value * 1.22)
    style_axis(ax_c)
    ax_c.legend(frameon=False, loc="upper right", ncol=2)
    add_panel_tag(ax_c, "(c)")
    save(fig, fig_dir / "fig6_region_irrigation_boxplots.png")


def build_fig7(fig_dir: Path) -> None:
    df = region_model_data(["Pest_Count", "Pest_Cost"]).rename(columns={"Pest_Count": "pest_count", "Pest_Cost": "pest_cost"})
    summary = df.groupby("region", as_index=False)[["pest_count", "pest_cost"]].agg(["count", "mean", "median", "min", "max"]).round(2)
    summary.to_csv(fig_dir / "fig7_region_pesticide_summary.csv", index=True, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)
    plot_region_boxplot(axes[0], df, "pest_count", "Pesticide application frequency", "(a)", 1)
    plot_region_boxplot(axes[1], df, "pest_cost", "Pesticide cost (CNY ha$^{-1}$)", "(b)")
    save(fig, fig_dir / "fig7_region_pesticide_boxplots.png")


def build_fig8(fig_dir: Path) -> None:
    model_df = pd.read_csv(DATA_DIR / "model_vars.csv", usecols=["uuid", "Region", "Yield"]).copy()
    cost_df = pd.read_csv(DATA_DIR / "cost_required_ha.csv", usecols=["uuid", "total_cost_yuan_per_ha"], encoding="utf-8-sig").copy()
    df = model_df.merge(cost_df, on="uuid", how="inner")
    df = df.rename(columns={"Region": "region_code", "Yield": "yield_ha", "total_cost_yuan_per_ha": "cost_ha"})
    df["region"] = df["region_code"].map(REGION_LABELS)
    df["profit_ha"] = df["yield_ha"] * 2.4 - df["cost_ha"]
    df = df.dropna(subset=["region", "yield_ha", "cost_ha", "profit_ha"]).copy()
    df.groupby("region", as_index=False)[["yield_ha", "cost_ha", "profit_ha"]].agg(["count", "mean", "median", "min", "max"]).round(2).to_csv(
        fig_dir / "fig8_region_yield_cost_profit_summary.csv", index=True, encoding="utf-8-sig"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.4), constrained_layout=True)
    specs = [
        ("yield_ha", "Yield (kg ha$^{-1}$)", "(a)"),
        ("cost_ha", "Total cost (CNY ha$^{-1}$)", "(b)"),
        ("profit_ha", "Profit (CNY ha$^{-1}$)", "(c)"),
    ]
    for ax, (column, label, tag) in zip(axes.ravel()[:3], specs):
        plot_region_boxplot(ax, df, column, label, tag)

    ax = axes.ravel()[3]
    x_col, y_col = "cost_ha", "yield_ha"
    x_med, y_med = float(df[x_col].median()), float(df[y_col].median())
    x_low, x_up = x_med * 0.9, x_med * 1.1
    y_low, y_up = y_med * 0.9, y_med * 1.1

    def zone(row: pd.Series) -> str:
        x_val, y_val = row[x_col], row[y_col]
        if x_low <= x_val <= x_up and y_low <= y_val <= y_up:
            return "M"
        if x_val >= x_med and y_val >= y_med:
            return "H-H"
        if x_val < x_med and y_val >= y_med:
            return "L-H"
        if x_val < x_med and y_val < y_med:
            return "L-L"
        return "H-L"

    df["zone"] = df.apply(zone, axis=1)
    markers = {"H-H": "^", "L-H": "o", "M": "s", "L-L": "v", "H-L": "X"}
    for region in REGIONS:
        for zone_label, marker in markers.items():
            sub = df[(df["region"] == region) & (df["zone"] == zone_label)]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=48,
                marker=marker,
                color=REGION_COLORS[region],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
    ax.axvline(x_med, color="#40464D", linewidth=0.9, linestyle="--")
    ax.axhline(y_med, color="#40464D", linewidth=0.9, linestyle="--")
    ax.axvspan(x_low, x_up, color="#B8C2CC", alpha=0.13, zorder=0)
    ax.axhspan(y_low, y_up, color="#B8C2CC", alpha=0.13, zorder=0)
    ax.set_xlabel("Production input cost (CNY ha$^{-1}$)")
    ax.set_ylabel("Grain yield (kg ha$^{-1}$)")
    style_axis(ax, grid_axis="both")
    add_panel_tag(ax, "(d)")
    region_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=REGION_COLORS[r], markeredgecolor="white", markersize=7, label=r)
        for r in REGIONS
    ]
    zone_handles = [
        Line2D([0], [0], marker=m, linestyle="None", color="#505A64", markerfacecolor="#505A64", markersize=6, label=z)
        for z, m in markers.items()
    ]
    first = ax.legend(handles=region_handles, frameon=False, loc="upper right", handletextpad=0.2)
    ax.add_artist(first)
    ax.legend(handles=zone_handles, frameon=False, loc="lower right", ncol=2, columnspacing=0.7, handletextpad=0.2)
    save(fig, fig_dir / "fig8_region_yield_cost_profit_boxplots.png")


def build_fig9(fig_dir: Path) -> None:
    labels = {
        "xgboost": "XGBoost",
        "random_forest": "Random\nForest",
        "elastic_net": "Elastic\nNet",
        "linear_regression": "Linear\nRegression",
    }
    df = pd.read_csv(MODEL_OUTPUT_DIR / "summary_leaderboard.csv")
    df = df[df["status"] == "ok"].sort_values("valid_rmse").reset_index(drop=True)
    df["label"] = df["model"].map(labels)
    x = np.arange(len(df))
    width = 0.32
    colors = {"Train": "#3D6F99", "Validation": "#D1845C"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)
    for ax, train_col, valid_col, ylabel, tag in [
        (axes[0], "train_r2", "valid_r2", "R$^2$", "(a)"),
        (axes[1], "train_rmse", "valid_rmse", "RMSE (kg ha$^{-1}$)", "(b)"),
    ]:
        b1 = ax.bar(x - width / 2, df[train_col], width, color=colors["Train"], alpha=0.9, edgecolor="#2F343A", linewidth=0.8, label="Train")
        b2 = ax.bar(x + width / 2, df[valid_col], width, color=colors["Validation"], alpha=0.86, edgecolor="#2F343A", linewidth=0.8, label="Validation")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"])
        y_max = float(df[[train_col, valid_col]].to_numpy().max())
        ax.set_ylim(0, y_max * 1.22)
        for bars in (b1, b2):
            for bar in bars:
                value = float(bar.get_height())
                text = f"{value:.2f}" if "r2" in train_col else f"{value:.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2, value + y_max * 0.025, text, ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)
        style_axis(ax)
        add_panel_tag(ax, tag)
    axes[0].legend(frameon=False, loc="upper right")
    save(fig, fig_dir / "fig9_model_performance_comparison.png")


def build_fig10(fig_dir: Path) -> None:
    feature_labels = {
        "Irr_Drip": "Drip irrigation",
        "Irr_Sprinkler": "Sprinkler irrigation",
        "Irr_None": "Rainfed",
        "Irr_Flood": "Flood irrigation",
        "Irr_Method": "Irrigation method",
        "RH": "Mean relative humidity",
        "Irr_Elec": "Irrigation electricity use",
        "Density": "Plant density",
        "Pest_Cost": "Pesticide input cost",
        "Fer_K": "K application rate",
        "Rad": "Total radiation",
        "Precip": "Total precipitation",
        "Fer_P": "P application rate",
        "Fer_N": "N application rate",
        "Sow_DOY": "Sowing date",
        "Region": "Region",
        "Irr_Count": "Irrigation frequency",
        "Pest_Count": "Pesticide application frequency",
        "Wind": "Maximum wind speed",
        "Lodging": "Lodging rate",
        "Fer_Count": "Fertilization frequency",
    }
    df = pd.read_csv(MODEL_OUTPUT_DIR / "xgboost" / "feature_importance.csv").copy()
    mask = df["feature"].isin(["Irr_Drip", "Irr_Sprinkler", "Irr_None", "Irr_Flood"])
    if mask.any():
        total = float(df.loc[mask, "importance"].sum())
        df = pd.concat([df.loc[~mask], pd.DataFrame([{"feature": "Irr_Method", "importance": total}])], ignore_index=True)
    df["label"] = df["feature"].map(lambda x: feature_labels.get(x, x.replace("_", " ")))
    df["label"] = df["label"].map(lambda s: "\n".join(textwrap.wrap(s, width=28)))
    df = df.sort_values("importance", ascending=False).head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10.2, 6.6), constrained_layout=True)
    ax.barh(df["label"], df["importance"], height=0.58, color="#4E79A7", alpha=0.92, edgecolor="#2F343A", linewidth=0.8)
    xmax = float(df["importance"].max())
    ax.set_xlim(0, xmax * 1.2)
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Predictor")
    for row in df.itertuples(index=False):
        ax.text(row.importance + xmax * 0.025, row.label, f"{row.importance:.3f}", va="center", fontsize=ANNOTATION_FONT_SIZE)
    style_axis(ax, grid_axis="x")
    save(fig, fig_dir / "fig10_xgboost_feature_importance.png")


def build_fig11(fig_dir: Path) -> None:
    scenario_labels = {"Baseline median": "Baseline", "Optimal yield": "Max-yield strategy"}
    df = pd.read_csv(OPT_OUTPUT).copy()
    df = df[df["result_type"].isin(scenario_labels)].copy()
    for column in ["yield_value", "profit_value", "input_cost"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["scenario"] = df["result_type"].map(scenario_labels)
    df = df.dropna(subset=["region_name", "scenario", "yield_value", "profit_value", "input_cost"])
    pivot = df.pivot(index="region_name", columns="scenario")
    summary = pd.DataFrame({"region": REGIONS})
    for column, prefix in [("yield_value", "yield"), ("profit_value", "profit"), ("input_cost", "input_cost")]:
        baseline = pivot[column]["Baseline"].reindex(REGIONS)
        optimal = pivot[column]["Max-yield strategy"].reindex(REGIONS)
        summary[f"{prefix}_baseline"] = baseline.to_numpy(dtype=float)
        summary[f"{prefix}_optimal"] = optimal.to_numpy(dtype=float)
        summary[f"{prefix}_change"] = (optimal - baseline).to_numpy(dtype=float)
        summary[f"{prefix}_change_pct"] = ((optimal - baseline) / baseline * 100).to_numpy(dtype=float)
    summary.round(2).to_csv(fig_dir / "fig11_region_optimization_summary.csv", index=False, encoding="utf-8-sig")

    specs = [
        ("yield_value", "Grain yield (kg ha$^{-1}$)", "(a)"),
        ("profit_value", "Net profit (CNY ha$^{-1}$)", "(b)"),
        ("input_cost", "Production input cost (CNY ha$^{-1}$)", "(c)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8), constrained_layout=False)
    x = np.arange(len(REGIONS))
    width = 0.32
    for i, (ax, (column, ylabel, tag)) in enumerate(zip(axes, specs)):
        baseline = df[df["scenario"] == "Baseline"].set_index("region_name")[column].reindex(REGIONS).to_numpy(dtype=float)
        optimal = df[df["scenario"] == "Max-yield strategy"].set_index("region_name")[column].reindex(REGIONS).to_numpy(dtype=float)
        b1 = ax.bar(x - width / 2, baseline, width, color=SCENARIO_COLORS["Baseline"], edgecolor="#2F343A", linewidth=0.8, label="Baseline")
        b2 = ax.bar(x + width / 2, optimal, width, color=SCENARIO_COLORS["Max-yield strategy"], edgecolor="#2F343A", linewidth=0.8, label="Max-yield strategy")
        y_max = float(np.nanmax(np.concatenate([baseline, optimal])))
        ax.set_ylim(0, y_max * 1.32)
        for bars in (b1, b2):
            for bar in bars:
                value = float(bar.get_height())
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + y_max * 0.025,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_FONT_SIZE,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(REGIONS)
        ax.set_ylabel(ylabel, labelpad=8)
        style_axis(ax)
        add_panel_tag(ax, tag)
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
    save(fig, fig_dir / "fig11_region_optimization_results.png")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    build_fig3(args.fig_dir)
    build_fig4(args.fig_dir)
    build_fig5(args.fig_dir)
    build_fig6(args.fig_dir)
    build_fig7(args.fig_dir)
    build_fig8(args.fig_dir)
    build_fig9(args.fig_dir)
    build_fig10(args.fig_dir)
    build_fig11(args.fig_dir)


if __name__ == "__main__":
    main()
