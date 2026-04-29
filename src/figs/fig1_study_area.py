from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import mercantile
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
import requests
from dotenv import load_dotenv
from PIL import Image
from pyproj import CRS, Transformer
import matplotlib.patches as patches


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "trial_locations.csv"
DEFAULT_OUTPUT = ROOT / "fig" / "fig1_study_area_map.png"
CHINA_SHP = ROOT / "data" / "china_shp" / "国家矢量.shp"
NCP_SHP = ROOT / "data" / "china_shp" / "ncp.shp"
MAIZE_RASTER = ROOT / "data" / "china_shp" / "maize_map" / "china-maize-2024-WGS84-v1-mosaic_albers_1000m_average.tif"

REGION_STYLE = {
    "shandong": {"label": "Shandong", "color": "#f28e2b", "china_label_offset": (28, -10)},
    "hebei": {"label": "Hebei", "color": "#4e79a7", "china_label_offset": (-22, 14)},
}

BASE_FONT_SIZE = 16
TICK_FONT_SIZE = 22
PANEL_LABEL_SIZE = 20
REGION_LABEL_SIZE = 15
POINT_LABEL_SIZE = 24
SCALE_FONT_SIZE = 14
NORTH_ARROW_FONT_SIZE = 18
CHINA_LABEL_SIZE = 18
SOUTH_CHINA_SEA_LONLAT_BOUNDS = (105.0, 125.0, 3.0, 25.0)
CHINA_OVERVIEW_LONLAT_BOUNDS = (73.0, 135.5, 17.0, 54.5)
OVERVIEW_LABEL_SIZE = 26
OVERVIEW_ROMAN_SIZE = 30
SOUTH_CHINA_SEA_INSET_POS = (0.835, 0.0, 0.18, 0.30)
WEB_MERCATOR_CRS = "EPSG:3857"
MAIZE_RASTER_MAX_DIMENSION = 3600
MAIZE_MIN_FRACTION = 0.001
MAIZE_RGB = (0.93, 0.57, 0.09)
MAIZE_ALPHA_MIN = 0.22
MAIZE_ALPHA_MAX = 0.84
NCP_FACE_COLOR = "#6bbf8f"
NCP_EDGE_COLOR = "#145a42"
NCP_EDGE_LINEWIDTH = 3.5
CHINA_PANEL_FACE_COLOR = "#edf3f0"
CHINA_PANEL_HEIGHT_RATIO = 1.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot study area map with Tianditu imagery.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file containing trial points with longitude and latitude.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=13,
        help="Tile zoom level used for the Tianditu basemap.",
    )
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=0,
        help="Label the first N points in the source table. Set 0 to disable labels.",
    )
    return parser.parse_args()


def build_tianditu_url(layer: str, tk: str) -> str:
    return (
        f"https://t0.tianditu.gov.cn/{layer}_w/wmts?"
        "SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
        f"&LAYER={layer}&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles"
        "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
        f"&tk={tk}"
    )


def load_points(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"region", "centroid_lon", "centroid_lat"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")

    df = df.dropna(subset=["centroid_lon", "centroid_lat"]).copy()
    df["region"] = df["region"].astype(str).str.strip().str.lower()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["centroid_lon"], df["centroid_lat"]),
        crs="EPSG:4326",
    )
    return gdf.to_crs(epsg=3857)


def load_china_boundary() -> gpd.GeoDataFrame:
    if not CHINA_SHP.exists():
        raise FileNotFoundError(f"China shapefile not found: {CHINA_SHP}")

    china = gpd.read_file(CHINA_SHP)
    if china.empty:
        raise ValueError(f"China shapefile is empty: {CHINA_SHP}")

    if china.crs is None:
        minx, miny, maxx, maxy = china.total_bounds
        looks_like_lonlat = all(
            [
                -180 <= minx <= 180,
                -180 <= maxx <= 180,
                -90 <= miny <= 90,
                -90 <= maxy <= 90,
            ]
        )
        if looks_like_lonlat:
            china = china.set_crs("EPSG:4326")
        else:
            china = china.set_crs(
                CRS.from_proj4(
                    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
                    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
                )
            )

    return china.to_crs(epsg=3857)


def load_ncp_boundary() -> gpd.GeoDataFrame:
    if not NCP_SHP.exists():
        raise FileNotFoundError(f"NCP shapefile not found: {NCP_SHP}")

    ncp = gpd.read_file(NCP_SHP)
    if ncp.empty:
        raise ValueError(f"NCP shapefile is empty: {NCP_SHP}")

    if ncp.crs is None:
        prj_path = NCP_SHP.with_suffix(".prj")
        if not prj_path.exists():
            raise ValueError(f"NCP shapefile has no CRS and no .prj file: {NCP_SHP}")
        ncp = ncp.set_crs(CRS.from_wkt(prj_path.read_text(encoding="utf-8")))

    return ncp.to_crs(epsg=3857)


def maize_fraction_to_rgba(data: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*data.shape, 4), dtype=np.float32)
    rgba[..., 0] = MAIZE_RGB[0]
    rgba[..., 1] = MAIZE_RGB[1]
    rgba[..., 2] = MAIZE_RGB[2]

    valid = np.isfinite(data) & (data > MAIZE_MIN_FRACTION)
    if np.any(valid):
        scaled = np.clip((data[valid] - MAIZE_MIN_FRACTION) / (1.0 - MAIZE_MIN_FRACTION), 0.0, 1.0)
        rgba[..., 3][valid] = MAIZE_ALPHA_MIN + np.sqrt(scaled) * (MAIZE_ALPHA_MAX - MAIZE_ALPHA_MIN)

    return rgba


def load_maize_raster_3857(
    raster_path: Path = MAIZE_RASTER,
    max_dimension: int = MAIZE_RASTER_MAX_DIMENSION,
) -> tuple[np.ndarray, list[float]]:
    if not raster_path.exists():
        raise FileNotFoundError(f"Maize raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Maize raster has no CRS: {raster_path}")

        left, bottom, right, top = transform_bounds(src.crs, WEB_MERCATOR_CRS, *src.bounds, densify_pts=21)
        x_span = right - left
        y_span = top - bottom
        resolution = max(x_span / max_dimension, y_span / max_dimension)
        width = max(1, int(np.ceil(x_span / resolution)))
        height = max(1, int(np.ceil(y_span / resolution)))
        dst_transform = from_bounds(left, bottom, right, top, width, height)
        dst = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=WEB_MERCATOR_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

    return maize_fraction_to_rgba(dst), [left, right, bottom, top]


def add_china_panel_context_layers(ax: plt.Axes) -> gpd.GeoDataFrame:
    ncp = load_ncp_boundary()
    ncp.plot(ax=ax, facecolor=NCP_FACE_COLOR, edgecolor="none", alpha=0.20, zorder=2.6)

    maize_rgba, maize_extent = load_maize_raster_3857()
    ax.imshow(maize_rgba, extent=maize_extent, interpolation="nearest", zorder=3.2)

    return ncp


def group_centroid(group: gpd.GeoDataFrame):
    geometry = group.geometry.union_all() if hasattr(group.geometry, "union_all") else group.unary_union
    return geometry.centroid


def expand_bounds(bounds: tuple[float, float, float, float], padding_ratio: float = 0.12) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    dx = maxx - minx
    dy = maxy - miny
    pad_x = max(dx * padding_ratio, 5000)
    pad_y = max(dy * padding_ratio, 5000)
    return minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y


def lonlat_bounds_to_3857(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    west, east, south, north = bounds
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    minx, miny = transformer.transform(west, south)
    maxx, maxy = transformer.transform(east, north)
    return minx, maxx, miny, maxy


def download_tile(tile: mercantile.Tile, url_template: str, session: requests.Session) -> Image.Image:
    url = url_template.format(x=tile.x, y=tile.y, z=tile.z)
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return Image.open(__import__("io").BytesIO(response.content)).convert("RGBA")


def rgba_to_grayscale(image: Image.Image, alpha_scale: float = 1.0) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    gray = np.dot(rgb, np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    out = np.concatenate([gray_rgb, np.clip(alpha * alpha_scale, 0, 255)], axis=2)
    return Image.fromarray(out.astype(np.uint8), mode="RGBA")


def merge_tiles(bounds_3857: tuple[float, float, float, float], zoom: int, url_template: str, session: requests.Session) -> tuple[Image.Image, list[float]]:
    minx, maxx, miny, maxy = bounds_3857
    lonlat_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = lonlat_transformer.transform(minx, miny)
    east, north = lonlat_transformer.transform(maxx, maxy)

    tiles = list(mercantile.tiles(west, south, east, north, [zoom]))
    if not tiles:
        raise ValueError("No map tiles found for the requested bounds.")

    xs = sorted({tile.x for tile in tiles})
    ys = sorted({tile.y for tile in tiles})
    width = 256 * len(xs)
    height = 256 * len(ys)
    mosaic = Image.new("RGBA", (width, height))

    x_index = {x: idx for idx, x in enumerate(xs)}
    y_index = {y: idx for idx, y in enumerate(ys)}

    for tile in tiles:
        image = download_tile(tile, url_template, session)
        offset = (x_index[tile.x] * 256, y_index[tile.y] * 256)
        mosaic.paste(image, offset)

    west_all, south_all, east_all, north_all = mercantile.bounds(xs[0], ys[-1], zoom)
    west_last, south_last, east_last, north_last = mercantile.bounds(xs[-1], ys[0], zoom)
    extent_4326 = [west_all, east_last, south_all, north_last]

    mercator_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    west_m, south_m = mercator_transformer.transform(extent_4326[0], extent_4326[2])
    east_m, north_m = mercator_transformer.transform(extent_4326[1], extent_4326[3])
    return mosaic, [west_m, east_m, south_m, north_m]


def add_basemap(
    ax: plt.Axes,
    bounds_3857: tuple[float, float, float, float],
    zoom: int,
    tianditu_key: str,
    layer: str = "img",
    grayscale: bool = False,
    alpha: float = 1.0,
) -> None:
    session = requests.Session()
    try:
        img, img_extent = merge_tiles(bounds_3857, zoom, build_tianditu_url(layer, tianditu_key), session)
    finally:
        session.close()

    if grayscale:
        img = rgba_to_grayscale(img, alpha_scale=alpha)

    ax.imshow(img, extent=img_extent, interpolation="bilinear", zorder=1)
    if layer == "img":
        ax.add_patch(
            patches.Rectangle(
                (bounds_3857[0], bounds_3857[2]),
                bounds_3857[1] - bounds_3857[0],
                bounds_3857[3] - bounds_3857[2],
                facecolor="white",
                edgecolor="none",
                alpha=0.15,
                zorder=2,
            )
        )


def add_point_labels(ax: plt.Axes, gdf_3857: gpd.GeoDataFrame, label_top_n: int) -> None:
    if label_top_n <= 0 or "new_name" not in gdf_3857.columns:
        return

    for _, row in gdf_3857.head(label_top_n).iterrows():
        ax.annotate(
            row["new_name"],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=POINT_LABEL_SIZE,
            color="black",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            zorder=6,
        )


def nice_degree_step(span: float) -> float:
    candidates = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0])
    target = max(span / 3.5, candidates[0])
    return float(candidates[np.argmin(np.abs(candidates - target))])


def make_degree_ticks(min_value: float, max_value: float, step: float | None = None) -> np.ndarray:
    if np.isclose(min_value, max_value):
        return np.array([min_value])

    step = step if step is not None else nice_degree_step(max_value - min_value)
    start = np.ceil(min_value / step) * step
    end = np.floor(max_value / step) * step
    ticks = np.arange(start, end + step * 0.5, step)

    if ticks.size < 3:
        ticks = np.linspace(min_value, max_value, 3)
    return np.round(ticks, 4)


def format_lon_label(value: float) -> str:
    return f"{value:.2f} °E"


def format_lat_label(value: float) -> str:
    return f"{value:.2f} °N"


def format_lon_label_overview(value: float) -> str:
    return f"{value:.0f} °E"


def format_lat_label_overview(value: float) -> str:
    return f"{value:.0f} °N"


def add_lon_lat_ticks(
    ax: plt.Axes,
    bounds_3857: tuple[float, float, float, float],
    lon_step: float | None = None,
    lat_step: float | None = None,
) -> None:
    minx, maxx, miny, maxy = bounds_3857
    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    west, south = to_lonlat.transform(minx, miny)
    east, north = to_lonlat.transform(maxx, maxy)

    lon_ticks = make_degree_ticks(west, east, step=lon_step)
    lat_ticks = make_degree_ticks(south, north, step=lat_step)

    x_tick_positions = [to_mercator.transform(lon, south)[0] for lon in lon_ticks]
    y_tick_positions = [to_mercator.transform(west, lat)[1] for lat in lat_ticks]

    ax.set_xticks(x_tick_positions)
    ax.set_yticks(y_tick_positions)
    ax.set_xticklabels([format_lon_label(lon) for lon in lon_ticks], fontsize=TICK_FONT_SIZE)
    ax.set_yticklabels([format_lat_label(lat) for lat in lat_ticks], fontsize=TICK_FONT_SIZE, va="center")
    ax.tick_params(axis="both", direction="out", length=4.5, width=1.0, colors="black", pad=4)
    ax.grid(True, linestyle="--", linewidth=0.55, color="white", alpha=0.35, zorder=3)


def add_overview_ticks(ax: plt.Axes, bounds_3857: tuple[float, float, float, float]) -> None:
    minx, maxx, miny, maxy = bounds_3857
    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    west, south = to_lonlat.transform(minx, miny)
    east, north = to_lonlat.transform(maxx, maxy)

    lon_ticks = np.arange(np.ceil(west / 10) * 10, np.floor(east / 10) * 10 + 0.5, 10)
    lat_ticks = np.arange(np.ceil(south / 10) * 10, np.floor(north / 10) * 10 + 0.5, 10)
    x_tick_positions = [to_mercator.transform(lon, south)[0] for lon in lon_ticks]
    y_tick_positions = [to_mercator.transform(west, lat)[1] for lat in lat_ticks]

    ax.set_xticks(x_tick_positions)
    ax.set_yticks(y_tick_positions)
    ax.set_xticklabels([format_lon_label_overview(lon) for lon in lon_ticks], fontsize=TICK_FONT_SIZE)
    ax.set_yticklabels([format_lat_label_overview(lat) for lat in lat_ticks], fontsize=TICK_FONT_SIZE)
    ax.tick_params(
        axis="both",
        which="major",
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=True,
        labelbottom=False,
        labelleft=True,
        labelright=True,
        direction="out",
        length=6,
        width=1.2,
        colors="black",
        pad=4,
    )


def choose_scale_length(width_m: float) -> int:
    candidates = [500, 1000, 2000, 5000, 10000, 20000, 25000, 50000, 100000]
    target = width_m * 0.18
    valid = [value for value in candidates if value <= target]
    return valid[-1] if valid else candidates[0]


def add_scale_bar(ax: plt.Axes, bounds_3857: tuple[float, float, float, float]) -> None:
    minx, maxx, miny, maxy = bounds_3857
    width_m = maxx - minx
    height_m = maxy - miny
    scale_length = choose_scale_length(width_m)

    x0 = minx + width_m * 0.08
    y0 = miny + height_m * 0.08
    bar_height = height_m * 0.010
    half_length = scale_length / 2

    ax.add_patch(plt.Rectangle((x0, y0), half_length, bar_height, facecolor="black", edgecolor="black", zorder=7))
    ax.add_patch(plt.Rectangle((x0 + half_length, y0), half_length, bar_height, facecolor="white", edgecolor="black", zorder=7))

    tick_y = y0 + bar_height
    for xpos in (x0, x0 + half_length, x0 + scale_length):
        ax.plot([xpos, xpos], [y0, tick_y], color="black", linewidth=0.9, zorder=8)

    unit = "km" if scale_length >= 1000 else "m"
    factor = 1000 if unit == "km" else 1
    labels = ["0", f"{half_length / factor:g}", f"{scale_length / factor:g} {unit}"]
    for xpos, label in zip((x0, x0 + half_length, x0 + scale_length), labels):
        ax.text(
            xpos,
            y0 - bar_height * 1.15,
            label,
            ha="center",
            va="top",
            fontsize=SCALE_FONT_SIZE,
            color="black",
            zorder=8,
        )


def add_north_arrow(ax: plt.Axes, bounds_3857: tuple[float, float, float, float]) -> None:
    minx, maxx, miny, maxy = bounds_3857
    width = maxx - minx
    height = maxy - miny

    # 位置
    x = maxx - width * 0.10
    y = maxy - height * 0.16
    h = height * 0.075
    w = width * 0.022

    # 顶点（箭头尖）
    tip = (x, y + h)
    left = (x - w, y)
    right = (x + w, y)
    center = (x, y + h * 0.45)

    tri_black = patches.Polygon(
        [left, center, tip],
        closed=True,
        facecolor="black",
        edgecolor="black",
        linewidth=1.1,
        zorder=8
    )

    tri_white = patches.Polygon(
        [center, right, tip],
        closed=True,
        facecolor="white",
        edgecolor="black",
        linewidth=1.1,
        zorder=9
    )

    arrow_bg = patches.FancyBboxPatch(
        (x - width * 0.045, y - height * 0.02),
        width * 0.09,
        height * 0.16,
        boxstyle="round,pad=0.02",
        facecolor="white",
        edgecolor="none",
        alpha=0.58,
        zorder=7,
    )
    ax.add_patch(arrow_bg)
    ax.add_patch(tri_black)
    ax.add_patch(tri_white)

    ax.text(
        x,
        y + h * 1.05,
        "N",
        ha="center",
        va="bottom",
        fontsize=NORTH_ARROW_FONT_SIZE,
        fontweight="bold",
        color="black",
        zorder=10
    )


def add_overview_north_arrow(ax: plt.Axes, bounds_3857: tuple[float, float, float, float]) -> None:
    minx, maxx, miny, maxy = bounds_3857
    width = maxx - minx
    height = maxy - miny

    x = maxx - width * 0.06
    y = maxy - height * 0.18
    h = height * 0.11
    w = width * 0.016

    tip = (x, y + h)
    left = (x - w, y)
    right = (x + w, y)
    inner = (x, y + h * 0.42)

    ax.add_patch(
        patches.Polygon([left, inner, tip], closed=True, facecolor="black", edgecolor="black", linewidth=1.1, zorder=8)
    )
    ax.add_patch(
        patches.Polygon([inner, right, tip], closed=True, facecolor="white", edgecolor="black", linewidth=1.1, zorder=9)
    )
    ax.text(x, y + h * 1.08, "N", ha="center", va="bottom", fontsize=NORTH_ARROW_FONT_SIZE + 2, color="black", zorder=10)


def add_panel_label(ax: plt.Axes, text: str) -> None:
    panel_text = ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color="#dbe6e5",
        zorder=9,
    )
    panel_text.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black", alpha=0.8)])


def add_south_china_sea_inset(ax: plt.Axes, china: gpd.GeoDataFrame) -> None:
    inset = ax.inset_axes(SOUTH_CHINA_SEA_INSET_POS, transform=ax.transAxes)
    minx, maxx, miny, maxy = lonlat_bounds_to_3857(SOUTH_CHINA_SEA_LONLAT_BOUNDS)
    inset.set_xlim(minx, maxx)
    inset.set_ylim(miny, maxy)
    inset.set_facecolor(CHINA_PANEL_FACE_COLOR)
    china.plot(ax=inset, facecolor="none", edgecolor="black", linewidth=0.7, zorder=3)
    china.boundary.plot(ax=inset, color="black", linewidth=0.55, zorder=4)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_aspect("equal")
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")


def plot_china_panel(
    ax: plt.Axes,
    points: gpd.GeoDataFrame,
    region_bounds: dict[str, tuple[float, float, float, float]],
) -> None:
    _ = region_bounds
    china = load_china_boundary()
    bounds_3857 = lonlat_bounds_to_3857(CHINA_OVERVIEW_LONLAT_BOUNDS)

    ax.set_xlim(bounds_3857[0], bounds_3857[1])
    ax.set_ylim(bounds_3857[2], bounds_3857[3])
    ax.set_facecolor(CHINA_PANEL_FACE_COLOR)

    ncp = add_china_panel_context_layers(ax)
    china.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, zorder=4)
    china.boundary.plot(ax=ax, color="black", linewidth=1, zorder=4.4)
    ncp.boundary.plot(ax=ax, color=NCP_EDGE_COLOR, linewidth=NCP_EDGE_LINEWIDTH, zorder=5.4)

    for region, group in points.groupby("region"):
        style = REGION_STYLE.get(region, {"label": region.title(), "color": "#f2a81d"})
        group.plot(
            ax=ax,
            markersize=32,
            color=style["color"],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.95,
            zorder=6,
        )

    add_south_china_sea_inset(ax, china)
    add_overview_ticks(ax, bounds_3857)
    add_overview_north_arrow(ax, bounds_3857)

    ax.set_aspect("auto")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")


def plot_region_panel(
    ax: plt.Axes,
    region: str,
    group: gpd.GeoDataFrame,
    zoom: int,
    label_top_n: int,
    tianditu_key: str,
    idx: int,
    ) -> None:
    _ = idx
    style = REGION_STYLE.get(region, {"label": region.title(), "color": "#e15759"})
    minx, maxx, miny, maxy = expand_bounds(tuple(group.total_bounds), padding_ratio=0.18)
    bounds_3857 = (minx, maxx, miny, maxy)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    add_basemap(ax, bounds_3857, zoom, tianditu_key)

    group.plot(
        ax=ax,
        markersize=72,
        color=style["color"],
        edgecolor="white",
        linewidth=2,
        alpha=0.95,
        zorder=4,
    )
    add_point_labels(ax, group, label_top_n)
    lat_step = 0.2 if idx == 0 else 0.04 if idx == 1 else None
    add_lon_lat_ticks(ax, bounds_3857, lat_step=lat_step)
    add_scale_bar(ax, bounds_3857)
    if idx == 1:
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    else:
        ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)

    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("E" if idx == 0 else "W")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")


def plot_map(points: gpd.GeoDataFrame, output_path: Path, zoom: int, label_top_n: int, tianditu_key: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 1,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "figure.titlesize": BASE_FONT_SIZE + 2,
            "axes.linewidth": 1.0,
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )

    region_order = [region for region in REGION_STYLE if region in points["region"].unique()]
    remaining_regions = sorted(set(points["region"]) - set(region_order))
    region_order.extend(remaining_regions)
    region_order = region_order[:2]
    region_bounds = {
        region: expand_bounds(tuple(points.loc[points["region"] == region].total_bounds), padding_ratio=0.18)
        for region in region_order
    }

    fig = plt.figure(figsize=(16.8, 11.9), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[CHINA_PANEL_HEIGHT_RATIO, 1.0], hspace=0.02, wspace=0.012)

    china_ax = fig.add_subplot(gs[0, :])
    plot_china_panel(china_ax, points, region_bounds=region_bounds)
    add_panel_label(china_ax, "(a)")

    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for idx, (ax, region) in enumerate(zip(axes, region_order)):
        group = points.loc[points["region"] == region].copy()
        region_label = REGION_STYLE.get(region, {"label": region.title()})["label"]
        plot_region_panel(ax=ax, region=region, group=group, zoom=zoom, label_top_n=label_top_n, tianditu_key=tianditu_key, idx=idx)
        add_panel_label(ax, f"({chr(98 + idx)}) {region_label}")
        if idx > 0:
            ax.set_ylabel("")

    for ax in axes[len(region_order):]:
        ax.axis("off")

    fig.subplots_adjust(left=0.038, right=0.992, bottom=0.05, top=0.99)
    china_pos = china_ax.get_position()
    left_pos = axes[0].get_position()
    right_pos = axes[1].get_position()
    china_ax.set_position([left_pos.x0, china_pos.y0, right_pos.x1 - left_pos.x0, china_pos.height])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    load_dotenv()

    tianditu_key = os.getenv("tianditu_api_key")
    if not tianditu_key:
        raise EnvironmentError("Missing tianditu_api_key in environment variables or .env file.")

    points = load_points(args.input)
    plot_map(points=points, output_path=args.output, zoom=args.zoom, label_top_n=args.label_top_n, tianditu_key=tianditu_key)
    print(f"Saved study area map to: {args.output}")


if __name__ == "__main__":
    main()
