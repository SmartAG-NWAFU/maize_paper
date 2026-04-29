#!/usr/bin/env python
"""Merge the provincial maize GeoTIFFs into one sparse national GeoTIFF.

The input rasters are binary uint8 classification rasters in EPSG:4326.  They
share a resolution but their grids are not perfectly aligned, so this script
resamples each source block to one common grid with nearest-neighbor sampling.

By default, overlapping pixels are merged with max(). This preserves maize
pixels when a neighboring province's background value overlaps them.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from rasterio.warp import reproject


DEFAULT_PATTERN = "classified-*-maize-2024-WGS84-v1.tif"
DEFAULT_OUTPUT = "china-maize-2024-WGS84-v1-mosaic.tif"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Merge provincial maize classification GeoTIFFs into one "
            "national sparse tiled GeoTIFF."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir,
        help="Directory containing source GeoTIFFs. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Input filename glob. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / DEFAULT_OUTPUT,
        help=f"Output GeoTIFF path. Default: {DEFAULT_OUTPUT} in this directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--merge-rule",
        choices=("max", "overwrite"),
        default="max",
        help=(
            "How to handle overlaps. 'max' is recommended for binary 0/1 "
            "maize maps. 'overwrite' lets later files replace earlier files."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=int,
        default=0,
        help="Output nodata/background value. Default: 0.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Output tile size in pixels. Default: 512.",
    )
    parser.add_argument(
        "--compress",
        default="deflate",
        help="GeoTIFF compression. Default: deflate.",
    )
    parser.add_argument(
        "--zlevel",
        type=int,
        default=6,
        help="DEFLATE compression level. Default: 6.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned mosaic geometry and exit without writing.",
    )
    return parser.parse_args()


def list_inputs(input_dir: Path, pattern: str, output: Path) -> list[Path]:
    output = output.resolve()
    paths = []
    for path in sorted(input_dir.glob(pattern)):
        if path.resolve() != output:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No input rasters matched {input_dir / pattern}")
    return paths


def validate_sources(paths: Iterable[Path]) -> tuple[dict, tuple[float, float], tuple[float, float, float, float]]:
    paths = list(paths)
    with rasterio.open(paths[0]) as first:
        profile = first.profile.copy()
        crs = first.crs
        dtype = first.dtypes[0]
        count = first.count
        res_x, res_y = first.res

    lefts: list[float] = []
    bottoms: list[float] = []
    rights: list[float] = []
    tops: list[float] = []

    for path in paths:
        with rasterio.open(path) as src:
            if src.crs != crs:
                raise ValueError(f"CRS mismatch: {path.name} has {src.crs}, expected {crs}")
            if src.count != count:
                raise ValueError(f"Band count mismatch: {path.name} has {src.count}, expected {count}")
            if src.dtypes[0] != dtype:
                raise ValueError(f"Dtype mismatch: {path.name} has {src.dtypes[0]}, expected {dtype}")
            if not math.isclose(src.res[0], res_x, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"X resolution mismatch: {path.name} has {src.res[0]}, expected {res_x}")
            if not math.isclose(src.res[1], res_y, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Y resolution mismatch: {path.name} has {src.res[1]}, expected {res_y}")
            if src.transform.b != 0 or src.transform.d != 0:
                raise ValueError(f"Rotated rasters are not supported: {path.name}")

            lefts.append(src.bounds.left)
            bottoms.append(src.bounds.bottom)
            rights.append(src.bounds.right)
            tops.append(src.bounds.top)

    union_bounds = (min(lefts), min(bottoms), max(rights), max(tops))
    return profile, (res_x, res_y), union_bounds


def aligned_grid(bounds: tuple[float, float, float, float], res: tuple[float, float]) -> tuple[Affine, int, int, tuple[float, float, float, float]]:
    left, bottom, right, top = bounds
    res_x, res_y = res

    out_left = math.floor(left / res_x) * res_x
    out_bottom = math.floor(bottom / res_y) * res_y
    out_right = math.ceil(right / res_x) * res_x
    out_top = math.ceil(top / res_y) * res_y

    width = int(round((out_right - out_left) / res_x))
    height = int(round((out_top - out_bottom) / res_y))
    transform = from_origin(out_left, out_top, res_x, res_y)
    return transform, width, height, (out_left, out_bottom, out_right, out_top)


def bounds_to_window(
    bounds: tuple[float, float, float, float],
    transform: Affine,
    width: int,
    height: int,
) -> Window | None:
    left, bottom, right, top = bounds
    inv = ~transform

    col_left, row_top = inv * (left, top)
    col_right, row_bottom = inv * (right, bottom)

    col_off = max(0, math.floor(min(col_left, col_right)))
    row_off = max(0, math.floor(min(row_top, row_bottom)))
    col_stop = min(width, math.ceil(max(col_left, col_right)))
    row_stop = min(height, math.ceil(max(row_top, row_bottom)))

    win_width = col_stop - col_off
    win_height = row_stop - row_off
    if win_width <= 0 or win_height <= 0:
        return None
    return Window(col_off, row_off, win_width, win_height)


def merge_block(
    dst: rasterio.io.DatasetWriter,
    dst_window: Window,
    src_array: np.ndarray,
    src_transform: Affine,
    src_crs,
    merge_rule: str,
    nodata: int,
) -> bool:
    dst_window = dst_window.round_offsets().round_lengths()
    dst_shape = (int(dst_window.height), int(dst_window.width))
    if dst_shape[0] <= 0 or dst_shape[1] <= 0:
        return False

    projected = np.full(dst_shape, nodata, dtype=dst.dtypes[0])
    reproject(
        source=src_array,
        destination=projected,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=window_transform(dst_window, dst.transform),
        dst_crs=dst.crs,
        dst_nodata=nodata,
        resampling=Resampling.nearest,
    )

    if projected.max() == nodata and merge_rule == "max":
        return False

    if merge_rule == "max":
        current = dst.read(1, window=dst_window)
        merged = np.maximum(current, projected)
    else:
        merged = projected

    dst.write(merged, 1, window=dst_window)
    return True


def create_output_profile(
    base_profile: dict,
    transform: Affine,
    width: int,
    height: int,
    nodata: int,
    block_size: int,
    compress: str,
    zlevel: int,
) -> dict:
    profile = base_profile.copy()
    profile.update(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=block_size,
        blockysize=block_size,
        compress=compress,
        BIGTIFF="YES",
        SPARSE_OK="TRUE",
        NUM_THREADS="ALL_CPUS",
    )
    if compress.lower() == "deflate":
        profile["zlevel"] = zlevel
    return profile


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()

    if output.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    paths = list_inputs(input_dir, args.pattern, output)
    base_profile, res, union_bounds = validate_sources(paths)
    transform, width, height, aligned_bounds = aligned_grid(union_bounds, res)

    print(f"Input rasters: {len(paths)}")
    print(f"Output: {output}")
    print(f"CRS: {base_profile['crs']}")
    print(f"Resolution: {res[0]}, {res[1]}")
    print(f"Aligned bounds: {aligned_bounds}")
    print(f"Output size: {width} x {height} pixels")
    print(f"Merge rule: {args.merge_rule}")

    if args.dry_run:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    profile = create_output_profile(
        base_profile,
        transform,
        width,
        height,
        args.nodata,
        args.block_size,
        args.compress,
        args.zlevel,
    )

    blocks_written = 0
    with rasterio.open(output, "w+", **profile) as dst:
        for source_index, path in enumerate(paths, start=1):
            print(f"[{source_index}/{len(paths)}] {path.name}")
            with rasterio.open(path) as src:
                for _, src_window in src.block_windows(1):
                    src_array = src.read(1, window=src_window)
                    if args.merge_rule == "max" and src_array.max() == args.nodata:
                        continue

                    src_bounds = window_bounds(src_window, src.transform)
                    dst_window = bounds_to_window(src_bounds, dst.transform, dst.width, dst.height)
                    if dst_window is None:
                        continue

                    wrote = merge_block(
                        dst,
                        dst_window,
                        src_array,
                        src.window_transform(src_window),
                        src.crs,
                        args.merge_rule,
                        args.nodata,
                    )
                    if wrote:
                        blocks_written += 1

    print(f"Done. Blocks written: {blocks_written}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RasterioIOError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
