#!/usr/bin/env python
"""Resample the merged maize raster to a 1000 m grid.

The mosaic is stored in EPSG:4326, whose pixel units are degrees.  A real
1000 m grid therefore requires reprojection to a projected CRS first.  The
default target CRS below is a WGS84 Albers equal-area projection suitable for
China-scale raster aggregation.

For the binary maize map, the default ``average`` method writes a float32
maize fraction from 0 to 1 for each 1000 m cell.  Use ``--method max`` for a
binary presence map or ``--method nearest`` for nearest-neighbor resampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import calculate_default_transform, reproject


DEFAULT_INPUT = "china-maize-2024-WGS84-v1-mosaic.tif"
DEFAULT_DST_CRS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
METHODS = ("average", "max", "nearest", "mode")
DEFAULT_SRC_NODATA = 255.0


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Reproject and resample the maize mosaic to a meter-based grid."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / DEFAULT_INPUT,
        help=f"Input mosaic GeoTIFF. Default: {DEFAULT_INPUT} in this directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GeoTIFF path. Default is derived from input, method, and resolution.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1000.0,
        help="Output pixel size in target CRS units. Default: 1000 meters.",
    )
    parser.add_argument(
        "--dst-crs",
        default=DEFAULT_DST_CRS,
        help=(
            "Target projected CRS. Default: WGS84 Albers equal-area for China. "
            "You can also pass values like EPSG:3857."
        ),
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="average",
        help=(
            "Resampling method. average gives maize fraction 0-1; max gives "
            "presence/absence; nearest keeps nearest source pixel. Default: average."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
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
        "--threads",
        type=int,
        default=4,
        help="Number of GDAL warp threads. Default: 4.",
    )
    parser.add_argument(
        "--warp-mem-limit",
        type=int,
        default=512,
        help="GDAL warp memory limit in MB. Default: 512.",
    )
    parser.add_argument(
        "--src-nodata",
        type=float,
        default=DEFAULT_SRC_NODATA,
        help=(
            "Source nodata value passed to GDAL warp. Default: 255, so the "
            "source mosaic's metadata nodata=0 is ignored and 0 is kept as "
            "valid background for the uint8 0/1 raster."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned output geometry and exit without writing.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path, method: str, resolution: float) -> Path:
    res_label = f"{int(resolution)}m" if resolution.is_integer() else f"{resolution:g}m"
    return input_path.with_name(f"{input_path.stem}_albers_{res_label}_{method}.tif")


def output_dtype(method: str) -> str:
    if method == "average":
        return "float32"
    return "uint8"


def output_nodata(method: str) -> float | None:
    if method == "average":
        return -9999.0
    return None


def build_profile(
    src: rasterio.io.DatasetReader,
    dst_crs: CRS,
    transform,
    width: int,
    height: int,
    method: str,
    block_size: int,
    compress: str,
    zlevel: int,
) -> dict:
    dtype = output_dtype(method)
    nodata = output_nodata(method)

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        crs=dst_crs,
        transform=transform,
        width=width,
        height=height,
        count=1,
        dtype=dtype,
        nodata=nodata,
        tiled=True,
        blockxsize=block_size,
        blockysize=block_size,
        compress=compress,
        BIGTIFF="IF_SAFER",
        NUM_THREADS="ALL_CPUS",
    )
    if compress.lower() == "deflate":
        profile["zlevel"] = zlevel
        if dtype.startswith("float"):
            profile["predictor"] = 3
    return profile


def print_summary(
    input_path: Path,
    output_path: Path,
    src: rasterio.io.DatasetReader,
    dst_crs: CRS,
    transform,
    width: int,
    height: int,
    method: str,
    resolution: float,
    src_nodata: float,
) -> None:
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Source CRS: {src.crs}")
    print(f"Target CRS: {dst_crs.to_string()}")
    print(f"Target resolution: {resolution:g} m")
    print(f"Output size: {width} x {height} pixels")
    print(f"Output dtype: {output_dtype(method)}")
    print(f"Output nodata: {output_nodata(method)}")
    print(f"Source nodata used for warp: {src_nodata}")
    print(f"Resampling method: {method}")
    print(f"Output transform: {transform}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(input_path, args.method, args.resolution).resolve()
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster does not exist: {input_path}")
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")

    dst_crs = CRS.from_user_input(args.dst_crs)
    resampling = getattr(Resampling, args.method)

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(args.resolution, args.resolution),
        )
        print_summary(
            input_path,
            output_path,
            src,
            dst_crs,
            transform,
            width,
            height,
            args.method,
            args.resolution,
            args.src_nodata,
        )

        if args.dry_run:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = build_profile(
            src,
            dst_crs,
            transform,
            width,
            height,
            args.method,
            args.block_size,
            args.compress,
            args.zlevel,
        )

        dst_nodata = output_nodata(args.method)
        destination = np.empty((height, width), dtype=output_dtype(args.method))
        if dst_nodata is None:
            destination.fill(0)
        else:
            destination.fill(dst_nodata)

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=args.src_nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
            num_threads=args.threads,
            warp_mem_limit=args.warp_mem_limit,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(destination, 1)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RasterioIOError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
