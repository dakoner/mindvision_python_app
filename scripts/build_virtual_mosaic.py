from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class FramePlacement:
    frame_index: int
    x_mm: float
    y_mm: float
    z_mm: float
    pixel_x: float
    pixel_y: float


@dataclass(frozen=True)
class RowRecording:
    rgb_path: Path
    csv_path: Path
    output_tiff_path: Path
    output_manifest_path: Path
    area_index: int
    row_index: int
    area_output_dir: Path


@dataclass(frozen=True)
class AreaMosaicInput:
    area_index: int
    area_output_dir: Path
    row_recordings: list[RowRecording]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-row TIFF virtual mosaics from scan run directories created "
            "by the row-by-row scan recorder."
        )
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "videos",
        help="Directory containing scan_<timestamp>/ runs.",
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        help="Explicit scan run directory to process.",
    )
    parser.add_argument(
        "--scan-metadata",
        type=Path,
        help="Explicit scan_metadata.json to process.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of CSV rows to place per row TIFF.",
    )
    return parser.parse_args()


def discover_scan_metadata_files(videos_dir: Path) -> list[Path]:
    return sorted(videos_dir.glob("scan_*/scan_metadata.json"))


def resolve_scan_metadata_files(args: argparse.Namespace) -> list[Path]:
    if args.scan_metadata:
        return [args.scan_metadata.resolve()]

    if args.scan_dir:
        metadata_path = (args.scan_dir / "scan_metadata.json").resolve()
        if not metadata_path.exists():
            raise SystemExit(f"Missing scan metadata file: {metadata_path}")
        return [metadata_path]

    metadata_files = discover_scan_metadata_files(args.videos_dir.resolve())
    if not metadata_files:
        raise SystemExit(f"No scan metadata files found in {args.videos_dir}")
    return [path.resolve() for path in metadata_files]


def load_scan_metadata(metadata_path: Path) -> dict:
    try:
        return json.loads(metadata_path.read_text())
    except Exception as exc:
        raise SystemExit(f"Failed to read scan metadata {metadata_path}: {exc}") from exc


def load_row_metadata(
    csv_path: Path,
    *,
    max_frames: int | None,
    pixel_scale_px_per_mm: float,
) -> list[FramePlacement]:
    placements: list[FramePlacement] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_mm = float(row["x"])
            y_mm = float(row["y"])
            placements.append(
                FramePlacement(
                    frame_index=int(row["frame_index"]),
                    x_mm=x_mm,
                    y_mm=y_mm,
                    z_mm=float(row.get("z", 0.0)),
                    pixel_x=float(row.get("pixel_x", x_mm * pixel_scale_px_per_mm)),
                    pixel_y=float(row.get("pixel_y", y_mm * pixel_scale_px_per_mm)),
                )
            )
            if max_frames is not None and len(placements) >= max_frames:
                break

    if not placements:
        raise SystemExit(f"No frame rows found in {csv_path}")
    return placements


def discover_row_recordings(scan_metadata: dict, metadata_path: Path) -> list[RowRecording]:
    scan_root = metadata_path.parent
    row_recordings: list[RowRecording] = []
    scan_regions = scan_metadata.get("scan_regions_mm", [])

    for area_index, area_info in enumerate(scan_regions, start=1):
        area_output_dir = Path(area_info.get("output_dir") or (scan_root / f"area_{area_index:03d}"))
        if not area_output_dir.exists():
            print(f"warning: area directory missing, skipping: {area_output_dir}")
            continue

        csv_paths = sorted(area_output_dir.glob("*_meta.csv"))
        for csv_path in csv_paths:
            stem = csv_path.name.removesuffix("_meta.csv")
            rgb_path = area_output_dir / f"{stem}.rgb"
            if not rgb_path.exists():
                print(f"warning: missing RGB file for {csv_path.name}, skipping")
                continue

            row_suffix = stem.rsplit("_row_", 1)
            if len(row_suffix) != 2:
                print(f"warning: unexpected row recording name, skipping: {stem}")
                continue

            row_index = int(row_suffix[1])
            output_tiff_path = area_output_dir / f"{stem}_virtual_mosaic.tif"
            output_manifest_path = area_output_dir / f"{stem}_virtual_mosaic.json"
            row_recordings.append(
                RowRecording(
                    rgb_path=rgb_path,
                    csv_path=csv_path,
                    output_tiff_path=output_tiff_path,
                    output_manifest_path=output_manifest_path,
                    area_index=area_index,
                    row_index=row_index,
                    area_output_dir=area_output_dir,
                )
            )

    return row_recordings


def inspect_rgb(rgb_path: Path, frame_width_px: int, frame_height_px: int) -> tuple[int, int]:
    frame_bytes = frame_width_px * frame_height_px * 3
    file_size = rgb_path.stat().st_size
    if frame_bytes <= 0:
        raise SystemExit(f"Invalid frame dimensions for RGB input: {frame_width_px}x{frame_height_px}")
    if file_size % frame_bytes != 0:
        raise SystemExit(
            f"RGB file size is not divisible by frame size: {rgb_path} ({file_size} bytes)"
        )
    return frame_bytes, file_size // frame_bytes


def read_rgb_frame(
    handle,
    *,
    frame_index: int,
    frame_bytes: int,
    frame_width_px: int,
    frame_height_px: int,
) -> np.ndarray:
    handle.seek(frame_index * frame_bytes)
    frame_data = handle.read(frame_bytes)
    if len(frame_data) != frame_bytes:
        raise EOFError(f"Could not read frame {frame_index}")
    frame_rgb = np.frombuffer(frame_data, dtype=np.uint8).reshape((frame_height_px, frame_width_px, 3))
    return np.flip(frame_rgb, axis=(0, 1)).copy()


def compute_canvas_bounds(
    placements: list[FramePlacement],
    frame_width_px: int,
    frame_height_px: int,
) -> tuple[int, int, int, int]:
    min_x_px = math.inf
    max_x_px = -math.inf
    min_y_px = math.inf
    max_y_px = -math.inf

    half_width = frame_width_px / 2.0
    half_height = frame_height_px / 2.0

    for placement in placements:
        min_x_px = min(min_x_px, math.floor(placement.pixel_x - half_width))
        max_x_px = max(max_x_px, math.ceil(placement.pixel_x + half_width))
        min_y_px = min(min_y_px, math.floor(placement.pixel_y - half_height))
        max_y_px = max(max_y_px, math.ceil(placement.pixel_y + half_height))

    return int(min_x_px), int(max_x_px), int(min_y_px), int(max_y_px)


def create_mosaic_file(output_path: Path, height: int, width: int) -> np.memmap:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return tifffile.memmap(
        str(output_path),
        shape=(height, width, 3),
        dtype=np.uint8,
        photometric="rgb",
        bigtiff=True,
    )


def place_frame(
    mosaic: np.ndarray,
    frame_rgb: np.ndarray,
    placement: FramePlacement,
    *,
    min_x_world_px: int,
    max_y_world_px: int,
    mosaic_width: int,
    mosaic_height: int,
) -> None:
    frame_height, frame_width = frame_rgb.shape[:2]

    left_world_px = int(round(placement.pixel_x - (frame_width / 2.0)))
    top_world_px = int(round(placement.pixel_y + (frame_height / 2.0)))

    x0 = left_world_px - min_x_world_px
    y0 = max_y_world_px - top_world_px
    x1 = x0 + frame_width
    y1 = y0 + frame_height

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(mosaic_width, x1)
    dst_y1 = min(mosaic_height, y1)

    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    mosaic[dst_y0:dst_y1, dst_x0:dst_x1] = frame_rgb[src_y0:src_y1, src_x0:src_x1]


def write_manifest(
    manifest_path: Path,
    *,
    scan_metadata_path: Path,
    area_index: int,
    row_index: int,
    rgb_path: Path,
    csv_path: Path,
    output_path: Path,
    frame_width_px: int,
    frame_height_px: int,
    canvas_width_px: int,
    canvas_height_px: int,
    min_x_world_px: int,
    max_x_world_px: int,
    min_y_world_px: int,
    max_y_world_px: int,
    frame_rows: int,
    frames_written: int,
) -> None:
    manifest = {
        "scan_metadata": str(scan_metadata_path),
        "area_index": area_index,
        "row_index": row_index,
        "rgb": str(rgb_path),
        "csv": str(csv_path),
        "output": str(output_path),
        "frame_size_px": [frame_width_px, frame_height_px],
        "canvas_size_px": [canvas_width_px, canvas_height_px],
        "world_bounds_px": {
            "min_x": min_x_world_px,
            "max_x": max_x_world_px,
            "min_y": min_y_world_px,
            "max_y": max_y_world_px,
        },
        "frame_rows": frame_rows,
        "frames_written": frames_written,
        "orientation": "Frames are flipped horizontally and vertically to match the stage mosaic view.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def write_area_manifest(
    manifest_path: Path,
    *,
    scan_metadata_path: Path,
    area_index: int,
    output_path: Path,
    canvas_width_px: int,
    canvas_height_px: int,
    min_x_world_px: int,
    max_x_world_px: int,
    min_y_world_px: int,
    max_y_world_px: int,
    row_outputs: list[str],
) -> None:
    manifest = {
        "scan_metadata": str(scan_metadata_path),
        "area_index": area_index,
        "output": str(output_path),
        "canvas_size_px": [canvas_width_px, canvas_height_px],
        "world_bounds_px": {
            "min_x": min_x_world_px,
            "max_x": max_x_world_px,
            "min_y": min_y_world_px,
            "max_y": max_y_world_px,
        },
        "row_outputs": row_outputs,
        "orientation": "Area mosaic assembled from per-row TIFF mosaics.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_row_virtual_mosaic(
    row_recording: RowRecording,
    *,
    scan_metadata_path: Path,
    scan_metadata: dict,
    max_frames: int | None,
) -> None:
    frame_width_px = int(scan_metadata["image_dimensions_px"]["width"])
    frame_height_px = int(scan_metadata["image_dimensions_px"]["height"])
    pixel_scale_px_per_mm = float(scan_metadata.get("pixel_scale_px_per_mm") or 0.0)
    placements = load_row_metadata(
        row_recording.csv_path,
        max_frames=max_frames,
        pixel_scale_px_per_mm=pixel_scale_px_per_mm,
    )

    frame_bytes, rgb_frame_count = inspect_rgb(
        row_recording.rgb_path,
        frame_width_px,
        frame_height_px,
    )

    min_x_world_px, max_x_world_px, min_y_world_px, max_y_world_px = compute_canvas_bounds(
        placements,
        frame_width_px,
        frame_height_px,
    )
    canvas_width_px = max_x_world_px - min_x_world_px
    canvas_height_px = max_y_world_px - min_y_world_px
    if canvas_width_px <= 0 or canvas_height_px <= 0:
        raise SystemExit(f"Computed invalid canvas for {row_recording.csv_path}")

    print(f"Processing area {row_recording.area_index:03d} row {row_recording.row_index:03d}")
    print(f"  rgb: {row_recording.rgb_path.name}")
    print(f"  csv rows: {len(placements)}")
    print(f"  rgb frames: {rgb_frame_count}")
    print(f"  frame size: {frame_width_px}x{frame_height_px}")
    print(f"  mosaic size: {canvas_width_px}x{canvas_height_px}")
    print(f"  output: {row_recording.output_tiff_path}")

    mosaic = create_mosaic_file(row_recording.output_tiff_path, canvas_height_px, canvas_width_px)
    mosaic[:] = 0

    frames_written = 0
    with row_recording.rgb_path.open("rb") as handle:
        for placement in placements:
            if placement.frame_index >= rgb_frame_count:
                print(
                    f"  warning: frame index {placement.frame_index} exceeds RGB frame count {rgb_frame_count}"
                )
                continue

            frame_rgb = read_rgb_frame(
                handle,
                frame_index=placement.frame_index,
                frame_bytes=frame_bytes,
                frame_width_px=frame_width_px,
                frame_height_px=frame_height_px,
            )
            place_frame(
                mosaic,
                frame_rgb,
                placement,
                min_x_world_px=min_x_world_px,
                max_y_world_px=max_y_world_px,
                mosaic_width=canvas_width_px,
                mosaic_height=canvas_height_px,
            )
            frames_written += 1

    mosaic.flush()

    write_manifest(
        row_recording.output_manifest_path,
        scan_metadata_path=scan_metadata_path,
        area_index=row_recording.area_index,
        row_index=row_recording.row_index,
        rgb_path=row_recording.rgb_path,
        csv_path=row_recording.csv_path,
        output_path=row_recording.output_tiff_path,
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
        min_x_world_px=min_x_world_px,
        max_x_world_px=max_x_world_px,
        min_y_world_px=min_y_world_px,
        max_y_world_px=max_y_world_px,
        frame_rows=len(placements),
        frames_written=frames_written,
    )
    print(f"  manifest: {row_recording.output_manifest_path}")
    print(f"  frames written: {frames_written}")


def group_row_recordings_by_area(row_recordings: list[RowRecording]) -> list[AreaMosaicInput]:
    grouped: dict[int, list[RowRecording]] = {}
    for row_recording in row_recordings:
        grouped.setdefault(row_recording.area_index, []).append(row_recording)

    area_inputs: list[AreaMosaicInput] = []
    for area_index in sorted(grouped):
        area_rows = sorted(grouped[area_index], key=lambda item: item.row_index)
        area_inputs.append(
            AreaMosaicInput(
                area_index=area_index,
                area_output_dir=area_rows[0].area_output_dir,
                row_recordings=area_rows,
            )
        )
    return area_inputs


def build_area_virtual_mosaic(
    area_input: AreaMosaicInput,
    *,
    scan_metadata_path: Path,
) -> None:
    row_manifests: list[dict] = []
    for row_recording in area_input.row_recordings:
        if not row_recording.output_manifest_path.exists() or not row_recording.output_tiff_path.exists():
            print(f"warning: missing row mosaic output, skipping area assembly input {row_recording.output_tiff_path}")
            continue
        row_manifests.append(json.loads(row_recording.output_manifest_path.read_text()))

    if not row_manifests:
        print(f"warning: no completed row mosaics for area {area_input.area_index:03d}")
        return

    min_x_world_px = min(manifest["world_bounds_px"]["min_x"] for manifest in row_manifests)
    max_x_world_px = max(manifest["world_bounds_px"]["max_x"] for manifest in row_manifests)
    min_y_world_px = min(manifest["world_bounds_px"]["min_y"] for manifest in row_manifests)
    max_y_world_px = max(manifest["world_bounds_px"]["max_y"] for manifest in row_manifests)
    canvas_width_px = max_x_world_px - min_x_world_px
    canvas_height_px = max_y_world_px - min_y_world_px
    if canvas_width_px <= 0 or canvas_height_px <= 0:
        raise SystemExit(f"Computed invalid area canvas for area {area_input.area_index:03d}")

    output_tiff_path = area_input.area_output_dir / f"area_{area_input.area_index:03d}_virtual_mosaic.tif"
    output_manifest_path = area_input.area_output_dir / f"area_{area_input.area_index:03d}_virtual_mosaic.json"

    print(f"Assembling area {area_input.area_index:03d}")
    print(f"  row mosaics: {len(row_manifests)}")
    print(f"  output: {output_tiff_path}")
    print(f"  mosaic size: {canvas_width_px}x{canvas_height_px}")

    mosaic = create_mosaic_file(output_tiff_path, canvas_height_px, canvas_width_px)
    mosaic[:] = 0

    row_outputs: list[str] = []
    for row_recording, row_manifest in zip(area_input.row_recordings, row_manifests):
        row_tiff = tifffile.imread(str(row_recording.output_tiff_path))
        row_bounds = row_manifest["world_bounds_px"]
        x0 = int(row_bounds["min_x"] - min_x_world_px)
        y0 = int(max_y_world_px - row_bounds["max_y"])
        y1 = y0 + row_tiff.shape[0]
        x1 = x0 + row_tiff.shape[1]
        mosaic[y0:y1, x0:x1] = row_tiff
        row_outputs.append(str(row_recording.output_tiff_path))

    mosaic.flush()

    write_area_manifest(
        output_manifest_path,
        scan_metadata_path=scan_metadata_path,
        area_index=area_input.area_index,
        output_path=output_tiff_path,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
        min_x_world_px=min_x_world_px,
        max_x_world_px=max_x_world_px,
        min_y_world_px=min_y_world_px,
        max_y_world_px=max_y_world_px,
        row_outputs=row_outputs,
    )
    print(f"  manifest: {output_manifest_path}")


def process_scan_run(scan_metadata_path: Path, max_frames: int | None) -> None:
    scan_metadata = load_scan_metadata(scan_metadata_path)
    row_recordings = discover_row_recordings(scan_metadata, scan_metadata_path)
    if not row_recordings:
        print(f"warning: no row recordings found for {scan_metadata_path}")
        return

    print(f"Scan run: {scan_metadata_path.parent.name}")
    print(f"  row recordings: {len(row_recordings)}")
    for row_recording in row_recordings:
        build_row_virtual_mosaic(
            row_recording,
            scan_metadata_path=scan_metadata_path,
            scan_metadata=scan_metadata,
            max_frames=max_frames,
        )

    for area_input in group_row_recordings_by_area(row_recordings):
        build_area_virtual_mosaic(
            area_input,
            scan_metadata_path=scan_metadata_path,
        )


def main() -> None:
    args = parse_args()
    scan_metadata_files = resolve_scan_metadata_files(args)
    for scan_metadata_path in scan_metadata_files:
        process_scan_run(scan_metadata_path, args.max_frames)


if __name__ == "__main__":
    main()