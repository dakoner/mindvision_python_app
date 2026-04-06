from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
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
    mkv_path: Path
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


@dataclass(frozen=True)
class CompletedRowMosaic:
    row_recording: RowRecording
    manifest: dict


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
    parser.add_argument(
        "--skip-row-mosaics",
        action="store_true",
        help="Skip building per-row TIFF virtual mosaics.",
    )
    parser.add_argument(
        "--optimize-x",
        action="store_true",
        help="Optimize frame X placements using image similarity (NCC).",
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
            mkv_path = area_output_dir / f"{stem}.mkv"
            if not mkv_path.exists():
                print(f"warning: missing MKV file for {csv_path.name}, skipping")
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
                    mkv_path=mkv_path,
                    csv_path=csv_path,
                    output_tiff_path=output_tiff_path,
                    output_manifest_path=output_manifest_path,
                    area_index=area_index,
                    row_index=row_index,
                    area_output_dir=area_output_dir,
                )
            )

    return row_recordings


def create_mosaic_file(output_path: Path, height_px: int, width_px: int) -> np.ndarray:
    mosaic = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    tifffile.imwrite(str(output_path), mosaic)
    return mosaic


def composite_nonzero_rgb(
    mosaic: np.ndarray,
    frame_rgb: np.ndarray,
    *,
    min_x_world_px: float,
    max_y_world_px: float,
    x0: int,
    y0: int,
    mosaic_width: int,
    mosaic_height: int,
) -> None:
    h, w, _ = frame_rgb.shape
    x1, y1 = x0 + w, y0 + h
    
    # Clip to mosaic boundaries
    x0_clipped = max(0, x0)
    y0_clipped = max(0, y0)
    x1_clipped = min(mosaic_width, x1)
    y1_clipped = min(mosaic_height, y1)

    if x0_clipped < x1_clipped and y0_clipped < y1_clipped:
        # Calculate slice in frame_rgb
        frame_x0 = x0_clipped - x0
        frame_y0 = y0_clipped - y0
        frame_x1 = frame_x0 + (x1_clipped - x0_clipped)
        frame_y1 = frame_y0 + (y1_clipped - y0_clipped)
        
        mosaic[y0_clipped:y1_clipped, x0_clipped:x1_clipped] = frame_rgb[frame_y0:frame_y1, frame_x0:frame_x1]


def write_manifest(
    manifest_path: Path,
    *,
    scan_metadata_path: Path,
    area_index: int,
    row_index: int,
    mkv_path: Path,
    csv_path: Path,
    output_path: Path,
    frame_width_px: int,
    frame_height_px: int,
    canvas_width_px: int,
    canvas_height_px: int,
    min_x_world_px: float,
    max_x_world_px: float,
    min_y_world_px: float,
    max_y_world_px: float,
    frame_rows: int,
    frames_written: int,
) -> None:
    manifest = {
        "scan_metadata_path": str(scan_metadata_path),
        "area_index": area_index,
        "row_index": row_index,
        "mkv_path": str(mkv_path),
        "csv_path": str(csv_path),
        "output_path": str(output_path),
        "frame_width_px": frame_width_px,
        "frame_height_px": frame_height_px,
        "canvas_width_px": canvas_width_px,
        "canvas_height_px": canvas_height_px,
        "world_bounds_px": {
            "min_x": min_x_world_px,
            "max_x": max_x_world_px,
            "min_y": min_y_world_px,
            "max_y": max_y_world_px,
        },
        "frame_rows": frame_rows,
        "frames_written": frames_written,
    }
    manifest_path.write_text(json.dumps(manifest, indent=4))


def write_area_manifest(
    output_path: Path,
    *,
    scan_metadata_path: Path,
    area_index: int,
    output_path_tiff: Path,
    canvas_width_px: int,
    canvas_height_px: int,
    min_x_world_px: float,
    max_x_world_px: float,
    min_y_world_px: float,
    max_y_world_px: float,
    row_outputs: list[str],
) -> None:
    manifest = {
        "scan_metadata_path": str(scan_metadata_path),
        "area_index": area_index,
        "output_path": str(output_path_tiff),
        "canvas_width_px": canvas_width_px,
        "canvas_height_px": canvas_height_px,
        "world_bounds_px": {
            "min_x": min_x_world_px,
            "max_x": max_x_world_px,
            "min_y": min_y_world_px,
            "max_y": max_y_world_px,
        },
        "row_outputs": row_outputs,
    }
    output_path.write_text(json.dumps(manifest, indent=4))


def optimize_placement(frame1: np.ndarray, frame2: np.ndarray, dx_est: float, dy_est: float) -> tuple[float, float]:
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    H, W = gray1.shape
    
    shift_x = int(round(dx_est))
    shift_y = int(round(dy_est))
    
    x_min = max(0, shift_x)
    x_max = min(W, W + shift_x)
    y_min = max(0, shift_y)
    y_max = min(H, H + shift_y)
    
    if x_max - x_min < 50 or y_max - y_min < 50:
        return float(dx_est), float(dy_est)
        
    tw, th = int((x_max - x_min)*0.5), int((y_max - y_min)*0.5)
    cx, cy = (x_min + x_max)//2, (y_min + y_max)//2
    
    tx0 = cx - tw//2
    ty0 = cy - th//2
    tx1 = tx0 + tw
    ty1 = ty0 + th
    
    template = gray1[ty0:ty1, tx0:tx1]
    
    search_cx = cx - shift_x
    search_cy = cy - shift_y
    
    sw, sh = tw + 60, th + 60
    sx0 = max(0, search_cx - sw//2)
    sy0 = max(0, search_cy - sh//2)
    sx1 = min(W, search_cx + sw//2)
    sy1 = min(H, search_cy + sh//2)
    
    search_region = gray2[sy0:sy1, sx0:sx1]
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return float(dx_est), float(dy_est)
        
    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.3:
        return float(dx_est), float(dy_est)
        
    match_x2 = sx0 + max_loc[0]
    match_y2 = sy0 + max_loc[1]
    
    opt_dx = tx0 - match_x2
    opt_dy = ty0 - match_y2
    
    return float(opt_dx), float(opt_dy)


def build_row_virtual_mosaic(
    row_recording: RowRecording,
    *,
    scan_metadata_path: Path,
    scan_metadata: dict,
    max_frames: int | None,
    optimize_x: bool = False,
) -> None:
    pixel_scale_px_per_mm = scan_metadata.get("pixel_scale_px_per_mm", 1.0)
    placements = load_row_metadata(
        row_recording.csv_path,
        max_frames=max_frames,
        pixel_scale_px_per_mm=pixel_scale_px_per_mm,
    )

    mkv_path = row_recording.mkv_path

    if not placements:        return

    cap = cv2.VideoCapture(str(mkv_path))
    if not cap.isOpened():
        print(f"warning: failed to open video file {mkv_path}")
        return

    frame_width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads in BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    N = len(placements)
    if len(frames) < N:
        print(f"warning: video {mkv_path.name} has fewer frames ({len(frames)}) than placements ({N}).")
        N = min(N, len(frames))
        placements = placements[:N]

    if optimize_x and len(placements) > 1:
        optimized_placements = [placements[0]]
        for i in range(1, len(placements)):
            prev_p = optimized_placements[-1]
            curr_p = placements[i]
            
            dx_est = curr_p.pixel_x - placements[i-1].pixel_x
            dy_est = curr_p.pixel_y - placements[i-1].pixel_y
            
            opt_dx, opt_dy = optimize_placement(frames[i-1], frames[i], dx_est, dy_est)
            
            new_p = FramePlacement(
                frame_index=curr_p.frame_index,
                x_mm=curr_p.x_mm,
                y_mm=curr_p.y_mm,
                z_mm=curr_p.z_mm,
                pixel_x=prev_p.pixel_x + opt_dx,
                pixel_y=prev_p.pixel_y + opt_dy,
            )
            optimized_placements.append(new_p)
        placements = optimized_placements

    min_x_world_px = min(p.pixel_x for p in placements)
    max_x_world_px = max(p.pixel_x for p in placements)
    min_y_world_px = min(p.pixel_y for p in placements)
    max_y_world_px = max(p.pixel_y for p in placements)
    
    canvas_width_px = int(max_x_world_px - min_x_world_px) + frame_width_px
    canvas_height_px = int(max_y_world_px - min_y_world_px) + frame_height_px
    
    mosaic = np.zeros((canvas_height_px, canvas_width_px, 3), dtype=np.uint8)

    frames_written = 0
    for i, placement in enumerate(placements):
        frame_rgb = frames[i]
        
        x0 = int(placement.pixel_x - min_x_world_px)
        y0 = int(max_y_world_px - placement.pixel_y)
        
        composite_nonzero_rgb(
            mosaic,
            frame_rgb,
            min_x_world_px=min_x_world_px,
            max_y_world_px=max_y_world_px,
            x0=x0,
            y0=y0,
            mosaic_width=canvas_width_px,
            mosaic_height=canvas_height_px,
        )
        frames_written += 1

    tifffile.imwrite(str(row_recording.output_tiff_path), mosaic)

    write_manifest(
        row_recording.output_manifest_path,
        scan_metadata_path=scan_metadata_path,
        area_index=row_recording.area_index,
        row_index=row_recording.row_index,
        mkv_path=row_recording.mkv_path,
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
    completed_rows: list[CompletedRowMosaic] = []
    for row_recording in area_input.row_recordings:
        if not row_recording.output_manifest_path.exists() or not row_recording.output_tiff_path.exists():
            print(f"warning: missing row mosaic output, skipping area assembly input {row_recording.output_tiff_path}")
            continue
        completed_rows.append(
            CompletedRowMosaic(
                row_recording=row_recording,
                manifest=json.loads(row_recording.output_manifest_path.read_text()),
            )
        )

    if not completed_rows:
        print(f"warning: no completed row mosaics for area {area_input.area_index:03d}")
        return

    min_x_world_px = min(item.manifest["world_bounds_px"]["min_x"] for item in completed_rows)
    max_x_world_px = max(item.manifest["world_bounds_px"]["max_x"] for item in completed_rows)
    min_y_world_px = min(item.manifest["world_bounds_px"]["min_y"] for item in completed_rows)
    max_y_world_px = max(item.manifest["world_bounds_px"]["max_y"] for item in completed_rows)
    canvas_width_px = int(max_x_world_px - min_x_world_px)
    canvas_height_px = int(max_y_world_px - min_y_world_px)
    if canvas_width_px <= 0 or canvas_height_px <= 0:
        raise SystemExit(f"Computed invalid area canvas for area {area_input.area_index:03d}")

    output_tiff_path = area_input.area_output_dir / f"area_{area_input.area_index:03d}_virtual_mosaic.tif"
    output_manifest_path = area_input.area_output_dir / f"area_{area_input.area_index:03d}_virtual_mosaic.json"

    print(f"Assembling area {area_input.area_index:03d}")
    print(f"  row mosaics: {len(completed_rows)}")
    print(f"  output: {output_tiff_path}")
    print(f"  mosaic size: {canvas_width_px}x{canvas_height_px}")

    mosaic = create_mosaic_file(output_tiff_path, canvas_height_px, canvas_width_px)
    mosaic[:] = 0

    row_outputs: list[str] = []
    for completed_row in completed_rows:
        row_recording = completed_row.row_recording
        row_tiff = tifffile.imread(str(row_recording.output_tiff_path))
        row_bounds = completed_row.manifest["world_bounds_px"]
        x0 = int(row_bounds["min_x"] - min_x_world_px)
        y0 = int(max_y_world_px - row_bounds["max_y"])
        composite_nonzero_rgb(mosaic, row_tiff, x0=x0, y0=y0, 
                               min_x_world_px=min_x_world_px, 
                               max_y_world_px=max_y_world_px,
                               mosaic_width=canvas_width_px,
                               mosaic_height=canvas_height_px)
        row_outputs.append(str(row_recording.output_tiff_path))

    tifffile.imwrite(str(output_tiff_path), mosaic)

    write_area_manifest(        output_manifest_path,
        scan_metadata_path=scan_metadata_path,
        area_index=area_input.area_index,
        output_path_tiff=output_tiff_path,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
        min_x_world_px=min_x_world_px,
        max_x_world_px=max_x_world_px,
        min_y_world_px=min_y_world_px,
        max_y_world_px=max_y_world_px,
        row_outputs=row_outputs,
    )
    print(f"  manifest: {output_manifest_path}")


def process_scan_run(
    scan_metadata_path: Path, 
    max_frames: int | None,
    skip_row_mosaics: bool,
    optimize_x: bool
) -> None:
    scan_metadata = load_scan_metadata(scan_metadata_path)
    row_recordings = discover_row_recordings(scan_metadata, scan_metadata_path)
    if not row_recordings:
        print(f"warning: no row recordings found for {scan_metadata_path}")
        return

    print(f"Scan run: {scan_metadata_path.parent.name}")
    print(f"  row recordings: {len(row_recordings)}")
    
    if not skip_row_mosaics:
        for row_recording in row_recordings:
            build_row_virtual_mosaic(
                row_recording,
                scan_metadata_path=scan_metadata_path,
                scan_metadata=scan_metadata,
                max_frames=max_frames,
                optimize_x=optimize_x,
            )
    else:
        print("  skipping row mosaic building")

    for area_input in group_row_recordings_by_area(row_recordings):
        build_area_virtual_mosaic(
            area_input,
            scan_metadata_path=scan_metadata_path,
        )


def main() -> None:
    args = parse_args()
    scan_metadata_files = resolve_scan_metadata_files(args)
    for scan_metadata_path in scan_metadata_files:
        process_scan_run(
            scan_metadata_path, 
            args.max_frames,
            args.skip_row_mosaics,
            args.optimize_x,
        )


if __name__ == "__main__":
    main()
