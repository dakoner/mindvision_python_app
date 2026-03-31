from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile


DEFAULT_PIXELS_PER_MM = 2067.0


@dataclass(frozen=True)
class FramePlacement:
    frame_index: int
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class MosaicWrite:
    dst_x0: int
    dst_y0: int
    dst_x1: int
    dst_y1: int
    patch: np.ndarray


class ProgressTracker:
    def __init__(
        self,
        *,
        total_target_frames: int,
        total_placements: int,
        video_frame_count: int,
        report_interval_s: float = 1.0,
    ) -> None:
        self.total_target_frames = max(1, total_target_frames)
        self.total_placements = max(1, total_placements)
        self.video_frame_count = max(0, video_frame_count)
        self.report_interval_s = report_interval_s
        self.start_time = time.monotonic()
        self.last_report_time = 0.0

    def report(
        self,
        *,
        current_video_frame: int,
        processed_target_frames: int,
        frames_written: int,
        queued_tasks: int,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and (now - self.last_report_time) < self.report_interval_s:
            return

        elapsed_s = now - self.start_time
        target_pct = (processed_target_frames / self.total_target_frames) * 100.0
        placement_pct = (frames_written / self.total_placements) * 100.0
        video_pct = 0.0
        if self.video_frame_count > 0:
            video_pct = (min(current_video_frame, self.video_frame_count) / self.video_frame_count) * 100.0

        rate = 0.0
        if elapsed_s > 0:
            rate = processed_target_frames / elapsed_s

        print(
            "  progress: "
            f"targets {processed_target_frames}/{self.total_target_frames} ({target_pct:.1f}%), "
            f"placements {frames_written}/{self.total_placements} ({placement_pct:.1f}%), "
            f"video {current_video_frame}/{self.video_frame_count} ({video_pct:.1f}%), "
            f"queued {queued_tasks}, "
            f"rate {rate:.2f} target frames/s, "
            f"elapsed {elapsed_s:.1f}s"
        )
        self.last_report_time = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a large stage-space virtual image from a recorded MKV and its "
            "matching CSV metadata."
        )
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "videos",
        help="Directory containing recording_*.mkv and recording_*_meta.csv files.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Explicit MKV file to process. If omitted, matching files are discovered in videos-dir.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Explicit metadata CSV to process. If omitted, matching files are discovered in videos-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output TIFF path. If omitted, writes <recording>_virtual_mosaic.tif next to the inputs.",
    )
    parser.add_argument(
        "--pixels-per-mm",
        type=float,
        default=DEFAULT_PIXELS_PER_MM,
        help="Calibration used to convert stage millimeters to image pixels.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of CSV rows to place, useful for quick tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Number of worker threads used to prepare frame writes.",
    )
    return parser.parse_args()


def discover_pairs(videos_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for csv_path in sorted(videos_dir.glob("*_meta.csv")):
        stem = csv_path.name.removesuffix("_meta.csv")
        video_path = videos_dir / f"{stem}.mkv"
        if video_path.exists():
            pairs.append((video_path, csv_path))
    return pairs


def resolve_pairs(args: argparse.Namespace) -> list[tuple[Path, Path, Path]]:
    if args.video and args.csv:
        output_path = args.output or args.video.with_name(f"{args.video.stem}_virtual_mosaic.tif")
        return [(args.video.resolve(), args.csv.resolve(), output_path.resolve())]

    if args.video or args.csv:
        raise SystemExit("Provide both --video and --csv together, or neither.")

    pairs = discover_pairs(args.videos_dir.resolve())
    if not pairs:
        raise SystemExit(f"No recording pairs found in {args.videos_dir}")

    resolved: list[tuple[Path, Path, Path]] = []
    for video_path, csv_path in pairs:
        output_path = video_path.with_name(f"{video_path.stem}_virtual_mosaic.tif")
        resolved.append((video_path.resolve(), csv_path.resolve(), output_path.resolve()))
    return resolved


def load_metadata(csv_path: Path, max_frames: int | None) -> list[FramePlacement]:
    placements: list[FramePlacement] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            placements.append(
                FramePlacement(
                    frame_index=int(row["frame_index"]),
                    x_mm=float(row["x"]),
                    y_mm=float(row["y"]),
                    z_mm=float(row.get("z", 0.0)),
                )
            )
            if max_frames is not None and len(placements) >= max_frames:
                break

    if not placements:
        raise SystemExit(f"No frame rows found in {csv_path}")
    return placements


def inspect_video(video_path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise SystemExit(f"Could not read first frame from {video_path}")
        height, width = frame.shape[:2]

    capture.release()
    return width, height, frame_count


def compute_canvas_bounds(
    placements: list[FramePlacement],
    frame_width_px: int,
    frame_height_px: int,
    pixels_per_mm: float,
) -> tuple[int, int, int, int]:
    min_x_px = math.inf
    max_x_px = -math.inf
    min_y_px = math.inf
    max_y_px = -math.inf

    half_width = frame_width_px / 2.0
    half_height = frame_height_px / 2.0

    for placement in placements:
        center_x_px = placement.x_mm * pixels_per_mm
        center_y_px = placement.y_mm * pixels_per_mm

        min_x_px = min(min_x_px, math.floor(center_x_px - half_width))
        max_x_px = max(max_x_px, math.ceil(center_x_px + half_width))
        min_y_px = min(min_y_px, math.floor(center_y_px - half_height))
        max_y_px = max(max_y_px, math.ceil(center_y_px + half_height))

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


def prepare_frame_writes(
    frame_bgr: np.ndarray,
    placements: list[FramePlacement],
    pixels_per_mm: float,
    min_x_world_px: int,
    max_y_world_px: int,
    mosaic_width: int,
    mosaic_height: int,
) -> list[MosaicWrite]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.flip(frame_rgb, -1)

    frame_height, frame_width = frame_rgb.shape[:2]
    writes: list[MosaicWrite] = []

    for placement in placements:
        center_x_px = placement.x_mm * pixels_per_mm
        center_y_px = placement.y_mm * pixels_per_mm

        left_world_px = int(round(center_x_px - (frame_width / 2.0)))
        top_world_px = int(round(center_y_px + (frame_height / 2.0)))

        x0 = left_world_px - min_x_world_px
        y0 = max_y_world_px - top_world_px
        x1 = x0 + frame_width
        y1 = y0 + frame_height

        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(mosaic_width, x1)
        dst_y1 = min(mosaic_height, y1)

        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            continue

        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        writes.append(
            MosaicWrite(
                dst_x0=dst_x0,
                dst_y0=dst_y0,
                dst_x1=dst_x1,
                dst_y1=dst_y1,
                patch=frame_rgb[src_y0:src_y1, src_x0:src_x1].copy(),
            )
        )

    return writes


def apply_writes(mosaic: np.ndarray, writes: list[MosaicWrite]) -> int:
    for write in writes:
        mosaic[write.dst_y0:write.dst_y1, write.dst_x0:write.dst_x1] = write.patch
    return len(writes)


def write_manifest(
    manifest_path: Path,
    *,
    video_path: Path,
    csv_path: Path,
    output_path: Path,
    pixels_per_mm: float,
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
        "video": str(video_path),
        "csv": str(csv_path),
        "output": str(output_path),
        "pixels_per_mm": pixels_per_mm,
        "frame_size_px": [frame_width_px, frame_height_px],
        "canvas_size_px": [canvas_width_px, canvas_height_px],
        "world_bounds_px": {
            "min_x": min_x_world_px,
            "max_x": max_x_world_px,
            "min_y": min_y_world_px,
            "max_y": max_y_world_px,
        },
        "world_bounds_mm": {
            "min_x": min_x_world_px / pixels_per_mm,
            "max_x": max_x_world_px / pixels_per_mm,
            "min_y": min_y_world_px / pixels_per_mm,
            "max_y": max_y_world_px / pixels_per_mm,
        },
        "frame_rows": frame_rows,
        "frames_written": frames_written,
        "orientation": "Frames are flipped horizontally and vertically to match the stage mosaic view.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_virtual_mosaic(
    video_path: Path,
    csv_path: Path,
    output_path: Path,
    pixels_per_mm: float,
    max_frames: int | None,
    workers: int,
) -> None:
    placements = load_metadata(csv_path, max_frames)
    frame_width_px, frame_height_px, video_frame_count = inspect_video(video_path)

    min_x_world_px, max_x_world_px, min_y_world_px, max_y_world_px = compute_canvas_bounds(
        placements,
        frame_width_px,
        frame_height_px,
        pixels_per_mm,
    )
    canvas_width_px = max_x_world_px - min_x_world_px
    canvas_height_px = max_y_world_px - min_y_world_px

    if canvas_width_px <= 0 or canvas_height_px <= 0:
        raise SystemExit("Computed mosaic canvas has invalid dimensions.")

    print(f"Processing {video_path.name}")
    print(f"  metadata rows: {len(placements)}")
    print(f"  video frames: {video_frame_count}")
    print(f"  frame size: {frame_width_px}x{frame_height_px}")
    print(f"  mosaic size: {canvas_width_px}x{canvas_height_px}")
    print(f"  output: {output_path}")
    print(f"  workers: {workers}")

    mosaic = create_mosaic_file(output_path, canvas_height_px, canvas_width_px)
    mosaic[:] = 0

    placements_by_index: dict[int, list[FramePlacement]] = defaultdict(list)
    for placement in placements:
        placements_by_index[placement.frame_index].append(placement)

    target_indices = sorted(placements_by_index)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    target_cursor = 0
    apply_cursor = 0
    current_frame_index = 0
    frames_written = 0
    progress = ProgressTracker(
        total_target_frames=len(target_indices),
        total_placements=len(placements),
        video_frame_count=video_frame_count,
    )
    max_pending_tasks = max(1, workers * 2)
    pending_futures: dict[int, concurrent.futures.Future[list[MosaicWrite]]] = {}

    def drain_completed(force_wait: bool = False) -> int:
        nonlocal apply_cursor, frames_written

        while apply_cursor < len(target_indices):
            target_index = target_indices[apply_cursor]
            future = pending_futures.get(target_index)
            if future is None:
                break
            if not force_wait and not future.done():
                break

            writes = future.result()
            frames_written += apply_writes(mosaic, writes)
            del pending_futures[target_index]
            apply_cursor += 1
            progress.report(
                current_video_frame=current_frame_index,
                processed_target_frames=apply_cursor,
                frames_written=frames_written,
                queued_tasks=len(pending_futures),
            )

        return apply_cursor

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while target_cursor < len(target_indices):
            ok, frame = capture.read()
            if not ok:
                break

            target_index = target_indices[target_cursor]
            if current_frame_index < target_index:
                current_frame_index += 1
                progress.report(
                    current_video_frame=current_frame_index,
                    processed_target_frames=apply_cursor,
                    frames_written=frames_written,
                    queued_tasks=len(pending_futures),
                )
                continue

            if current_frame_index == target_index:
                pending_futures[target_index] = executor.submit(
                    prepare_frame_writes,
                    frame.copy(),
                    placements_by_index[target_index],
                    pixels_per_mm,
                    min_x_world_px,
                    max_y_world_px,
                    canvas_width_px,
                    canvas_height_px,
                )
                target_cursor += 1

                if len(pending_futures) >= max_pending_tasks:
                    drain_completed(force_wait=True)
                else:
                    drain_completed(force_wait=False)

            current_frame_index += 1
            progress.report(
                current_video_frame=current_frame_index,
                processed_target_frames=apply_cursor,
                frames_written=frames_written,
                queued_tasks=len(pending_futures),
            )

        capture.release()

        while pending_futures:
            drain_completed(force_wait=True)

    mosaic.flush()

    missing_targets = len(target_indices) - target_cursor
    if missing_targets:
        print(f"  warning: {missing_targets} frame indices were not found in the video")

    progress.report(
        current_video_frame=current_frame_index,
        processed_target_frames=apply_cursor,
        frames_written=frames_written,
        queued_tasks=0,
        force=True,
    )

    manifest_path = output_path.with_suffix(".json")
    write_manifest(
        manifest_path,
        video_path=video_path,
        csv_path=csv_path,
        output_path=output_path,
        pixels_per_mm=pixels_per_mm,
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
    print(f"  manifest: {manifest_path}")
    print(f"  frames written: {frames_written}")


def main() -> None:
    args = parse_args()
    pairs = resolve_pairs(args)
    for video_path, csv_path, output_path in pairs:
        build_virtual_mosaic(
            video_path=video_path,
            csv_path=csv_path,
            output_path=output_path,
            pixels_per_mm=args.pixels_per_mm,
            max_frames=args.max_frames,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()