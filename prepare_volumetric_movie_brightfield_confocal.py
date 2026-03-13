"""Render synchronized 1x4 brightfield/confocal z-stack movies from Imaris files.

Expected channel names by default:

1. White Light - Brightfield
2. Confocal - Green
3. Confocal - Red

For each ``.ims`` file, this script renders an MP4 movie across a selected z
window with 4 synchronized panels:

1. Brightfield
2. Green
3. Red
4. Merged red+green

The green and red panels can include labeled scale bars. Per-run and per-output
metadata are written as JSON sidecars plus a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import h5py  # type: ignore
import imageio.v2 as imageio  # type: ignore
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imaris_tools.metadata import _discover_channels, read_metadata

DEFAULT_BRIGHTFIELD_CHANNEL_NAME = "White Light - Brightfield"
DEFAULT_GREEN_CHANNEL_NAME = "Confocal - Green"
DEFAULT_RED_CHANNEL_NAME = "Confocal - Red"


@dataclass(frozen=True)
class PreparedMovieRecord:
    source_path: Path
    output_path: Path
    output_metadata_path: Path
    brightfield_channel_index: int
    green_channel_index: int
    red_channel_index: int
    z_start_requested: int
    z_end_requested: int
    z_start_effective: int
    z_end_effective: int
    roi_enabled: bool
    roi_center_x_requested: int
    roi_center_y_requested: int
    roi_size_requested: int
    roi_x_start: int
    roi_x_end: int
    roi_y_start: int
    roi_y_end: int
    frame_count: int
    fps: float
    brightfield_min_value: float
    brightfield_max_value: float
    green_min_value: float
    green_max_value: float
    red_min_value: float
    red_max_value: float


class BrightfieldConfocalMoviePreparer:
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        recursive: bool = False,
        overwrite: bool = False,
        resolution_level: int = 0,
        time_point: int = 0,
        z_start: int = 90,
        z_end: int = 672,
        brightfield_channel_name: str = DEFAULT_BRIGHTFIELD_CHANNEL_NAME,
        green_channel_name: str = DEFAULT_GREEN_CHANNEL_NAME,
        red_channel_name: str = DEFAULT_RED_CHANNEL_NAME,
        background_subtract_sigma: float = 0.0,
        median_filter_size: int = 0,
        preserve_edge_pixels: int = 0,
        invert_display: bool = False,
        scale_mode: str = "percentile",
        display_gamma: float = 1.0,
        scale_low_percentile: float = 0.0,
        scale_high_percentile: float = 99.5,
        include_scale_bars: bool = True,
        scale_bar_width: int = 14,
        fps: float = 12.0,
        progress_interval: int = 25,
        scale_estimation_z_step: int = 5,
        scale_estimation_xy_step: int = 16,
        roi_center_x: Optional[int] = None,
        roi_center_y: Optional[int] = None,
        roi_size: Optional[int] = None,
        cli_args: Optional[list[str]] = None,
    ) -> None:
        self.input_path = input_path.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.recursive = recursive
        self.overwrite = overwrite
        self.resolution_level = resolution_level
        self.time_point = time_point
        self.z_start = int(z_start)
        self.z_end = int(z_end)
        self.brightfield_channel_name = brightfield_channel_name.strip()
        self.green_channel_name = green_channel_name.strip()
        self.red_channel_name = red_channel_name.strip()
        self.background_subtract_sigma = max(0.0, float(background_subtract_sigma))
        self.median_filter_size = max(0, int(median_filter_size))
        self.preserve_edge_pixels = max(0, int(preserve_edge_pixels))
        self.invert_display = invert_display
        self.scale_mode = scale_mode
        self.display_gamma = max(1e-6, float(display_gamma))
        self.scale_low_percentile = float(scale_low_percentile)
        self.scale_high_percentile = float(scale_high_percentile)
        self.include_scale_bars = include_scale_bars
        self.scale_bar_width = max(2, int(scale_bar_width))
        self.fps = float(fps)
        self.progress_interval = max(1, int(progress_interval))
        self.scale_estimation_z_step = max(1, int(scale_estimation_z_step))
        self.scale_estimation_xy_step = max(1, int(scale_estimation_xy_step))
        self.roi_center_x = None if roi_center_x is None else int(roi_center_x)
        self.roi_center_y = None if roi_center_y is None else int(roi_center_y)
        self.roi_size = None if roi_size is None else int(roi_size)
        self.roi_enabled = any(value is not None for value in (self.roi_center_x, self.roi_center_y, self.roi_size))
        if self.roi_enabled and any(value is None for value in (self.roi_center_x, self.roi_center_y, self.roi_size)):
            raise ValueError("Square ROI view requires --roi-center-x, --roi-center-y, and --roi-size together.")
        if self.roi_size is not None and self.roi_size <= 0:
            raise ValueError("--roi-size must be a positive integer.")
        self.cli_args = list(cli_args or [])

    def discover_files(self) -> list[Path]:
        if self.input_path.is_file():
            if self.input_path.suffix.lower() != ".ims":
                raise ValueError(f"Input file must have .ims extension: {self.input_path}")
            return [self.input_path]

        iterator: Iterable[Path]
        if self.recursive:
            iterator = self.input_path.rglob("*.ims")
        else:
            iterator = self.input_path.glob("*.ims")
        return sorted(path for path in iterator if path.is_file())

    def prepare_all(self) -> list[PreparedMovieRecord]:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        if not self.input_path.is_dir() and not self.input_path.is_file():
            raise FileNotFoundError(f"Input path is neither a directory nor a file: {self.input_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ims_files = self.discover_files()
        records: list[PreparedMovieRecord] = []
        run_metadata_path = self.output_dir / "preparation_run_metadata.json"
        self._write_json(run_metadata_path, self._build_run_metadata(ims_files))

        self._print_run_header(ims_files, run_metadata_path)

        for file_index, ims_path in enumerate(ims_files, start=1):
            print(f"\n[info] Processing movie {file_index}/{len(ims_files)}: {ims_path.name}")
            print(f"[step] Reading metadata: {ims_path}")
            metadata = read_metadata(
                ims_path,
                resolution_level=self.resolution_level,
                time_point=self.time_point,
            )
            print(f"[step] Metadata loaded. Channels reported: {len(getattr(metadata, 'channels', []))}")
            channel_map = self._resolve_required_channels(metadata)
            if channel_map is None:
                print(
                    "[warn] "
                    f"Skipping {ims_path.name}: required channels not found "
                    f"('{self.brightfield_channel_name}', '{self.green_channel_name}', '{self.red_channel_name}')."
                )
                continue
            print(
                "[step] Resolved channels: "
                f"brightfield={channel_map['brightfield'][0]} ({channel_map['brightfield'][1]}), "
                f"green={channel_map['green'][0]} ({channel_map['green'][1]}), "
                f"red={channel_map['red'][0]} ({channel_map['red'][1]})"
            )

            with h5py.File(ims_path, "r") as handle:
                print("[step] Discovering channel datasets in file")
                datasets = dict(
                    _discover_channels(
                        handle,
                        resolution_level=self.resolution_level,
                        time_point=self.time_point,
                    )
                )
                print(f"[step] Dataset discovery complete. Found {len(datasets)} channel datasets.")
                bf_dataset = datasets[channel_map["brightfield"][0]]
                green_dataset = datasets[channel_map["green"][0]]
                red_dataset = datasets[channel_map["red"][0]]
                for dataset_index, dataset in sorted(datasets.items()):
                    print(f"[step] Channel dataset {dataset_index}: shape={tuple(int(dim) for dim in dataset.shape)}")

                z_window = self._effective_z_window(green_dataset)
                print(
                    f"[step] Effective movie z window={z_window['z_start']}..{z_window['z_end']} "
                    f"({z_window['z_count']} slices)"
                )
                roi_window = self._resolve_roi_window(tuple(int(dim) for dim in bf_dataset.shape[-2:]))
                self._print_roi_summary(roi_window)

                print("[step] Estimating display ranges across z window")
                brightfield_scale = self._estimate_display_range(
                    bf_dataset,
                    z_window=z_window,
                    channel_label=f"brightfield ch{channel_map['brightfield'][0]}",
                    preprocess=False,
                    roi_window=roi_window,
                )
                green_scale = self._estimate_display_range(
                    green_dataset,
                    z_window=z_window,
                    channel_label=f"green ch{channel_map['green'][0]}",
                    preprocess=True,
                    roi_window=roi_window,
                )
                red_scale = self._estimate_display_range(
                    red_dataset,
                    z_window=z_window,
                    channel_label=f"red ch{channel_map['red'][0]}",
                    preprocess=True,
                    roi_window=roi_window,
                )

                self._print_scale_summary("brightfield", brightfield_scale)
                self._print_scale_summary("green", green_scale)
                self._print_scale_summary("red", red_scale)

                output_path = self.output_dir / self._build_output_name(ims_path, z_window, roi_window)
                output_metadata_path = output_path.with_suffix(".metadata.json")
                if output_path.exists() and not self.overwrite:
                    print(f"[warn] Output exists, skipping (use --overwrite to replace): {output_path}")
                    continue

                writer = self._open_mp4_writer(output_path)
                try:
                    for frame_offset, z_index in enumerate(range(z_window["z_start"], z_window["z_end"] + 1), start=1):
                        if frame_offset == 1 or frame_offset % self.progress_interval == 0 or z_index == z_window["z_end"]:
                            print(
                                f"[progress] rendering frame {frame_offset}/{z_window['z_count']} "
                                f"(z={z_index})"
                            )
                        brightfield = self._crop_to_roi(np.asarray(bf_dataset[z_index, ...], dtype=np.float32), roi_window)
                        green = self._crop_to_roi(np.asarray(green_dataset[z_index, ...], dtype=np.float32), roi_window)
                        red = self._crop_to_roi(np.asarray(red_dataset[z_index, ...], dtype=np.float32), roi_window)

                        green = self._preprocess_fluorescence_frame(green, channel_label="green frame")
                        red = self._preprocess_fluorescence_frame(red, channel_label="red frame")

                        frame = self._compose_frame(
                            brightfield=brightfield,
                            green=green,
                            red=red,
                            brightfield_scale=brightfield_scale,
                            green_scale=green_scale,
                            red_scale=red_scale,
                            z_index=z_index,
                            z_window=z_window,
                        )
                        writer.append_data(frame)
                finally:
                    writer.close()

            self._write_json(
                output_metadata_path,
                self._build_movie_metadata(
                    ims_path=ims_path,
                    output_path=output_path,
                    output_metadata_path=output_metadata_path,
                    run_metadata_path=run_metadata_path,
                    channel_map=channel_map,
                    z_window=z_window,
                    roi_window=roi_window,
                    brightfield_scale=brightfield_scale,
                    green_scale=green_scale,
                    red_scale=red_scale,
                ),
            )
            records.append(
                PreparedMovieRecord(
                    source_path=ims_path,
                    output_path=output_path,
                    output_metadata_path=output_metadata_path,
                    brightfield_channel_index=channel_map["brightfield"][0],
                    green_channel_index=channel_map["green"][0],
                    red_channel_index=channel_map["red"][0],
                    z_start_requested=self.z_start,
                    z_end_requested=self.z_end,
                    z_start_effective=z_window["z_start"],
                    z_end_effective=z_window["z_end"],
                    roi_enabled=bool(roi_window["enabled"]),
                    roi_center_x_requested=int(roi_window["requested_center_x"]),
                    roi_center_y_requested=int(roi_window["requested_center_y"]),
                    roi_size_requested=int(roi_window["requested_size"]),
                    roi_x_start=int(roi_window["x_start"]),
                    roi_x_end=int(roi_window["x_end"]),
                    roi_y_start=int(roi_window["y_start"]),
                    roi_y_end=int(roi_window["y_end"]),
                    frame_count=z_window["z_count"],
                    fps=self.fps,
                    brightfield_min_value=brightfield_scale[0],
                    brightfield_max_value=brightfield_scale[1],
                    green_min_value=green_scale[0],
                    green_max_value=green_scale[1],
                    red_min_value=red_scale[0],
                    red_max_value=red_scale[1],
                )
            )
            print(f"[ok] Wrote movie: {output_path}")
            print(f"[ok] Wrote sidecar metadata: {output_metadata_path}")

        manifest = self.output_dir / "prepared_manifest.csv"
        print(f"[step] Writing manifest CSV: {manifest}")
        self._write_manifest(records, manifest, run_metadata_path=run_metadata_path)
        print(f"[info] Wrote manifest with {len(records)} record(s): {manifest}")
        return records

    def _print_run_header(self, ims_files: list[Path], run_metadata_path: Path) -> None:
        print("[info] ================================================")
        print(f"[info] Input path: {self.input_path}")
        print(f"[info] Output directory: {self.output_dir}")
        print(f"[info] .ims files discovered: {len(ims_files)}")
        print(f"[info] Recursive search: {self.recursive}")
        print(f"[info] Resolution level: {self.resolution_level}, time point: {self.time_point}")
        print(f"[info] Requested z window: {self.z_start}..{self.z_end} (inclusive)")
        print(
            "[info] Channel names: "
            f"brightfield='{self.brightfield_channel_name}', "
            f"green='{self.green_channel_name}', "
            f"red='{self.red_channel_name}'"
        )
        print(
            "[info] Fluorescence preprocessing: "
            f"background sigma={self.background_subtract_sigma:.2f}, "
            f"median filter size={self.median_filter_size}, "
            f"preserve edge pixels={self.preserve_edge_pixels}"
        )
        print(
            "[info] Display scaling: "
            f"mode={self.scale_mode}, gamma={self.display_gamma:.3f}, "
            f"percentiles={self.scale_low_percentile:.2f}->{self.scale_high_percentile:.2f}"
        )
        print(f"[info] Invert display: {self.invert_display}")
        print(f"[info] Include green/red scale bars: {self.include_scale_bars}")
        if self.roi_enabled:
            print(
                "[info] Square ROI view: "
                f"center=({self.roi_center_x}, {self.roi_center_y}), size={self.roi_size}"
            )
        else:
            print("[info] Square ROI view: disabled (full field of view)")
        print(f"[info] Movie FPS: {self.fps}")
        print(f"[info] Render progress interval: every {self.progress_interval} frames")
        print(
            "[info] Scale estimation sampling: "
            f"every {self.scale_estimation_z_step} z slices, every {self.scale_estimation_xy_step} pixels in x/y"
        )
        print(f"[info] Run metadata JSON: {run_metadata_path}")
        print("[info] ================================================")

    def _settings_dict(self) -> dict[str, object]:
        return {
            "recursive": self.recursive,
            "overwrite": self.overwrite,
            "resolution_level": self.resolution_level,
            "time_point": self.time_point,
            "z_start": self.z_start,
            "z_end": self.z_end,
            "brightfield_channel_name": self.brightfield_channel_name,
            "green_channel_name": self.green_channel_name,
            "red_channel_name": self.red_channel_name,
            "background_subtract_sigma": self.background_subtract_sigma,
            "median_filter_size": self.median_filter_size,
            "preserve_edge_pixels": self.preserve_edge_pixels,
            "invert_display": self.invert_display,
            "scale_mode": self.scale_mode,
            "display_gamma": self.display_gamma,
            "scale_low_percentile": self.scale_low_percentile,
            "scale_high_percentile": self.scale_high_percentile,
            "include_scale_bars": self.include_scale_bars,
            "scale_bar_width": self.scale_bar_width,
            "fps": self.fps,
            "progress_interval": self.progress_interval,
            "scale_estimation_z_step": self.scale_estimation_z_step,
            "scale_estimation_xy_step": self.scale_estimation_xy_step,
            "roi_center_x": self.roi_center_x,
            "roi_center_y": self.roi_center_y,
            "roi_size": self.roi_size,
        }

    def _build_run_metadata(self, ims_files: list[Path]) -> dict[str, object]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable,
            "cli_args": self.cli_args,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "ims_files_discovered": [str(path) for path in ims_files],
            "settings": self._settings_dict(),
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_movie_metadata(
        self,
        *,
        ims_path: Path,
        output_path: Path,
        output_metadata_path: Path,
        run_metadata_path: Path,
        channel_map: dict[str, tuple[int, str]],
        z_window: dict[str, int],
        roi_window: dict[str, int | bool],
        brightfield_scale: tuple[float, float],
        green_scale: tuple[float, float],
        red_scale: tuple[float, float],
    ) -> dict[str, object]:
        return {
            "source_path": str(ims_path),
            "output_path": str(output_path),
            "output_metadata_path": str(output_metadata_path),
            "run_metadata_path": str(run_metadata_path),
            "channels": {
                "brightfield": {
                    "index": channel_map["brightfield"][0],
                    "name": channel_map["brightfield"][1],
                    "display_range": {"min": brightfield_scale[0], "max": brightfield_scale[1]},
                },
                "green": {
                    "index": channel_map["green"][0],
                    "name": channel_map["green"][1],
                    "display_range": {"min": green_scale[0], "max": green_scale[1]},
                },
                "red": {
                    "index": channel_map["red"][0],
                    "name": channel_map["red"][1],
                    "display_range": {"min": red_scale[0], "max": red_scale[1]},
                },
            },
            "z_window": z_window,
            "roi_window": roi_window,
            "settings": self._settings_dict(),
        }

    def _resolve_required_channels(self, metadata: object) -> Optional[dict[str, tuple[int, str]]]:
        channels = getattr(metadata, "channels", [])
        lookup = {self._normalize_channel_name(getattr(channel, "name", "")): channel for channel in channels}
        brightfield = lookup.get(self._normalize_channel_name(self.brightfield_channel_name))
        green = lookup.get(self._normalize_channel_name(self.green_channel_name))
        red = lookup.get(self._normalize_channel_name(self.red_channel_name))
        if brightfield is None or green is None or red is None:
            return None
        return {
            "brightfield": (int(brightfield.index), str(brightfield.name)),
            "green": (int(green.index), str(green.name)),
            "red": (int(red.index), str(red.name)),
        }

    @staticmethod
    def _normalize_channel_name(name: str) -> str:
        return " ".join((name or "").strip().lower().split())

    def _effective_z_window(self, dataset: h5py.Dataset) -> dict[str, int]:
        if dataset.ndim < 3:
            return {"total_z": 1, "z_start": 0, "z_end": 0, "z_count": 1}
        total_z = int(dataset.shape[0])
        z_start = min(max(self.z_start, 0), total_z - 1)
        z_end = min(max(self.z_end, z_start), total_z - 1)
        return {
            "total_z": total_z,
            "z_start": z_start,
            "z_end": z_end,
            "z_count": z_end - z_start + 1,
        }

    def _estimate_display_range(
        self,
        dataset: h5py.Dataset,
        *,
        z_window: dict[str, int],
        channel_label: str,
        preprocess: bool,
        roi_window: dict[str, int | bool],
    ) -> tuple[float, float]:
        print(
            f"[step] Estimating display range for {channel_label} "
            f"(preprocess={preprocess}, mode={self.scale_mode})"
        )
        z_indices = list(range(z_window["z_start"], z_window["z_end"] + 1, self.scale_estimation_z_step))
        if z_indices[-1] != z_window["z_end"]:
            z_indices.append(z_window["z_end"])

        sample_values: list[np.ndarray] = []
        global_min = float("inf")
        global_max = float("-inf")
        for sample_idx, z_index in enumerate(z_indices, start=1):
            frame = np.asarray(dataset[z_index, ...], dtype=np.float32)
            frame = self._crop_to_roi(frame, roi_window)
            if preprocess:
                frame = self._preprocess_fluorescence_frame(frame, channel_label=f"{channel_label} scale-estimation")
            global_min = min(global_min, float(np.min(frame)))
            global_max = max(global_max, float(np.max(frame)))
            sampled = frame[:: self.scale_estimation_xy_step, :: self.scale_estimation_xy_step]
            sample_values.append(sampled.reshape(-1))
            if sample_idx == 1 or sample_idx % self.progress_interval == 0 or sample_idx == len(z_indices):
                print(
                    f"[progress] {channel_label} scale estimation: "
                    f"{sample_idx}/{len(z_indices)} sampled z slices"
                )

        if not sample_values:
            return 0.0, 0.0
        sample = np.concatenate(sample_values)
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return 0.0, 0.0

        if self.scale_mode == "full-range":
            vmin = global_min
            vmax = global_max
        else:
            vmin = float(np.percentile(finite, self.scale_low_percentile))
            vmax = float(np.percentile(finite, self.scale_high_percentile))
        if vmax <= vmin:
            vmin = global_min
            vmax = global_max
        if vmax <= vmin:
            return vmin, vmax
        return vmin, vmax

    def _preprocess_fluorescence_frame(self, frame: np.ndarray, *, channel_label: str) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, median_filter  # type: ignore

        processed = np.array(frame, copy=True)
        if self.background_subtract_sigma > 0:
            background = gaussian_filter(processed, sigma=self.background_subtract_sigma, mode="nearest")
            processed = np.clip(processed - background, 0.0, None)
        if self.median_filter_size >= 3:
            processed = median_filter(processed, size=self.median_filter_size, mode="nearest")
        if self.preserve_edge_pixels > 0:
            processed = self._restore_edge_pixels(
                original=frame,
                processed=processed,
                border=self.preserve_edge_pixels,
            )
        return processed

    @staticmethod
    def _crop_to_roi(array: np.ndarray, roi_window: dict[str, int | bool]) -> np.ndarray:
        if not bool(roi_window["enabled"]):
            return array
        return array[int(roi_window["y_start"]) : int(roi_window["y_end"]), int(roi_window["x_start"]) : int(roi_window["x_end"])]

    def _resolve_roi_window(self, image_shape: tuple[int, int]) -> dict[str, int | bool]:
        height, width = int(image_shape[0]), int(image_shape[1])
        if not self.roi_enabled:
            return {
                "enabled": False,
                "requested_center_x": -1,
                "requested_center_y": -1,
                "requested_size": 0,
                "x_start": 0,
                "x_end": width,
                "y_start": 0,
                "y_end": height,
                "width": width,
                "height": height,
            }

        roi_size = min(int(self.roi_size), width, height)
        center_x = min(max(int(self.roi_center_x), 0), width - 1)
        center_y = min(max(int(self.roi_center_y), 0), height - 1)
        x_start = center_x - (roi_size // 2)
        y_start = center_y - (roi_size // 2)
        x_start = min(max(x_start, 0), width - roi_size)
        y_start = min(max(y_start, 0), height - roi_size)
        x_end = x_start + roi_size
        y_end = y_start + roi_size
        return {
            "enabled": True,
            "requested_center_x": int(self.roi_center_x),
            "requested_center_y": int(self.roi_center_y),
            "requested_size": int(self.roi_size),
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
            "width": roi_size,
            "height": roi_size,
        }

    @staticmethod
    def _print_roi_summary(roi_window: dict[str, int | bool]) -> None:
        if not bool(roi_window["enabled"]):
            print("[step] Square ROI view disabled: using the full field of view")
            return
        print(
            "[step] Applying square ROI view: "
            f"x={roi_window['x_start']}:{roi_window['x_end']}, "
            f"y={roi_window['y_start']}:{roi_window['y_end']} "
            f"(size={roi_window['width']} px)"
        )

    def _build_output_name(self, ims_path: Path, z_window: dict[str, int], roi_window: dict[str, int | bool]) -> str:
        roi_suffix = ""
        if bool(roi_window["enabled"]):
            roi_suffix = (
                f"_roi_x{int(roi_window['requested_center_x']):04d}"
                f"_y{int(roi_window['requested_center_y']):04d}"
                f"_s{int(roi_window['requested_size']):04d}"
            )
        return f"{ims_path.stem}_zmovie_z{z_window['z_start']:04d}_{z_window['z_end']:04d}{roi_suffix}.mp4"

    @staticmethod
    def _restore_edge_pixels(
        *,
        original: np.ndarray,
        processed: np.ndarray,
        border: int,
    ) -> np.ndarray:
        if border <= 0:
            return processed
        height, width = processed.shape
        border = min(border, height // 2 if height > 1 else 1, width // 2 if width > 1 else 1)
        restored = np.array(processed, copy=True)
        restored[:border, :] = original[:border, :]
        restored[-border:, :] = original[-border:, :]
        restored[:, :border] = original[:, :border]
        restored[:, -border:] = original[:, -border:]
        return restored

    def _normalize_with_scale(self, frame: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
        vmin, vmax = scale
        if vmax <= vmin:
            return np.zeros_like(frame, dtype=np.uint8)
        clipped = np.clip(frame.astype(np.float32, copy=False), vmin, vmax)
        norm = (clipped - vmin) / (vmax - vmin)
        if self.display_gamma != 1.0:
            norm = np.power(norm, self.display_gamma)
        if self.invert_display:
            norm = 1.0 - norm
        return np.round(norm * 255.0).astype(np.uint8)

    def _compose_frame(
        self,
        *,
        brightfield: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
        brightfield_scale: tuple[float, float],
        green_scale: tuple[float, float],
        red_scale: tuple[float, float],
        z_index: int,
        z_window: dict[str, int],
    ) -> np.ndarray:
        brightfield_u8 = self._normalize_with_scale(brightfield, brightfield_scale)
        green_u8 = self._normalize_with_scale(green, green_scale)
        red_u8 = self._normalize_with_scale(red, red_scale)

        bf_rgb = np.repeat(brightfield_u8[..., None], 3, axis=2)
        green_rgb = np.zeros((*green_u8.shape, 3), dtype=np.uint8)
        red_rgb = np.zeros((*red_u8.shape, 3), dtype=np.uint8)
        merged_rgb = np.zeros((*red_u8.shape, 3), dtype=np.uint8)
        green_rgb[..., 1] = green_u8
        red_rgb[..., 0] = red_u8
        merged_rgb[..., 0] = red_u8
        merged_rgb[..., 1] = green_u8
        strip = np.concatenate([bf_rgb, green_rgb, red_rgb, merged_rgb], axis=1)
        if self.include_scale_bars:
            strip = self._append_scale_bars(strip, green_scale=green_scale, red_scale=red_scale)
        strip = self._draw_frame_overlay(strip, z_index=z_index, z_window=z_window)
        return strip

    def _append_scale_bars(
        self,
        strip: np.ndarray,
        *,
        green_scale: tuple[float, float],
        red_scale: tuple[float, float],
    ) -> np.ndarray:
        height, width, _ = strip.shape
        panel_width = width // 4
        gap = 8
        bar = max(12, self.scale_bar_width)
        label_width = 92
        out = np.full((height, width + 2 * (gap + bar + label_width), 3), 18, dtype=np.uint8)
        cursor = 0
        gradient = np.linspace(255, 0, num=height, dtype=np.uint8)[:, None]

        out[:, cursor : cursor + panel_width, :] = strip[:, :panel_width, :]
        cursor += panel_width

        out[:, cursor : cursor + panel_width, :] = strip[:, panel_width : (2 * panel_width), :]
        cursor += panel_width
        out[:, cursor : cursor + gap, :] = 32
        cursor += gap
        green_bar = np.zeros((height, bar, 3), dtype=np.uint8)
        green_bar[:, 1:-1, 1] = gradient
        green_bar[:, 0, :] = 255
        green_bar[:, -1, :] = 255
        green_bar[:2, :, :] = 255
        green_bar[-2:, :, :] = 255
        out[:, cursor : cursor + bar, :] = green_bar
        cursor += bar
        self._draw_scale_labels(out, x0=cursor, width=label_width, scale=green_scale, channel_name="Green")
        cursor += label_width

        out[:, cursor : cursor + panel_width, :] = strip[:, (2 * panel_width) : (3 * panel_width), :]
        cursor += panel_width
        out[:, cursor : cursor + gap, :] = 32
        cursor += gap
        red_bar = np.zeros((height, bar, 3), dtype=np.uint8)
        red_bar[:, 1:-1, 0] = gradient
        red_bar[:, 0, :] = 255
        red_bar[:, -1, :] = 255
        red_bar[:2, :, :] = 255
        red_bar[-2:, :, :] = 255
        out[:, cursor : cursor + bar, :] = red_bar
        cursor += bar
        self._draw_scale_labels(out, x0=cursor, width=label_width, scale=red_scale, channel_name="Red")
        cursor += label_width

        out[:, cursor : cursor + panel_width, :] = strip[:, (3 * panel_width) :, :]
        return out

    @staticmethod
    def _draw_scale_labels(
        canvas: np.ndarray,
        *,
        x0: int,
        width: int,
        scale: tuple[float, float],
        channel_name: str,
    ) -> None:
        label_region = canvas[:, x0 : x0 + width, :]
        image = Image.fromarray(label_region, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        max_text = f"{scale[1]:.3g}"
        min_text = f"{scale[0]:.3g}"
        draw.text((4, 4), channel_name, fill=(255, 255, 255), font=font)
        draw.text((4, 18), "max", fill=(255, 255, 255), font=font)
        draw.text((4, 30), max_text, fill=(255, 255, 255), font=font)
        bottom_y = max(4, image.height - 26)
        draw.text((4, bottom_y - 12), "min", fill=(255, 255, 255), font=font)
        draw.text((4, bottom_y), min_text, fill=(255, 255, 255), font=font)
        canvas[:, x0 : x0 + width, :] = np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _draw_frame_overlay(frame: np.ndarray, *, z_index: int, z_window: dict[str, int]) -> np.ndarray:
        image = Image.fromarray(frame, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = f"z={z_index}  window={z_window['z_start']}..{z_window['z_end']}"
        draw.rectangle((8, 8, 8 + 8 * len(text), 24), fill=(0, 0, 0))
        draw.text((12, 10), text, fill=(255, 255, 255), font=font)
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _print_scale_summary(channel_name: str, scale: tuple[float, float]) -> None:
        print(f"[step] Display range for {channel_name}: min={scale[0]:.3f}, max={scale[1]:.3f}")

    def _open_mp4_writer(self, path: Path):
        try:
            return imageio.get_writer(
                path,
                fps=self.fps,
                codec="libx264",
                format="FFMPEG",
                macro_block_size=None,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "MP4 writing requires an ffmpeg backend, but none is available in this environment. "
                "Install ffmpeg or imageio-ffmpeg in the same conda environment."
            ) from exc

    def _write_manifest(
        self,
        records: list[PreparedMovieRecord],
        path: Path,
        *,
        run_metadata_path: Path,
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_path",
                    "output_path",
                    "output_metadata_path",
                    "run_metadata_path",
                    "brightfield_channel_index",
                    "green_channel_index",
                    "red_channel_index",
                    "z_start_requested",
                    "z_end_requested",
                    "z_start_effective",
                    "z_end_effective",
                    "roi_enabled",
                    "roi_center_x_requested",
                    "roi_center_y_requested",
                    "roi_size_requested",
                    "roi_x_start",
                    "roi_x_end",
                    "roi_y_start",
                    "roi_y_end",
                    "frame_count",
                    "fps",
                    "brightfield_min_value",
                    "brightfield_max_value",
                    "green_min_value",
                    "green_max_value",
                    "red_min_value",
                    "red_max_value",
                    "recursive",
                    "overwrite",
                    "resolution_level",
                    "time_point",
                    "background_subtract_sigma",
                    "median_filter_size",
                    "preserve_edge_pixels",
                    "invert_display",
                    "scale_mode",
                    "display_gamma",
                    "scale_low_percentile",
                    "scale_high_percentile",
                    "include_scale_bars",
                    "scale_bar_width",
                    "progress_interval",
                    "scale_estimation_z_step",
                    "scale_estimation_xy_step",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "source_path": str(record.source_path),
                        "output_path": str(record.output_path),
                        "output_metadata_path": str(record.output_metadata_path),
                        "run_metadata_path": str(run_metadata_path),
                        "brightfield_channel_index": record.brightfield_channel_index,
                        "green_channel_index": record.green_channel_index,
                        "red_channel_index": record.red_channel_index,
                        "z_start_requested": record.z_start_requested,
                        "z_end_requested": record.z_end_requested,
                        "z_start_effective": record.z_start_effective,
                        "z_end_effective": record.z_end_effective,
                        "roi_enabled": record.roi_enabled,
                        "roi_center_x_requested": record.roi_center_x_requested,
                        "roi_center_y_requested": record.roi_center_y_requested,
                        "roi_size_requested": record.roi_size_requested,
                        "roi_x_start": record.roi_x_start,
                        "roi_x_end": record.roi_x_end,
                        "roi_y_start": record.roi_y_start,
                        "roi_y_end": record.roi_y_end,
                        "frame_count": record.frame_count,
                        "fps": f"{record.fps:.6g}",
                        "brightfield_min_value": f"{record.brightfield_min_value:.6g}",
                        "brightfield_max_value": f"{record.brightfield_max_value:.6g}",
                        "green_min_value": f"{record.green_min_value:.6g}",
                        "green_max_value": f"{record.green_max_value:.6g}",
                        "red_min_value": f"{record.red_min_value:.6g}",
                        "red_max_value": f"{record.red_max_value:.6g}",
                        "recursive": self.recursive,
                        "overwrite": self.overwrite,
                        "resolution_level": self.resolution_level,
                        "time_point": self.time_point,
                        "background_subtract_sigma": f"{self.background_subtract_sigma:.6g}",
                        "median_filter_size": self.median_filter_size,
                        "preserve_edge_pixels": self.preserve_edge_pixels,
                        "invert_display": self.invert_display,
                        "scale_mode": self.scale_mode,
                        "display_gamma": f"{self.display_gamma:.6g}",
                        "scale_low_percentile": f"{self.scale_low_percentile:.6g}",
                        "scale_high_percentile": f"{self.scale_high_percentile:.6g}",
                        "include_scale_bars": self.include_scale_bars,
                        "scale_bar_width": self.scale_bar_width,
                        "progress_interval": self.progress_interval,
                        "scale_estimation_z_step": self.scale_estimation_z_step,
                        "scale_estimation_xy_step": self.scale_estimation_xy_step,
                    }
                )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render synchronized 1x4 brightfield/green/red/merged MP4 movies "
            "from Imaris files across a restricted z window."
        )
    )
    parser.add_argument("input_path", type=Path, help="Path to a single .ims file or a directory containing .ims files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Desktop" / "volumetric_brightfield_confocal_movies",
        help="Output directory for MP4 movies (default: %(default)s).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .ims files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--resolution-level", type=int, default=0)
    parser.add_argument("--time-point", type=int, default=0)
    parser.add_argument("--z-start", type=int, default=90)
    parser.add_argument("--z-end", type=int, default=672)
    parser.add_argument("--brightfield-channel-name", default=DEFAULT_BRIGHTFIELD_CHANNEL_NAME)
    parser.add_argument("--green-channel-name", default=DEFAULT_GREEN_CHANNEL_NAME)
    parser.add_argument("--red-channel-name", default=DEFAULT_RED_CHANNEL_NAME)
    parser.add_argument("--background-subtract-sigma", type=float, default=0.0)
    parser.add_argument("--median-filter-size", type=int, default=0)
    parser.add_argument("--preserve-edge-pixels", type=int, default=0)
    parser.add_argument("--invert-display", action="store_true")
    parser.add_argument("--scale-mode", choices=("percentile", "full-range"), default="percentile")
    parser.add_argument("--display-gamma", type=float, default=1.0)
    parser.add_argument("--scale-low-percentile", type=float, default=0.0)
    parser.add_argument("--scale-high-percentile", type=float, default=99.5)
    parser.add_argument("--no-scale-bars", action="store_true")
    parser.add_argument("--scale-bar-width", type=int, default=14)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument(
        "--roi-center-x",
        type=int,
        default=None,
        help="X coordinate of the square ROI center in pixels. Use with --roi-center-y and --roi-size.",
    )
    parser.add_argument(
        "--roi-center-y",
        type=int,
        default=None,
        help="Y coordinate of the square ROI center in pixels. Use with --roi-center-x and --roi-size.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=None,
        help="Square ROI width/height in pixels. This crops the displayed field of view without masking.",
    )
    parser.add_argument(
        "--scale-estimation-z-step",
        type=int,
        default=5,
        help="Use every Nth z slice when estimating global display ranges (default: %(default)s).",
    )
    parser.add_argument(
        "--scale-estimation-xy-step",
        type=int,
        default=16,
        help="Use every Nth pixel in x/y when estimating global display ranges (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    preparer = BrightfieldConfocalMoviePreparer(
        input_path=args.input_path,
        output_dir=args.output_dir,
        recursive=args.recursive,
        overwrite=args.overwrite,
        resolution_level=args.resolution_level,
        time_point=args.time_point,
        z_start=args.z_start,
        z_end=args.z_end,
        brightfield_channel_name=args.brightfield_channel_name,
        green_channel_name=args.green_channel_name,
        red_channel_name=args.red_channel_name,
        background_subtract_sigma=args.background_subtract_sigma,
        median_filter_size=args.median_filter_size,
        preserve_edge_pixels=args.preserve_edge_pixels,
        invert_display=args.invert_display,
        scale_mode=args.scale_mode,
        display_gamma=args.display_gamma,
        scale_low_percentile=args.scale_low_percentile,
        scale_high_percentile=args.scale_high_percentile,
        include_scale_bars=not args.no_scale_bars,
        scale_bar_width=args.scale_bar_width,
        fps=args.fps,
        progress_interval=args.progress_interval,
        scale_estimation_z_step=args.scale_estimation_z_step,
        scale_estimation_xy_step=args.scale_estimation_xy_step,
        roi_center_x=args.roi_center_x,
        roi_center_y=args.roi_center_y,
        roi_size=args.roi_size,
        cli_args=(argv if argv is not None else sys.argv[1:]),
    )
    preparer.prepare_all()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
