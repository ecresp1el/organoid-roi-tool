"""Prepare brightfield/confocal Imaris volumes into 1x4 preview images.

This script scans a folder for ``.ims`` files, resolves the channels:

1. White Light - Brightfield
2. Confocal - Green
3. Confocal - Red

It computes max-intensity projections over a restricted z-window and exports
one 1x4 RGB strip per file:

1. Brightfield only
2. Green only
3. Red only
4. Merged red+green

The red and green panels include labeled intensity scale bars by default.
The brightfield and merged panels intentionally have no scale bars.
Outputs default to ``~/Desktop/volumetric_brightfield_confocal_outputs``.
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
import numpy as np
import tifffile as tiff  # type: ignore

from imaris_tools.metadata import _discover_channels, read_metadata

DEFAULT_BRIGHTFIELD_CHANNEL_NAME = "White Light - Brightfield"
DEFAULT_GREEN_CHANNEL_NAME = "Confocal - Green"
DEFAULT_RED_CHANNEL_NAME = "Confocal - Red"


@dataclass(frozen=True)
class PreparedBrightfieldConfocalRecord:
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
    height_px: int
    width_px: int
    brightfield_min_value: float
    brightfield_max_value: float
    green_min_value: float
    green_max_value: float
    red_min_value: float
    red_max_value: float


class BrightfieldConfocalVolumePreparer:
    """Prepare brightfield/green/red z-windowed overview strips from ``.ims`` files."""

    def __init__(
        self,
        input_dir: Path,
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
        cli_args: Optional[list[str]] = None,
    ) -> None:
        self.input_dir = input_dir.expanduser().resolve()
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
        self.cli_args = list(cli_args or [])

    def discover_files(self) -> list[Path]:
        iterator: Iterable[Path]
        if self.recursive:
            iterator = self.input_dir.rglob("*.ims")
        else:
            iterator = self.input_dir.glob("*.ims")
        return sorted(path for path in iterator if path.is_file())

    def prepare_all(self) -> list[PreparedBrightfieldConfocalRecord]:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input folder does not exist or is not a directory: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ims_files = self.discover_files()
        records: list[PreparedBrightfieldConfocalRecord] = []
        run_metadata_path = self.output_dir / "preparation_run_metadata.json"
        run_metadata = self._build_run_metadata(ims_files=ims_files)
        self._write_json(run_metadata_path, run_metadata)

        print("[info] ================================================")
        print(f"[info] Input directory: {self.input_dir}")
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
        print(f"[info] Invert display: {self.invert_display}")
        print(
            "[info] Display scaling: "
            f"mode={self.scale_mode}, gamma={self.display_gamma:.3f}, "
            f"percentiles={self.scale_low_percentile:.2f}->{self.scale_high_percentile:.2f}"
        )
        print(f"[info] Include green/red scale bars: {self.include_scale_bars}")
        print(f"[info] Run metadata JSON: {run_metadata_path}")
        print("[info] ================================================")

        for file_index, ims_path in enumerate(ims_files, start=1):
            print(f"\n[info] Processing file {file_index}/{len(ims_files)}: {ims_path.name}")
            metadata = read_metadata(
                ims_path,
                resolution_level=self.resolution_level,
                time_point=self.time_point,
            )
            channel_map = self._resolve_required_channels(metadata)
            if channel_map is None:
                print(
                    "[warn] "
                    f"Skipping {ims_path.name}: required channels not found "
                    f"('{self.brightfield_channel_name}', '{self.green_channel_name}', '{self.red_channel_name}')."
                )
                continue

            with h5py.File(ims_path, "r") as handle:
                datasets = dict(
                    _discover_channels(
                        handle,
                        resolution_level=self.resolution_level,
                        time_point=self.time_point,
                    )
                )
                brightfield_projection, brightfield_window = self._compute_windowed_projection(
                    datasets[channel_map["brightfield"][0]]
                )
                green_projection, green_window = self._compute_windowed_projection(
                    datasets[channel_map["green"][0]]
                )
                red_projection, red_window = self._compute_windowed_projection(
                    datasets[channel_map["red"][0]]
                )

                self._print_projection_proof(
                    datasets[channel_map["brightfield"][0]],
                    brightfield_projection,
                    channel_index=channel_map["brightfield"][0],
                    channel_name=channel_map["brightfield"][1],
                    z_window=brightfield_window,
                )
                self._print_projection_proof(
                    datasets[channel_map["green"][0]],
                    green_projection,
                    channel_index=channel_map["green"][0],
                    channel_name=channel_map["green"][1],
                    z_window=green_window,
                )
                self._print_projection_proof(
                    datasets[channel_map["red"][0]],
                    red_projection,
                    channel_index=channel_map["red"][0],
                    channel_name=channel_map["red"][1],
                    z_window=red_window,
                )

            output_path = self.output_dir / (
                f"{ims_path.stem}_brightfield_green_red_merged_"
                f"z{green_window['z_start']:04d}_{green_window['z_end']:04d}.tif"
            )
            if output_path.exists() and not self.overwrite:
                print(f"[warn] Output exists, skipping (use --overwrite to replace): {output_path}")
                continue

            brightfield_processed = brightfield_projection.astype(np.float32, copy=False)
            green_processed = self._preprocess_fluorescence_projection(
                green_projection,
                channel_label=f"green ch{channel_map['green'][0]}",
            )
            red_processed = self._preprocess_fluorescence_projection(
                red_projection,
                channel_label=f"red ch{channel_map['red'][0]}",
            )

            strip, brightfield_scale, green_scale, red_scale = self._compose_quadtych(
                brightfield_processed,
                green_processed,
                red_processed,
            )
            if self.include_scale_bars:
                strip = self._append_scale_bars(
                    strip,
                    green_scale=green_scale,
                    red_scale=red_scale,
                )

            output_metadata_path = output_path.with_suffix(".metadata.json")
            tiff.imwrite(output_path, strip, photometric="rgb")
            self._write_json(
                output_metadata_path,
                self._build_image_metadata(
                    ims_path=ims_path,
                    output_path=output_path,
                    output_metadata_path=output_metadata_path,
                    run_metadata_path=run_metadata_path,
                    channel_map=channel_map,
                    brightfield_scale=brightfield_scale,
                    green_scale=green_scale,
                    red_scale=red_scale,
                    z_window=green_window,
                ),
            )

            record = PreparedBrightfieldConfocalRecord(
                source_path=ims_path,
                output_path=output_path,
                output_metadata_path=output_metadata_path,
                brightfield_channel_index=channel_map["brightfield"][0],
                green_channel_index=channel_map["green"][0],
                red_channel_index=channel_map["red"][0],
                z_start_requested=self.z_start,
                z_end_requested=self.z_end,
                z_start_effective=green_window["z_start"],
                z_end_effective=green_window["z_end"],
                height_px=int(strip.shape[0]),
                width_px=int(strip.shape[1]),
                brightfield_min_value=brightfield_scale[0],
                brightfield_max_value=brightfield_scale[1],
                green_min_value=green_scale[0],
                green_max_value=green_scale[1],
                red_min_value=red_scale[0],
                red_max_value=red_scale[1],
            )
            records.append(record)
            print(
                "[ok] "
                f"{ims_path.name} -> {output_path.name} | "
                f"shape={strip.shape} | "
                f"z window={green_window['z_start']}..{green_window['z_end']} | "
                f"green scale=[{green_scale[0]:.3f}, {green_scale[1]:.3f}] | "
                f"red scale=[{red_scale[0]:.3f}, {red_scale[1]:.3f}]"
            )

        manifest = self.output_dir / "prepared_manifest.csv"
        self._write_manifest(records, manifest, run_metadata_path=run_metadata_path)
        print(f"\n[info] Wrote manifest with {len(records)} record(s): {manifest}")
        return records

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
        }

    def _build_run_metadata(self, *, ims_files: list[Path]) -> dict[str, object]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable,
            "cli_args": self.cli_args,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "ims_files_discovered": [str(path) for path in ims_files],
            "settings": self._settings_dict(),
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_image_metadata(
        self,
        *,
        ims_path: Path,
        output_path: Path,
        output_metadata_path: Path,
        run_metadata_path: Path,
        channel_map: dict[str, tuple[int, str]],
        brightfield_scale: tuple[float, float],
        green_scale: tuple[float, float],
        red_scale: tuple[float, float],
        z_window: dict[str, int],
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

    def _compute_windowed_projection(self, dataset: h5py.Dataset) -> tuple[np.ndarray, dict[str, int]]:
        if dataset.ndim < 3:
            data = np.asarray(dataset[()], dtype=np.float32)
            if data.ndim != 2:
                raise ValueError("Expected a 2D or 3D dataset for projection")
            return data, {
                "total_z": 1,
                "z_start": 0,
                "z_end": 0,
                "z_count": 1,
            }

        total_z = int(dataset.shape[0])
        if total_z <= 0:
            raise ValueError("Channel dataset has no z slices")

        z_start = min(max(self.z_start, 0), total_z - 1)
        z_end = min(max(self.z_end, z_start), total_z - 1)

        proj = np.asarray(dataset[z_start, ...], dtype=np.float32)
        for z_index in range(z_start + 1, z_end + 1):
            slice_data = np.asarray(dataset[z_index, ...], dtype=np.float32)
            np.maximum(proj, slice_data, out=proj)

        return proj, {
            "total_z": total_z,
            "z_start": z_start,
            "z_end": z_end,
            "z_count": z_end - z_start + 1,
        }

    def _normalize_u8(self, array: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        data = array.astype(np.float32, copy=False)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return np.zeros_like(data, dtype=np.uint8), (0.0, 0.0)

        if self.scale_mode == "full-range":
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            vmin = float(np.percentile(finite, self.scale_low_percentile))
            vmax = float(np.percentile(finite, self.scale_high_percentile))
        if vmax <= vmin:
            vmax = float(np.max(finite))
            vmin = float(np.min(finite))
        if vmax <= vmin:
            return np.zeros_like(data, dtype=np.uint8), (vmin, vmax)

        clipped = np.clip(data, vmin, vmax)
        norm = (clipped - vmin) / (vmax - vmin)
        if self.display_gamma != 1.0:
            norm = np.power(norm, self.display_gamma)
        if self.invert_display:
            norm = 1.0 - norm
        return np.round(norm * 255.0).astype(np.uint8), (vmin, vmax)

    def _preprocess_fluorescence_projection(self, array: np.ndarray, *, channel_label: str) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, median_filter  # type: ignore

        data = array.astype(np.float32, copy=False)
        processed = np.array(data, copy=True)
        raw_min = float(np.min(processed))
        raw_max = float(np.max(processed))

        if self.background_subtract_sigma > 0:
            background = gaussian_filter(processed, sigma=self.background_subtract_sigma, mode="nearest")
            processed = np.clip(processed - background, 0.0, None)

        if self.median_filter_size >= 3:
            processed = median_filter(processed, size=self.median_filter_size, mode="nearest")

        if self.preserve_edge_pixels > 0:
            processed = self._restore_edge_pixels(
                original=data,
                processed=processed,
                border=self.preserve_edge_pixels,
            )

        print(
            "[info] Preprocessed "
            f"{channel_label}: raw_range=[{raw_min:.3f}, {raw_max:.3f}] -> "
            f"processed_range=[{float(np.min(processed)):.3f}, {float(np.max(processed)):.3f}]"
        )
        return processed

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

    def _compose_quadtych(
        self,
        brightfield_projection: np.ndarray,
        green_projection: np.ndarray,
        red_projection: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], tuple[float, float]]:
        brightfield_u8, brightfield_scale = self._normalize_u8(brightfield_projection)
        green_u8, green_scale = self._normalize_u8(green_projection)
        red_u8, red_scale = self._normalize_u8(red_projection)

        brightfield_rgb = np.repeat(brightfield_u8[..., None], 3, axis=2)
        green_rgb = np.zeros((*green_u8.shape, 3), dtype=np.uint8)
        red_rgb = np.zeros((*red_u8.shape, 3), dtype=np.uint8)
        merged_rgb = np.zeros((*green_u8.shape, 3), dtype=np.uint8)

        green_rgb[..., 1] = green_u8
        red_rgb[..., 0] = red_u8
        merged_rgb[..., 0] = red_u8
        merged_rgb[..., 1] = green_u8

        strip = np.concatenate([brightfield_rgb, green_rgb, red_rgb, merged_rgb], axis=1)
        return strip, brightfield_scale, green_scale, red_scale

    def _print_projection_proof(
        self,
        dataset: h5py.Dataset,
        projection: np.ndarray,
        *,
        channel_index: int,
        channel_name: str,
        z_window: dict[str, int],
    ) -> None:
        proof = self._projection_proof_stats(dataset, projection, z_window=z_window)
        print(
            "[proof] "
            f"Channel {channel_index} ({channel_name}) raw shape={proof['raw_shape']} "
            f"-> total_z={proof['total_z']} -> used z={proof['z_start']}..{proof['z_end']} "
            f"({proof['z_count']} slices) -> plotted projection shape={proof['projection_shape']}"
        )
        print(
            "[proof] "
            f"Channel {channel_index} window slice maxima: "
            f"min={proof['slice_max_min']:.3f}, median={proof['slice_max_median']:.3f}, "
            f"max={proof['slice_max_max']:.3f}"
        )
        print(
            "[proof] "
            f"Channel {channel_index} first used slice range=[{proof['first_slice_min']:.3f}, {proof['first_slice_max']:.3f}] "
            f"-> max projection range=[{proof['projection_min']:.3f}, {proof['projection_max']:.3f}]"
        )
        print(
            "[proof] "
            f"Channel {channel_index} pixels increased beyond first used slice due to later z slices: "
            f"{proof['improved_pixels']}/{proof['pixel_count']} "
            f"({proof['improved_fraction']:.2%}), max_delta={proof['max_delta_vs_first']:.3f}"
        )

    @staticmethod
    def _projection_proof_stats(
        dataset: h5py.Dataset,
        projection: np.ndarray,
        *,
        z_window: dict[str, int],
    ) -> dict[str, object]:
        total_z = int(z_window["total_z"])
        z_start = int(z_window["z_start"])
        z_end = int(z_window["z_end"])
        z_count = int(z_window["z_count"])

        if dataset.ndim < 3:
            first_slice = np.asarray(dataset[()], dtype=np.float32)
            slice_maxima = np.array([float(np.max(first_slice))], dtype=np.float32)
        else:
            first_slice = np.asarray(dataset[z_start, ...], dtype=np.float32)
            slice_maxima = np.empty(z_count, dtype=np.float32)
            for local_idx, z_index in enumerate(range(z_start, z_end + 1)):
                slice_data = np.asarray(dataset[z_index, ...], dtype=np.float32)
                slice_maxima[local_idx] = float(np.max(slice_data))

        projection_f32 = projection.astype(np.float32, copy=False)
        delta_vs_first = projection_f32 - first_slice
        improved_mask = delta_vs_first > 0
        improved_pixels = int(np.count_nonzero(improved_mask))
        pixel_count = int(projection_f32.size)
        max_delta = float(np.max(delta_vs_first[improved_mask])) if improved_pixels else 0.0

        return {
            "raw_shape": tuple(int(dim) for dim in dataset.shape),
            "total_z": total_z,
            "z_start": z_start,
            "z_end": z_end,
            "z_count": z_count,
            "projection_shape": tuple(int(dim) for dim in projection.shape),
            "slice_max_min": float(np.min(slice_maxima)),
            "slice_max_median": float(np.median(slice_maxima)),
            "slice_max_max": float(np.max(slice_maxima)),
            "first_slice_min": float(np.min(first_slice)),
            "first_slice_max": float(np.max(first_slice)),
            "projection_min": float(np.min(projection_f32)),
            "projection_max": float(np.max(projection_f32)),
            "improved_pixels": improved_pixels,
            "pixel_count": pixel_count,
            "improved_fraction": (improved_pixels / pixel_count) if pixel_count else 0.0,
            "max_delta_vs_first": max_delta,
        }

    def _append_scale_bars(
        self,
        strip: np.ndarray,
        *,
        green_scale: tuple[float, float],
        red_scale: tuple[float, float],
    ) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        height, width, _ = strip.shape
        panel_width = width // 4
        gap = 8
        bar = max(12, self.scale_bar_width)
        label_width = 92

        out = np.full((height, width + 2 * (gap + bar + label_width), 3), 18, dtype=np.uint8)
        cursor = 0
        gradient = np.linspace(255, 0, num=height, dtype=np.uint8)[:, None]

        brightfield_panel = strip[:, :panel_width, :]
        out[:, cursor : cursor + panel_width, :] = brightfield_panel
        cursor += panel_width

        green_panel = strip[:, panel_width : (2 * panel_width), :]
        out[:, cursor : cursor + panel_width, :] = green_panel
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
        self._draw_scale_labels(
            out,
            x0=cursor,
            width=label_width,
            scale=green_scale,
            channel_name="Green",
            image_api=(Image, ImageDraw, ImageFont),
        )
        cursor += label_width

        red_panel = strip[:, (2 * panel_width) : (3 * panel_width), :]
        out[:, cursor : cursor + panel_width, :] = red_panel
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
        self._draw_scale_labels(
            out,
            x0=cursor,
            width=label_width,
            scale=red_scale,
            channel_name="Red",
            image_api=(Image, ImageDraw, ImageFont),
        )
        cursor += label_width

        merged_panel = strip[:, (3 * panel_width) :, :]
        out[:, cursor : cursor + panel_width, :] = merged_panel

        print(
            "[info] Added scale bars | "
            f"green={green_scale[0]:.3f}->{green_scale[1]:.3f}, "
            f"red={red_scale[0]:.3f}->{red_scale[1]:.3f}, "
            "brightfield/merged have no scale bar"
        )
        return out

    @staticmethod
    def _draw_scale_labels(
        canvas: np.ndarray,
        *,
        x0: int,
        width: int,
        scale: tuple[float, float],
        channel_name: str,
        image_api: tuple[object, object, object],
    ) -> None:
        Image, ImageDraw, ImageFont = image_api
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

    def _write_manifest(
        self,
        records: list[PreparedBrightfieldConfocalRecord],
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
                    "height_px",
                    "width_px",
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
                        "height_px": record.height_px,
                        "width_px": record.width_px,
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
                    }
                )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 1x4 brightfield/green/red/merged strips from Imaris files "
            "using a restricted z window."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .ims files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Desktop" / "volumetric_brightfield_confocal_outputs",
        help="Output directory for prepared strips (default: %(default)s).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .ims files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--resolution-level", type=int, default=0)
    parser.add_argument("--time-point", type=int, default=0)
    parser.add_argument(
        "--z-start",
        type=int,
        default=90,
        help="First z slice to include in the max projection window, inclusive (default: %(default)s).",
    )
    parser.add_argument(
        "--z-end",
        type=int,
        default=672,
        help="Last z slice to include in the max projection window, inclusive (default: %(default)s).",
    )
    parser.add_argument(
        "--brightfield-channel-name",
        default=DEFAULT_BRIGHTFIELD_CHANNEL_NAME,
        help="Exact channel name for brightfield (default: %(default)s).",
    )
    parser.add_argument(
        "--green-channel-name",
        default=DEFAULT_GREEN_CHANNEL_NAME,
        help="Exact channel name for the green confocal channel (default: %(default)s).",
    )
    parser.add_argument(
        "--red-channel-name",
        default=DEFAULT_RED_CHANNEL_NAME,
        help="Exact channel name for the red confocal channel (default: %(default)s).",
    )
    parser.add_argument(
        "--background-subtract-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma used to estimate and subtract fluorescence background before scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--median-filter-size",
        type=int,
        default=0,
        help="Median filter window size applied to green/red projections to suppress grainy pixels (default: %(default)s). Use 0 or 1 to disable.",
    )
    parser.add_argument(
        "--preserve-edge-pixels",
        type=int,
        default=0,
        help="Keep this many pixels at each image edge unmodified after fluorescence preprocessing (default: %(default)s).",
    )
    parser.add_argument(
        "--invert-display",
        action="store_true",
        help="Invert the display mapping so low values are white and high values are dark.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("percentile", "full-range"),
        default="percentile",
        help="Display scaling mode. 'percentile' clips by display percentiles; 'full-range' uses the actual min/max without percentile clipping.",
    )
    parser.add_argument(
        "--display-gamma",
        type=float,
        default=1.0,
        help="Gamma applied after display normalization. Values below 1 brighten dim structure (default: %(default)s).",
    )
    parser.add_argument(
        "--scale-low-percentile",
        type=float,
        default=0.0,
        help="Lower percentile used for display scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--scale-high-percentile",
        type=float,
        default=99.5,
        help="Upper percentile used for display scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--no-scale-bars",
        action="store_true",
        help="Disable green/red scale bars in output strips.",
    )
    parser.add_argument(
        "--scale-bar-width",
        type=int,
        default=14,
        help="Scale bar width in pixels per labeled panel (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    preparer = BrightfieldConfocalVolumePreparer(
        input_dir=args.input_dir,
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
        cli_args=(argv if argv is not None else sys.argv[1:]),
    )
    preparer.prepare_all()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
