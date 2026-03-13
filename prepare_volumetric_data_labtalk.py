"""Prepare red/green Imaris volumes into 1x3 preview images.

This script scans a folder for ``.ims`` files, identifies red and green channels,
computes max-intensity projections, and exports a 1x3 RGB strip per file:

1. Red channel only
2. Green channel only
3. Merged red+green

By default, each panel includes a per-panel intensity scale bar.
Outputs default to ``~/Desktop/volumetric_labtalk_outputs``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import h5py  # type: ignore
import numpy as np
import tifffile as tiff  # type: ignore

from imaris_tools.metadata import _discover_channels, read_metadata
from imaris_tools.projections import compute_max_projections


@dataclass(frozen=True)
class PreparedVolumeRecord:
    source_path: Path
    output_path: Path
    red_channel_index: int
    green_channel_index: int
    height_px: int
    width_px: int
    red_min_value: float
    red_max_value: float
    green_min_value: float
    green_max_value: float


class VolumetricDataLabtalkPreparer:
    """Object-oriented helper for preparing red/green overview strips from ``.ims`` files."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        recursive: bool = False,
        overwrite: bool = False,
        resolution_level: int = 0,
        time_point: int = 0,
        background_subtract_sigma: float = 12.0,
        smoothing_sigma: float = 0.6,
        scale_low_percentile: float = 1.0,
        scale_high_percentile: float = 99.8,
        include_scale_bars: bool = True,
        scale_bar_width: int = 14,
    ) -> None:
        self.input_dir = input_dir.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.recursive = recursive
        self.overwrite = overwrite
        self.resolution_level = resolution_level
        self.time_point = time_point
        self.background_subtract_sigma = max(0.0, background_subtract_sigma)
        self.smoothing_sigma = max(0.0, smoothing_sigma)
        self.scale_low_percentile = scale_low_percentile
        self.scale_high_percentile = scale_high_percentile
        self.include_scale_bars = include_scale_bars
        self.scale_bar_width = max(2, scale_bar_width)

    def discover_files(self) -> list[Path]:
        iterator: Iterable[Path]
        if self.recursive:
            iterator = self.input_dir.rglob("*.ims")
        else:
            iterator = self.input_dir.glob("*.ims")
        return sorted(path for path in iterator if path.is_file())

    def prepare_all(self) -> list[PreparedVolumeRecord]:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input folder does not exist or is not a directory: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ims_files = self.discover_files()
        records: list[PreparedVolumeRecord] = []

        print("[info] ================================================")
        print(f"[info] Input directory: {self.input_dir}")
        print(f"[info] Output directory: {self.output_dir}")
        print(f"[info] .ims files discovered: {len(ims_files)}")
        print(f"[info] Recursive search: {self.recursive}")
        print(f"[info] Resolution level: {self.resolution_level}, time point: {self.time_point}")
        print(
            "[info] Preprocessing: "
            f"background sigma={self.background_subtract_sigma:.2f}, "
            f"smoothing sigma={self.smoothing_sigma:.2f}"
        )
        print(
            "[info] Display scaling percentiles: "
            f"{self.scale_low_percentile:.2f} -> {self.scale_high_percentile:.2f}"
        )
        print(f"[info] Include per-panel scale bars: {self.include_scale_bars}")
        print("[info] ================================================")

        for file_index, ims_path in enumerate(ims_files, start=1):
            print(f"\n[info] Processing file {file_index}/{len(ims_files)}: {ims_path.name}")
            red_idx, green_idx = self._resolve_red_green_channels(ims_path)
            if red_idx is None or green_idx is None:
                print(f"[warn] Skipping {ims_path.name}: could not identify both red and green channels.")
                continue

            output_path = self.output_dir / f"{ims_path.stem}_red_green_merged.tif"
            if output_path.exists() and not self.overwrite:
                print(f"[warn] Output exists, skipping (use --overwrite to replace): {output_path}")
                continue

            metadata, projections = compute_max_projections(
                ims_path,
                resolution_level=self.resolution_level,
                time_point=self.time_point,
            )
            red_projection = projections[red_idx]
            green_projection = projections[green_idx]

            red_channel_name = self._channel_name(metadata, red_idx)
            green_channel_name = self._channel_name(metadata, green_idx)
            print(f"[info] Selected channels -> red: {red_idx} ({red_channel_name}), green: {green_idx} ({green_channel_name})")
            self._print_projection_proof(
                ims_path,
                metadata=metadata,
                projections=projections,
                channel_indices=(red_idx, green_idx),
            )

            red_processed = self._preprocess_projection(red_projection, channel_label=f"red ch{red_idx}")
            green_processed = self._preprocess_projection(green_projection, channel_label=f"green ch{green_idx}")
            strip, red_scale, green_scale = self._compose_triptych(red_processed, green_processed)
            if self.include_scale_bars:
                strip = self._append_scale_bars(strip, red_scale, green_scale)

            tiff.imwrite(output_path, strip, photometric="rgb")
            record = PreparedVolumeRecord(
                source_path=ims_path,
                output_path=output_path,
                red_channel_index=red_idx,
                green_channel_index=green_idx,
                height_px=int(strip.shape[0]),
                width_px=int(strip.shape[1]),
                red_min_value=red_scale[0],
                red_max_value=red_scale[1],
                green_min_value=green_scale[0],
                green_max_value=green_scale[1],
            )
            records.append(record)
            print(
                "[ok] "
                f"{ims_path.name} -> {output_path.name} | "
                f"shape={strip.shape} | "
                f"red scale=[{red_scale[0]:.3f}, {red_scale[1]:.3f}] | "
                f"green scale=[{green_scale[0]:.3f}, {green_scale[1]:.3f}]"
            )

        manifest = self.output_dir / "prepared_manifest.csv"
        self._write_manifest(records, manifest)
        print(f"\n[info] Wrote manifest with {len(records)} record(s): {manifest}")
        return records

    def _resolve_red_green_channels(self, ims_path: Path) -> tuple[Optional[int], Optional[int]]:
        metadata = read_metadata(
            ims_path,
            resolution_level=self.resolution_level,
            time_point=self.time_point,
        )
        if not metadata.channels:
            return None, None

        red_ranked = sorted(
            metadata.channels,
            key=lambda channel: self._channel_score(channel.name, channel.color_rgb, target="red"),
            reverse=True,
        )
        green_ranked = sorted(
            metadata.channels,
            key=lambda channel: self._channel_score(channel.name, channel.color_rgb, target="green"),
            reverse=True,
        )

        red_idx = red_ranked[0].index if red_ranked else None
        green_idx = next((ch.index for ch in green_ranked if ch.index != red_idx), None)

        if red_idx is not None and green_idx is not None:
            return red_idx, green_idx
        return None, None

    @staticmethod
    def _channel_name(metadata: object, channel_index: int) -> str:
        channels = getattr(metadata, "channels", [])
        for channel in channels:
            if getattr(channel, "index", None) == channel_index:
                return getattr(channel, "name", f"Channel {channel_index}")
        return f"Channel {channel_index}"

    @staticmethod
    def _channel_score(name: str, color_rgb: tuple[float, float, float], *, target: str) -> float:
        target = target.lower()
        clean_name = (name or "").lower()
        r, g, b = color_rgb

        if target == "red":
            score = (2.0 * r) - (g + b)
            keywords = ("red", "568", "594", "cy3", "tritc", "txred", "mcherry", "rfp")
        elif target == "green":
            score = (2.0 * g) - (r + b)
            keywords = ("green", "488", "gfp", "fitc", "alexa488")
        else:
            raise ValueError(f"Unknown target color: {target}")

        for keyword in keywords:
            if keyword in clean_name:
                score += 2.0
                break

        return score

    def _normalize_u8(self, array: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        data = array.astype(np.float32, copy=False)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return np.zeros_like(data, dtype=np.uint8), (0.0, 0.0)

        vmin = float(np.percentile(finite, self.scale_low_percentile))
        vmax = float(np.percentile(finite, self.scale_high_percentile))
        if vmax <= vmin:
            vmax = float(np.max(finite))
            vmin = float(np.min(finite))
        if vmax <= vmin:
            return np.zeros_like(data, dtype=np.uint8), (vmin, vmax)

        clipped = np.clip(data, vmin, vmax)
        norm = (clipped - vmin) / (vmax - vmin)
        return np.round(norm * 255.0).astype(np.uint8), (vmin, vmax)

    def _preprocess_projection(self, array: np.ndarray, *, channel_label: str) -> np.ndarray:
        from scipy.ndimage import gaussian_filter  # type: ignore

        data = array.astype(np.float32, copy=False)
        processed = np.array(data, copy=True)
        raw_min = float(np.min(processed))
        raw_max = float(np.max(processed))

        if self.background_subtract_sigma > 0:
            background = gaussian_filter(processed, sigma=self.background_subtract_sigma)
            processed = np.clip(processed - background, 0.0, None)

        if self.smoothing_sigma > 0:
            processed = gaussian_filter(processed, sigma=self.smoothing_sigma)

        print(
            "[info] Preprocessed "
            f"{channel_label}: raw_range=[{raw_min:.3f}, {raw_max:.3f}] -> "
            f"processed_range=[{float(np.min(processed)):.3f}, {float(np.max(processed)):.3f}]"
        )
        return processed

    def _compose_triptych(
        self,
        red_projection: np.ndarray,
        green_projection: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        red_u8, red_scale = self._normalize_u8(red_projection)
        green_u8, green_scale = self._normalize_u8(green_projection)

        red_rgb = np.zeros((*red_u8.shape, 3), dtype=np.uint8)
        green_rgb = np.zeros((*green_u8.shape, 3), dtype=np.uint8)
        merged_rgb = np.zeros((*red_u8.shape, 3), dtype=np.uint8)

        red_rgb[..., 0] = red_u8
        green_rgb[..., 1] = green_u8
        merged_rgb[..., 0] = red_u8
        merged_rgb[..., 1] = green_u8

        strip = np.concatenate([red_rgb, green_rgb, merged_rgb], axis=1)
        return strip, red_scale, green_scale

    def _print_projection_proof(
        self,
        ims_path: Path,
        *,
        metadata: object,
        projections: dict[int, np.ndarray],
        channel_indices: tuple[int, int],
    ) -> None:
        with h5py.File(ims_path, "r") as handle:
            datasets = dict(
                _discover_channels(
                    handle,
                    resolution_level=self.resolution_level,
                    time_point=self.time_point,
                )
            )

            for channel_index in channel_indices:
                dataset = datasets.get(channel_index)
                projection = projections.get(channel_index)
                if dataset is None or projection is None:
                    continue

                channel_name = self._channel_name(metadata, channel_index)
                proof = self._projection_proof_stats(dataset, projection)
                print(
                    "[proof] "
                    f"Channel {channel_index} ({channel_name}) raw shape={proof['raw_shape']} "
                    f"-> z_frames={proof['z_frames']} -> plotted projection shape={proof['projection_shape']}"
                )
                print(
                    "[proof] "
                    f"Channel {channel_index} slice maxima across z: "
                    f"min={proof['slice_max_min']:.3f}, median={proof['slice_max_median']:.3f}, "
                    f"max={proof['slice_max_max']:.3f}"
                )
                print(
                    "[proof] "
                    f"Channel {channel_index} z0 range=[{proof['first_slice_min']:.3f}, {proof['first_slice_max']:.3f}] "
                    f"-> max projection range=[{proof['projection_min']:.3f}, {proof['projection_max']:.3f}]"
                )
                print(
                    "[proof] "
                    f"Channel {channel_index} pixels increased beyond z0 due to later z slices: "
                    f"{proof['improved_pixels']}/{proof['pixel_count']} "
                    f"({proof['improved_fraction']:.2%}), max_delta={proof['max_delta_vs_first']:.3f}"
                )

    @staticmethod
    def _projection_proof_stats(dataset: h5py.Dataset, projection: np.ndarray) -> dict[str, object]:
        if dataset.ndim < 3:
            first_slice = np.asarray(dataset[()], dtype=np.float32)
            slice_maxima = np.array([float(np.max(first_slice))], dtype=np.float32)
            z_frames = 1
        else:
            z_frames = int(dataset.shape[0])
            first_slice = np.asarray(dataset[0, ...], dtype=np.float32)
            slice_maxima = np.empty(z_frames, dtype=np.float32)
            for z_index in range(z_frames):
                slice_data = np.asarray(dataset[z_index, ...], dtype=np.float32)
                slice_maxima[z_index] = float(np.max(slice_data))

        projection_f32 = projection.astype(np.float32, copy=False)
        delta_vs_first = projection_f32 - first_slice
        improved_mask = delta_vs_first > 0
        improved_pixels = int(np.count_nonzero(improved_mask))
        pixel_count = int(projection_f32.size)
        max_delta = float(np.max(delta_vs_first[improved_mask])) if improved_pixels else 0.0

        return {
            "raw_shape": tuple(int(dim) for dim in dataset.shape),
            "z_frames": z_frames,
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
        red_scale: tuple[float, float],
        green_scale: tuple[float, float],
    ) -> np.ndarray:
        height, width, _ = strip.shape
        panel_width = width // 3
        gap = 8
        bar = max(12, self.scale_bar_width)

        out = np.full((height, width + 3 * (gap + bar), 3), 18, dtype=np.uint8)
        cursor = 0

        gradient = np.linspace(255, 0, num=height, dtype=np.uint8)[:, None]

        red_panel = strip[:, :panel_width, :]
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

        merged_panel = strip[:, (2 * panel_width) :, :]
        out[:, cursor : cursor + panel_width, :] = merged_panel
        cursor += panel_width
        out[:, cursor : cursor + gap, :] = 32
        cursor += gap
        merged_bar = np.zeros((height, bar, 3), dtype=np.uint8)
        merged_bar[:, 1:-1, 0] = gradient
        merged_bar[:, 1:-1, 1] = gradient
        merged_bar[:, 0, :] = 255
        merged_bar[:, -1, :] = 255
        merged_bar[:2, :, :] = 255
        merged_bar[-2:, :, :] = 255
        out[:, cursor : cursor + bar, :] = merged_bar

        print(
            "[info] Added scale bars | "
            f"red={red_scale[0]:.3f}->{red_scale[1]:.3f}, "
            f"green={green_scale[0]:.3f}->{green_scale[1]:.3f}, "
            "merged uses both channel scales"
        )
        return out

    @staticmethod
    def _write_manifest(records: list[PreparedVolumeRecord], path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_path",
                    "output_path",
                    "red_channel_index",
                    "green_channel_index",
                    "height_px",
                    "width_px",
                    "red_min_value",
                    "red_max_value",
                    "green_min_value",
                    "green_max_value",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "source_path": str(record.source_path),
                        "output_path": str(record.output_path),
                        "red_channel_index": record.red_channel_index,
                        "green_channel_index": record.green_channel_index,
                        "height_px": record.height_px,
                        "width_px": record.width_px,
                        "red_min_value": f"{record.red_min_value:.6g}",
                        "red_max_value": f"{record.red_max_value:.6g}",
                        "green_min_value": f"{record.green_min_value:.6g}",
                        "green_max_value": f"{record.green_max_value:.6g}",
                    }
                )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 1x3 red/green/merged strips from Imaris files.")
    parser.add_argument("input_dir", type=Path, help="Directory containing .ims files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Desktop" / "volumetric_labtalk_outputs",
        help="Output directory for prepared strips (default: %(default)s).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .ims files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--resolution-level", type=int, default=0)
    parser.add_argument("--time-point", type=int, default=0)
    parser.add_argument(
        "--background-subtract-sigma",
        type=float,
        default=12.0,
        help="Gaussian sigma used to estimate and subtract background before scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=0.6,
        help="Small Gaussian smoothing sigma applied after background subtraction (default: %(default)s).",
    )
    parser.add_argument(
        "--scale-low-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used for display scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--scale-high-percentile",
        type=float,
        default=99.8,
        help="Upper percentile used for display scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--no-scale-bars",
        action="store_true",
        help="Disable per-panel scale bars in output strips.",
    )
    parser.add_argument(
        "--scale-bar-width",
        type=int,
        default=14,
        help="Scale bar width in pixels per panel (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    preparer = VolumetricDataLabtalkPreparer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        overwrite=args.overwrite,
        resolution_level=args.resolution_level,
        time_point=args.time_point,
        background_subtract_sigma=args.background_subtract_sigma,
        smoothing_sigma=args.smoothing_sigma,
        scale_low_percentile=args.scale_low_percentile,
        scale_high_percentile=args.scale_high_percentile,
        include_scale_bars=not args.no_scale_bars,
        scale_bar_width=args.scale_bar_width,
    )
    preparer.prepare_all()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
