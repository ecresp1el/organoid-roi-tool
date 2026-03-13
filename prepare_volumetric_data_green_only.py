"""Prepare green-only Imaris volumes into raw or display-scaled max-projection TIFFs.

This script accepts either a single ``.ims`` file or a directory containing
``.ims`` files. It resolves a single green channel, computes its max-intensity
projection across z, and writes one 2D TIFF per input file.

Outputs can be either raw max projections or display-mapped green visualization
TIFFs with a labeled scale bar. Provenance is written alongside each TIFF as
JSON plus a CSV manifest for the whole run.
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
from PIL import Image, ImageDraw, ImageFont

from imaris_tools.metadata import _discover_channels, read_metadata

DEFAULT_GREEN_CHANNEL_NAME = "Confocal - Green"


@dataclass(frozen=True)
class PreparedGreenProjectionRecord:
    source_path: Path
    output_path: Path
    output_metadata_path: Path
    green_channel_index: int
    height_px: int
    width_px: int
    raw_dtype: str
    green_min_value: float
    green_max_value: float
    output_mode: str
    display_min_value: float
    display_max_value: float


class GreenOnlyVolumePreparer:
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        recursive: bool = False,
        overwrite: bool = False,
        resolution_level: int = 0,
        time_point: int = 0,
        green_channel_name: str = DEFAULT_GREEN_CHANNEL_NAME,
        output_mode: str = "visualization",
        scale_mode: str = "percentile",
        display_gamma: float = 1.0,
        scale_low_percentile: float = 0.0,
        scale_high_percentile: float = 99.5,
        display_min: Optional[float] = None,
        display_max: Optional[float] = None,
        include_scale_bar: bool = True,
        scale_bar_width: int = 14,
        cli_args: Optional[list[str]] = None,
    ) -> None:
        self.input_path = input_path.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.recursive = recursive
        self.overwrite = overwrite
        self.resolution_level = int(resolution_level)
        self.time_point = int(time_point)
        self.green_channel_name = green_channel_name.strip()
        self.output_mode = output_mode
        self.scale_mode = scale_mode
        self.display_gamma = max(1e-6, float(display_gamma))
        self.scale_low_percentile = float(scale_low_percentile)
        self.scale_high_percentile = float(scale_high_percentile)
        self.display_min = None if display_min is None else float(display_min)
        self.display_max = None if display_max is None else float(display_max)
        if (self.display_min is None) != (self.display_max is None):
            raise ValueError("Use --display-min and --display-max together.")
        if self.display_min is not None and self.display_max is not None and self.display_max <= self.display_min:
            raise ValueError("--display-max must be greater than --display-min.")
        self.include_scale_bar = include_scale_bar
        self.scale_bar_width = max(2, int(scale_bar_width))
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

    def prepare_all(self) -> list[PreparedGreenProjectionRecord]:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        if not self.input_path.is_dir() and not self.input_path.is_file():
            raise FileNotFoundError(f"Input path is neither a directory nor a file: {self.input_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ims_files = self.discover_files()
        records: list[PreparedGreenProjectionRecord] = []
        run_metadata_path = self.output_dir / "preparation_run_metadata.json"
        self._write_json(run_metadata_path, self._build_run_metadata(ims_files))

        print("[info] ================================================")
        print(f"[info] Input path: {self.input_path}")
        print(f"[info] Output directory: {self.output_dir}")
        print(f"[info] .ims files discovered: {len(ims_files)}")
        print(f"[info] Recursive search: {self.recursive}")
        print(f"[info] Resolution level: {self.resolution_level}, time point: {self.time_point}")
        print(f"[info] Green channel name: '{self.green_channel_name}'")
        print(
            "[info] Output mode: "
            f"{self.output_mode} | scale mode={self.scale_mode}, gamma={self.display_gamma:.3f}, "
            f"percentiles={self.scale_low_percentile:.2f}->{self.scale_high_percentile:.2f}, "
            f"include scale bar={self.include_scale_bar}"
        )
        if self.display_min is not None and self.display_max is not None:
            print(f"[info] Fixed display range override: [{self.display_min:.3f}, {self.display_max:.3f}]")
        print(f"[info] Run metadata JSON: {run_metadata_path}")
        print("[info] ================================================")

        for file_index, ims_path in enumerate(ims_files, start=1):
            print(f"\n[info] Processing file {file_index}/{len(ims_files)}: {ims_path.name}")
            print(f"[step] Reading metadata: {ims_path}")
            metadata = read_metadata(
                ims_path,
                resolution_level=self.resolution_level,
                time_point=self.time_point,
            )
            green_channel = self._resolve_green_channel(metadata)
            if green_channel is None:
                print(
                    "[warn] "
                    f"Skipping {ims_path.name}: green channel '{self.green_channel_name}' was not found."
                )
                continue

            output_suffix = "_green_max_projection.tif"
            if self.output_mode == "visualization":
                output_suffix = "_green_max_projection_visualized.tif"
            output_path = self.output_dir / f"{ims_path.stem}{output_suffix}"
            output_metadata_path = output_path.with_suffix(".metadata.json")
            if output_path.exists() and not self.overwrite:
                print(f"[warn] Output exists, skipping (use --overwrite to replace): {output_path}")
                continue

            print(
                f"[step] Resolved green channel: index={green_channel.index} "
                f"name='{green_channel.name}'"
            )
            print("[step] Opening Imaris file and computing max projection")
            with h5py.File(ims_path, "r") as handle:
                datasets = dict(
                    _discover_channels(
                        handle,
                        resolution_level=self.resolution_level,
                        time_point=self.time_point,
                    )
                )
                dataset = datasets.get(int(green_channel.index))
                if dataset is None:
                    print(f"[warn] Skipping {ims_path.name}: dataset for channel {green_channel.index} is missing.")
                    continue
                projection = self._compute_max_projection(
                    dataset,
                    channel_label=f"green ch{green_channel.index}",
                )

            display_scale = self._resolve_display_scale(projection)
            output_array = projection
            if self.output_mode == "visualization":
                print(
                    f"[step] Rendering visualization with display range "
                    f"[{display_scale[0]:.3f}, {display_scale[1]:.3f}]"
                )
                output_array = self._render_visualization(
                    projection,
                    display_scale=display_scale,
                )

            print(f"[step] Writing TIFF output: {output_path}")
            if self.output_mode == "visualization":
                tiff.imwrite(output_path, output_array, photometric="rgb")
            else:
                tiff.imwrite(output_path, output_array)
            print(f"[step] Writing sidecar metadata: {output_metadata_path}")
            self._write_json(
                output_metadata_path,
                self._build_image_metadata(
                    ims_path=ims_path,
                    output_path=output_path,
                    output_metadata_path=output_metadata_path,
                    run_metadata_path=run_metadata_path,
                    green_channel_index=int(green_channel.index),
                    green_channel_name=str(green_channel.name),
                    projection=projection,
                    display_scale=display_scale,
                ),
            )

            record = PreparedGreenProjectionRecord(
                source_path=ims_path,
                output_path=output_path,
                output_metadata_path=output_metadata_path,
                green_channel_index=int(green_channel.index),
                height_px=int(projection.shape[0]),
                width_px=int(projection.shape[1]),
                raw_dtype=str(projection.dtype),
                green_min_value=float(np.min(projection)),
                green_max_value=float(np.max(projection)),
                output_mode=self.output_mode,
                display_min_value=float(display_scale[0]),
                display_max_value=float(display_scale[1]),
            )
            records.append(record)
            print(
                "[ok] "
                f"{ims_path.name} -> {output_path.name} | "
                f"shape={projection.shape} | dtype={projection.dtype} | "
                f"range=[{float(np.min(projection)):.3f}, {float(np.max(projection)):.3f}] | "
                f"display=[{display_scale[0]:.3f}, {display_scale[1]:.3f}]"
            )

        manifest_path = self.output_dir / "prepared_manifest.csv"
        print(f"[step] Writing manifest CSV: {manifest_path}")
        self._write_manifest(records, manifest_path, run_metadata_path=run_metadata_path)
        print(f"[info] Wrote manifest with {len(records)} record(s): {manifest_path}")
        return records

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
            "settings": {
                "recursive": self.recursive,
                "overwrite": self.overwrite,
                "resolution_level": self.resolution_level,
                "time_point": self.time_point,
                "green_channel_name": self.green_channel_name,
                "output_mode": self.output_mode,
                "scale_mode": self.scale_mode,
                "display_gamma": self.display_gamma,
                "scale_low_percentile": self.scale_low_percentile,
                "scale_high_percentile": self.scale_high_percentile,
                "display_min": self.display_min,
                "display_max": self.display_max,
                "include_scale_bar": self.include_scale_bar,
                "scale_bar_width": self.scale_bar_width,
            },
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
        green_channel_index: int,
        green_channel_name: str,
        projection: np.ndarray,
        display_scale: tuple[float, float],
    ) -> dict[str, object]:
        return {
            "source_path": str(ims_path),
            "output_path": str(output_path),
            "output_metadata_path": str(output_metadata_path),
            "run_metadata_path": str(run_metadata_path),
            "channel": {
                "index": green_channel_index,
                "name": green_channel_name,
            },
            "projection": {
                "shape": [int(dim) for dim in projection.shape],
                "dtype": str(projection.dtype),
                "min": float(np.min(projection)),
                "max": float(np.max(projection)),
            },
            "display": {
                "mode": self.output_mode,
                "scale_mode": self.scale_mode,
                "display_gamma": self.display_gamma,
                "range_min": float(display_scale[0]),
                "range_max": float(display_scale[1]),
                "include_scale_bar": self.include_scale_bar,
            },
        }

    def _resolve_green_channel(self, metadata: object):
        channels = getattr(metadata, "channels", [])
        if len(channels) == 1:
            return channels[0]
        lookup = {self._normalize_channel_name(getattr(channel, "name", "")): channel for channel in channels}
        return lookup.get(self._normalize_channel_name(self.green_channel_name))

    @staticmethod
    def _normalize_channel_name(name: str) -> str:
        return " ".join((name or "").strip().lower().split())

    @staticmethod
    def _compute_max_projection(dataset: h5py.Dataset, *, channel_label: str) -> np.ndarray:
        if dataset.ndim < 3:
            data = np.asarray(dataset[()], dtype=dataset.dtype)
            if data.ndim != 2:
                raise ValueError(f"{channel_label}: expected a 2D or 3D dataset")
            print(f"[step] {channel_label}: dataset is already 2D")
            return data

        total_z = int(dataset.shape[0])
        print(f"[step] {channel_label}: computing max projection across total_z={total_z}")
        projection = np.asarray(dataset[0, ...], dtype=dataset.dtype)
        for z_index in range(1, total_z):
            np.maximum(projection, dataset[z_index, ...], out=projection)
            if z_index == 1 or z_index % 10 == 0 or z_index == total_z - 1:
                print(f"[progress] {channel_label}: processed {z_index + 1}/{total_z} z slices")
        return np.array(projection)

    def _resolve_display_scale(self, projection: np.ndarray) -> tuple[float, float]:
        if self.display_min is not None and self.display_max is not None:
            return self.display_min, self.display_max
        finite = projection[np.isfinite(projection)]
        if finite.size == 0:
            return 0.0, 0.0
        if self.scale_mode == "full-range":
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            vmin = float(np.percentile(finite, self.scale_low_percentile))
            vmax = float(np.percentile(finite, self.scale_high_percentile))
        if vmax <= vmin:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        return vmin, vmax

    def _render_visualization(
        self,
        projection: np.ndarray,
        *,
        display_scale: tuple[float, float],
    ) -> np.ndarray:
        normalized = self._normalize_u8(projection, display_scale)
        green_rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
        green_rgb[..., 1] = normalized
        if not self.include_scale_bar:
            return green_rgb
        return self._append_scale_bar(green_rgb, display_scale=display_scale)

    def _normalize_u8(self, projection: np.ndarray, display_scale: tuple[float, float]) -> np.ndarray:
        vmin, vmax = display_scale
        if vmax <= vmin:
            return np.zeros_like(projection, dtype=np.uint8)
        clipped = np.clip(projection.astype(np.float32, copy=False), vmin, vmax)
        norm = (clipped - vmin) / (vmax - vmin)
        if self.display_gamma != 1.0:
            norm = np.power(norm, self.display_gamma)
        return np.round(norm * 255.0).astype(np.uint8)

    def _append_scale_bar(
        self,
        image_rgb: np.ndarray,
        *,
        display_scale: tuple[float, float],
    ) -> np.ndarray:
        height, width, _ = image_rgb.shape
        gap = 8
        bar = max(12, self.scale_bar_width)
        label_width = 92
        out = np.full((height, width + gap + bar + label_width, 3), 18, dtype=np.uint8)
        out[:, :width, :] = image_rgb
        cursor = width
        out[:, cursor : cursor + gap, :] = 32
        cursor += gap
        gradient = np.linspace(255, 0, num=height, dtype=np.uint8)[:, None]
        scale_bar = np.zeros((height, bar, 3), dtype=np.uint8)
        scale_bar[:, 1:-1, 1] = gradient
        scale_bar[:, 0, :] = 255
        scale_bar[:, -1, :] = 255
        scale_bar[:2, :, :] = 255
        scale_bar[-2:, :, :] = 255
        out[:, cursor : cursor + bar, :] = scale_bar
        cursor += bar
        self._draw_scale_labels(out, x0=cursor, width=label_width, display_scale=display_scale)
        return out

    @staticmethod
    def _draw_scale_labels(
        canvas: np.ndarray,
        *,
        x0: int,
        width: int,
        display_scale: tuple[float, float],
    ) -> None:
        label_region = canvas[:, x0 : x0 + width, :]
        image = Image.fromarray(label_region, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        max_text = f"{display_scale[1]:.3g}"
        min_text = f"{display_scale[0]:.3g}"
        draw.text((4, 4), "Green", fill=(255, 255, 255), font=font)
        draw.text((4, 18), "max", fill=(255, 255, 255), font=font)
        draw.text((4, 30), max_text, fill=(255, 255, 255), font=font)
        bottom_y = max(4, image.height - 26)
        draw.text((4, bottom_y - 12), "min", fill=(255, 255, 255), font=font)
        draw.text((4, bottom_y), min_text, fill=(255, 255, 255), font=font)
        canvas[:, x0 : x0 + width, :] = np.asarray(image, dtype=np.uint8)

    def _write_manifest(
        self,
        records: list[PreparedGreenProjectionRecord],
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
                    "green_channel_index",
                    "height_px",
                    "width_px",
                    "raw_dtype",
                    "green_min_value",
                    "green_max_value",
                    "output_mode",
                    "display_min_value",
                    "display_max_value",
                    "recursive",
                    "overwrite",
                    "resolution_level",
                    "time_point",
                    "green_channel_name",
                    "scale_mode",
                    "display_gamma",
                    "scale_low_percentile",
                    "scale_high_percentile",
                    "include_scale_bar",
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
                        "green_channel_index": record.green_channel_index,
                        "height_px": record.height_px,
                        "width_px": record.width_px,
                        "raw_dtype": record.raw_dtype,
                        "green_min_value": f"{record.green_min_value:.6g}",
                        "green_max_value": f"{record.green_max_value:.6g}",
                        "output_mode": record.output_mode,
                        "display_min_value": f"{record.display_min_value:.6g}",
                        "display_max_value": f"{record.display_max_value:.6g}",
                        "recursive": self.recursive,
                        "overwrite": self.overwrite,
                        "resolution_level": self.resolution_level,
                        "time_point": self.time_point,
                        "green_channel_name": self.green_channel_name,
                        "scale_mode": self.scale_mode,
                        "display_gamma": f"{self.display_gamma:.6g}",
                        "scale_low_percentile": f"{self.scale_low_percentile:.6g}",
                        "scale_high_percentile": f"{self.scale_high_percentile:.6g}",
                        "include_scale_bar": self.include_scale_bar,
                        "scale_bar_width": self.scale_bar_width,
                    }
                )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw green-channel max projections from one .ims file or a directory of .ims files."
    )
    parser.add_argument("input_path", type=Path, help="Path to a single .ims file or a directory containing .ims files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Desktop" / "volumetric_green_only_outputs",
        help="Output directory for green max-projection TIFFs (default: %(default)s).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .ims files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--resolution-level", type=int, default=0)
    parser.add_argument("--time-point", type=int, default=0)
    parser.add_argument(
        "--green-channel-name",
        default=DEFAULT_GREEN_CHANNEL_NAME,
        help="Exact green channel name when multiple channels exist (default: %(default)s).",
    )
    parser.add_argument(
        "--output-mode",
        choices=("visualization", "raw"),
        default="visualization",
        help="Write a display-scaled green visualization TIFF or the raw max-projection TIFF (default: %(default)s).",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("percentile", "full-range"),
        default="percentile",
        help="Display scaling mode for visualization output (default: %(default)s).",
    )
    parser.add_argument("--display-gamma", type=float, default=1.0)
    parser.add_argument("--scale-low-percentile", type=float, default=0.0)
    parser.add_argument("--scale-high-percentile", type=float, default=99.5)
    parser.add_argument("--display-min", type=float, default=None, help="Fixed display minimum for visualization output.")
    parser.add_argument("--display-max", type=float, default=None, help="Fixed display maximum for visualization output.")
    parser.add_argument("--no-scale-bar", action="store_true", help="Disable the green scale bar and numeric labels.")
    parser.add_argument("--scale-bar-width", type=int, default=14)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    preparer = GreenOnlyVolumePreparer(
        input_path=args.input_path,
        output_dir=args.output_dir,
        recursive=args.recursive,
        overwrite=args.overwrite,
        resolution_level=args.resolution_level,
        time_point=args.time_point,
        green_channel_name=args.green_channel_name,
        output_mode=args.output_mode,
        scale_mode=args.scale_mode,
        display_gamma=args.display_gamma,
        scale_low_percentile=args.scale_low_percentile,
        scale_high_percentile=args.scale_high_percentile,
        display_min=args.display_min,
        display_max=args.display_max,
        include_scale_bar=not args.no_scale_bar,
        scale_bar_width=args.scale_bar_width,
        cli_args=(argv if argv is not None else sys.argv[1:]),
    )
    preparer.prepare_all()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
