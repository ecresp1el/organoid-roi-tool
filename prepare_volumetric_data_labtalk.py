"""Prepare red/green Imaris volumes into 1x3 preview images.

This script scans a folder for ``.ims`` files, identifies red and green channels,
computes max-intensity projections, and exports a 1x3 RGB strip per file:

1. Red channel only
2. Green channel only
3. Merged red+green

Outputs default to ``~/Desktop/volumetric_labtalk_outputs``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import tifffile as tiff  # type: ignore

from imaris_tools import compute_max_projections, read_metadata


@dataclass(frozen=True)
class PreparedVolumeRecord:
    source_path: Path
    output_path: Path
    red_channel_index: int
    green_channel_index: int
    height_px: int
    width_px: int


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
    ) -> None:
        self.input_dir = input_dir.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.recursive = recursive
        self.overwrite = overwrite
        self.resolution_level = resolution_level
        self.time_point = time_point

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

        for ims_path in ims_files:
            red_idx, green_idx = self._resolve_red_green_channels(ims_path)
            if red_idx is None or green_idx is None:
                print(f"[warn] Skipping {ims_path.name}: could not identify both red and green channels.")
                continue

            _, projections = compute_max_projections(
                ims_path,
                resolution_level=self.resolution_level,
                time_point=self.time_point,
            )
            red_projection = projections[red_idx]
            green_projection = projections[green_idx]

            strip = self._compose_triptych(red_projection, green_projection)
            output_path = self.output_dir / f"{ims_path.stem}_red_green_merged.tif"

            if output_path.exists() and not self.overwrite:
                print(f"[warn] Output exists, skipping (use --overwrite to replace): {output_path}")
                continue

            tiff.imwrite(output_path, strip, photometric="rgb")
            records.append(
                PreparedVolumeRecord(
                    source_path=ims_path,
                    output_path=output_path,
                    red_channel_index=red_idx,
                    green_channel_index=green_idx,
                    height_px=int(strip.shape[0]),
                    width_px=int(strip.shape[1]),
                )
            )
            print(
                f"[ok] {ims_path.name}: red=Channel {red_idx}, green=Channel {green_idx} -> {output_path.name}"
            )

        manifest = self.output_dir / "prepared_manifest.csv"
        self._write_manifest(records, manifest)
        print(f"[info] Wrote manifest with {len(records)} record(s): {manifest}")
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
    def _channel_score(name: str, color_rgb: tuple[float, float, float], *, target: str) -> float:
        target = target.lower()
        clean_name = (name or "").lower()
        r, g, b = color_rgb

        if target == "red":
            score = (2.0 * r) - (g + b)
            keywords = ("red", "568", "594", "cy3", "tritc", "txred", "mcherry")
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

    @staticmethod
    def _normalize_u8(array: np.ndarray) -> np.ndarray:
        data = array.astype(np.float32, copy=False)
        max_val = float(np.max(data))
        if max_val <= 0:
            return np.zeros_like(data, dtype=np.uint8)
        return np.round((data / max_val) * 255.0).astype(np.uint8)

    def _compose_triptych(self, red_projection: np.ndarray, green_projection: np.ndarray) -> np.ndarray:
        red_u8 = self._normalize_u8(red_projection)
        green_u8 = self._normalize_u8(green_projection)

        red_rgb = np.zeros((*red_u8.shape, 3), dtype=np.uint8)
        green_rgb = np.zeros((*green_u8.shape, 3), dtype=np.uint8)
        merged_rgb = np.zeros((*red_u8.shape, 3), dtype=np.uint8)

        red_rgb[..., 0] = red_u8
        green_rgb[..., 1] = green_u8
        merged_rgb[..., 0] = red_u8
        merged_rgb[..., 1] = green_u8

        return np.concatenate([red_rgb, green_rgb, merged_rgb], axis=1)

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
    )
    preparer.prepare_all()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
