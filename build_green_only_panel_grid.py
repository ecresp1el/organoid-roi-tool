"""Build a 2xN green-only panel grid from raw max-projection TIFFs.

The grid uses one fixed display range for every tile so visual comparisons are
consistent across the full panel. Rows are grouped by filename prefix:

1. cl23
2. cl32
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff  # type: ignore
from PIL import Image, ImageDraw, ImageFont


ROW_ORDER = ("cl23", "cl32")


@dataclass(frozen=True)
class PanelEntry:
    row_key: str
    column_key: str
    label: str
    source_path: Path


class GreenOnlyPanelGridBuilder:
    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        *,
        display_min: float = 120.0,
        display_max: float = 180.0,
        tile_size: int = 256,
        gap: int = 16,
        margin: int = 20,
        scale_bar_width: int = 16,
        cli_args: Optional[list[str]] = None,
    ) -> None:
        self.input_dir = input_dir.expanduser().resolve()
        self.output_path = output_path.expanduser().resolve()
        self.display_min = float(display_min)
        self.display_max = float(display_max)
        self.tile_size = max(32, int(tile_size))
        self.gap = max(4, int(gap))
        self.margin = max(8, int(margin))
        self.scale_bar_width = max(8, int(scale_bar_width))
        if self.display_max <= self.display_min:
            raise ValueError("--display-max must be greater than --display-min.")
        self.cli_args = list(cli_args or [])

    def build(self) -> Path:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input folder does not exist or is not a directory: {self.input_dir}")

        entries = self._discover_entries()
        if not entries:
            raise FileNotFoundError(f"No '*_green_max_projection.tif' files found in {self.input_dir}")

        column_keys = sorted({entry.column_key for entry in entries}, key=self._column_sort_key)
        row_entries = {
            row_key: {entry.column_key: entry for entry in entries if entry.row_key == row_key}
            for row_key in ROW_ORDER
        }

        print("[info] ================================================")
        print(f"[info] Input directory: {self.input_dir}")
        print(f"[info] Output path: {self.output_path}")
        print(f"[info] Rows: {ROW_ORDER}")
        print(f"[info] Columns discovered: {len(column_keys)}")
        print(f"[info] Fixed display range: [{self.display_min:.3f}, {self.display_max:.3f}]")
        print("[info] ================================================")

        canvas = self._allocate_canvas(column_count=len(column_keys))
        self._draw_grid(
            canvas,
            row_entries=row_entries,
            column_keys=column_keys,
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[step] Writing panel TIFF: {self.output_path}")
        tiff.imwrite(self.output_path, canvas, photometric="rgb")

        metadata_path = self.output_path.with_suffix(".metadata.json")
        print(f"[step] Writing sidecar metadata: {metadata_path}")
        metadata_path.write_text(
            json.dumps(
                self._build_metadata(entries=entries, column_keys=column_keys),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[ok] Wrote panel grid: {self.output_path}")
        return self.output_path

    def _discover_entries(self) -> list[PanelEntry]:
        entries: list[PanelEntry] = []
        suffix = "_green_max_projection.tif"
        for path in sorted(self.input_dir.glob(f"*{suffix}")):
            stem = path.name[: -len(suffix)]
            if "_" not in stem:
                continue
            row_key, remainder = stem.split("_", 1)
            row_key = row_key.lower()
            if row_key not in ROW_ORDER:
                continue
            entries.append(
                PanelEntry(
                    row_key=row_key,
                    column_key=remainder,
                    label=self._simplify_label(remainder),
                    source_path=path,
                )
            )
        return entries

    def _allocate_canvas(self, *, column_count: int) -> np.ndarray:
        label_col_width = 96
        top_label_height = 52
        scale_bar_block = self.scale_bar_width + 84
        width = (
            self.margin
            + label_col_width
            + (column_count * self.tile_size)
            + ((column_count - 1) * self.gap)
            + self.gap
            + scale_bar_block
            + self.margin
        )
        height = (
            self.margin
            + top_label_height
            + (len(ROW_ORDER) * self.tile_size)
            + ((len(ROW_ORDER) - 1) * self.gap)
            + self.margin
        )
        return np.full((height, width, 3), 18, dtype=np.uint8)

    def _draw_grid(
        self,
        canvas: np.ndarray,
        *,
        row_entries: dict[str, dict[str, PanelEntry]],
        column_keys: list[str],
    ) -> None:
        label_col_width = 96
        top_label_height = 52
        x0 = self.margin + label_col_width
        y0 = self.margin + top_label_height

        image = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for col_index, column_key in enumerate(column_keys):
            x = x0 + col_index * (self.tile_size + self.gap)
            label = self._simplify_label(column_key)
            draw.text((x, self.margin + 6), label, fill=(255, 255, 255), font=font)

        for row_index, row_key in enumerate(ROW_ORDER):
            y = y0 + row_index * (self.tile_size + self.gap)
            draw.text((self.margin, y + (self.tile_size // 2) - 6), row_key.upper(), fill=(255, 255, 255), font=font)
            for col_index, column_key in enumerate(column_keys):
                x = x0 + col_index * (self.tile_size + self.gap)
                entry = row_entries[row_key].get(column_key)
                if entry is None:
                    draw.rectangle((x, y, x + self.tile_size - 1, y + self.tile_size - 1), outline=(90, 90, 90), width=1)
                    continue
                tile = self._load_tile(entry.source_path)
                tile_image = Image.fromarray(tile, mode="RGB").resize((self.tile_size, self.tile_size), Image.Resampling.NEAREST)
                image.paste(tile_image, (x, y))

        canvas[:, :, :] = np.asarray(image, dtype=np.uint8)
        self._append_scale_bar(canvas, x_start=x0 + len(column_keys) * (self.tile_size + self.gap))

    def _load_tile(self, path: Path) -> np.ndarray:
        raw = tiff.imread(path)
        if raw.ndim != 2:
            raise ValueError(f"Expected a 2D TIFF at {path}, got shape={raw.shape}")
        normalized = self._normalize_u8(raw.astype(np.float32, copy=False))
        tile = np.zeros((*normalized.shape, 3), dtype=np.uint8)
        tile[..., 1] = normalized
        return tile

    def _normalize_u8(self, data: np.ndarray) -> np.ndarray:
        clipped = np.clip(data, self.display_min, self.display_max)
        norm = (clipped - self.display_min) / (self.display_max - self.display_min)
        return np.round(norm * 255.0).astype(np.uint8)

    def _append_scale_bar(self, canvas: np.ndarray, *, x_start: int) -> None:
        height = (len(ROW_ORDER) * self.tile_size) + ((len(ROW_ORDER) - 1) * self.gap)
        bar_top = self.margin + 52
        gradient = np.linspace(255, 0, num=height, dtype=np.uint8)[:, None]
        bar = np.zeros((height, self.scale_bar_width, 3), dtype=np.uint8)
        bar[:, 1:-1, 1] = gradient
        bar[:, 0, :] = 255
        bar[:, -1, :] = 255
        bar[:2, :, :] = 255
        bar[-2:, :, :] = 255
        canvas[bar_top : bar_top + height, x_start : x_start + self.scale_bar_width, :] = bar

        label_region = canvas[bar_top : bar_top + height, x_start + self.scale_bar_width : x_start + self.scale_bar_width + 84, :]
        image = Image.fromarray(label_region, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((4, 4), "Green", fill=(255, 255, 255), font=font)
        draw.text((4, 18), "max", fill=(255, 255, 255), font=font)
        draw.text((4, 30), f"{self.display_max:.3g}", fill=(255, 255, 255), font=font)
        bottom_y = max(4, image.height - 26)
        draw.text((4, bottom_y - 12), "min", fill=(255, 255, 255), font=font)
        draw.text((4, bottom_y), f"{self.display_min:.3g}", fill=(255, 255, 255), font=font)
        canvas[bar_top : bar_top + height, x_start + self.scale_bar_width : x_start + self.scale_bar_width + 84, :] = np.asarray(image, dtype=np.uint8)

    def _build_metadata(self, *, entries: list[PanelEntry], column_keys: list[str]) -> dict[str, object]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable,
            "cli_args": self.cli_args,
            "input_dir": str(self.input_dir),
            "output_path": str(self.output_path),
            "display_range": {"min": self.display_min, "max": self.display_max},
            "rows": list(ROW_ORDER),
            "columns": column_keys,
            "entries": [
                {
                    "row_key": entry.row_key,
                    "column_key": entry.column_key,
                    "label": entry.label,
                    "source_path": str(entry.source_path),
                }
                for entry in entries
            ],
        }

    @staticmethod
    def _simplify_label(column_key: str) -> str:
        label = column_key
        label = label.replace("pv_reporter_", "")
        label = label.replace("_Confocal - Green_2026-03-09", "")
        label = label.replace("Confocal - Green_", "")
        label = label.replace("_", " ")
        label = re.sub(r"\s+", " ", label).strip()
        return label

    @staticmethod
    def _column_sort_key(column_key: str) -> tuple[int, str, int]:
        text = column_key.lower()
        condition_order = 0 if "igf" in text and "noigf" not in text else 1
        match = re.search(r"_(\d+)$", column_key)
        replicate = int(match.group(1)) if match else 0
        base = re.sub(r"_(\d+)$", "", column_key)
        return (condition_order, base, replicate)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 2xN cl23/cl32 green-only panel grid from raw max-projection TIFFs."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing '*_green_max_projection.tif' files.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path.home() / "Desktop" / "green_only_cl23_cl32_panel.tif",
        help="Output TIFF path for the panel grid (default: %(default)s).",
    )
    parser.add_argument("--display-min", type=float, default=120.0)
    parser.add_argument("--display-max", type=float, default=180.0)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--gap", type=int, default=16)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--scale-bar-width", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    builder = GreenOnlyPanelGridBuilder(
        input_dir=args.input_dir,
        output_path=args.output_path,
        display_min=args.display_min,
        display_max=args.display_max,
        tile_size=args.tile_size,
        gap=args.gap,
        margin=args.margin,
        scale_bar_width=args.scale_bar_width,
        cli_args=(argv if argv is not None else sys.argv[1:]),
    )
    builder.build()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
