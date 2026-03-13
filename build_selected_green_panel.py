"""Build a curated 2xN panel from selected green visualization TIFFs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff  # type: ignore
from PIL import Image, ImageDraw, ImageFont


class SelectedGreenPanelBuilder:
    def __init__(
        self,
        row1_paths: list[Path],
        row2_paths: list[Path],
        output_path: Path,
        *,
        row1_label: str = "CL23",
        row2_label: str = "CL32",
        title: str = "IGF Green Max Projection",
        crop_scale_bar: bool = True,
        tile_width: int = 300,
        gap: int = 18,
        margin: int = 24,
        cli_args: Optional[list[str]] = None,
    ) -> None:
        self.row1_paths = [path.expanduser().resolve() for path in row1_paths]
        self.row2_paths = [path.expanduser().resolve() for path in row2_paths]
        self.output_path = output_path.expanduser().resolve()
        self.row1_label = row1_label
        self.row2_label = row2_label
        self.title = title
        self.crop_scale_bar = crop_scale_bar
        self.tile_width = max(64, int(tile_width))
        self.gap = max(4, int(gap))
        self.margin = max(8, int(margin))
        self.cli_args = list(cli_args or [])

    def build(self) -> Path:
        if len(self.row1_paths) != len(self.row2_paths):
            raise ValueError("row1 and row2 must contain the same number of files.")
        if not self.row1_paths:
            raise ValueError("At least one file per row is required.")

        row1_images = [self._load_tile(path) for path in self.row1_paths]
        row2_images = [self._load_tile(path) for path in self.row2_paths]
        tile_height = row1_images[0].shape[0]
        scaled_height = int(round(tile_height * (self.tile_width / row1_images[0].shape[1])))
        title_height = 40
        label_width = 72
        width = self.margin + label_width + (len(row1_images) * self.tile_width) + ((len(row1_images) - 1) * self.gap) + self.margin
        height = self.margin + title_height + (2 * scaled_height) + self.gap + self.margin
        canvas = np.full((height, width, 3), 12, dtype=np.uint8)

        image = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((self.margin, self.margin), self.title, fill=(255, 255, 255), font=font)

        x0 = self.margin + label_width
        y0 = self.margin + title_height

        self._draw_row(
            image=image,
            row_images=row1_images,
            row_label=self.row1_label,
            y=y0,
            x0=x0,
            scaled_height=scaled_height,
            font=font,
        )
        self._draw_row(
            image=image,
            row_images=row2_images,
            row_label=self.row2_label,
            y=y0 + scaled_height + self.gap,
            x0=x0,
            scaled_height=scaled_height,
            font=font,
        )

        canvas[:, :, :] = np.asarray(image, dtype=np.uint8)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(self.output_path, canvas, photometric="rgb")

        metadata_path = self.output_path.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(self._build_metadata(), indent=2), encoding="utf-8")
        return self.output_path

    def _draw_row(
        self,
        *,
        image: Image.Image,
        row_images: list[np.ndarray],
        row_label: str,
        y: int,
        x0: int,
        scaled_height: int,
        font: ImageFont.ImageFont,
    ) -> None:
        draw = ImageDraw.Draw(image)
        draw.text((self.margin, y + (scaled_height // 2) - 6), row_label, fill=(255, 255, 255), font=font)
        for index, tile in enumerate(row_images):
            resized = Image.fromarray(tile, mode="RGB").resize((self.tile_width, scaled_height), Image.Resampling.NEAREST)
            x = x0 + index * (self.tile_width + self.gap)
            image.paste(resized, (x, y))
            draw.rectangle((x, y, x + self.tile_width - 1, y + scaled_height - 1), outline=(50, 80, 50), width=1)
            label = self._tile_label(index=index, path=(self.row1_paths if row_label == self.row1_label else self.row2_paths)[index])
            draw.text((x, y - 14), label, fill=(180, 220, 180), font=font)

    def _load_tile(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Tile does not exist: {path}")
        tile = tiff.imread(path)
        if tile.ndim != 3 or tile.shape[2] != 3:
            raise ValueError(f"Expected an RGB TIFF: {path}")
        if self.crop_scale_bar and tile.shape[1] > tile.shape[0]:
            tile = tile[:, : tile.shape[0], :]
        return np.asarray(tile, dtype=np.uint8)

    @staticmethod
    def _tile_label(*, index: int, path: Path) -> str:
        name = path.stem
        if "_2026-03-09_" in name:
            suffix = name.split("_2026-03-09_", 1)[1]
            suffix = suffix.replace("_green_max_projection_visualized", "")
            return suffix or "0"
        return str(index + 1)

    def _build_metadata(self) -> dict[str, object]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable,
            "cli_args": self.cli_args,
            "output_path": str(self.output_path),
            "title": self.title,
            "row1_label": self.row1_label,
            "row2_label": self.row2_label,
            "crop_scale_bar": self.crop_scale_bar,
            "row1_paths": [str(path) for path in self.row1_paths],
            "row2_paths": [str(path) for path in self.row2_paths],
        }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 2xN selected green TIFF panel.")
    parser.add_argument("--row1", dest="row1_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--row2", dest="row2_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--row1-label", default="CL23")
    parser.add_argument("--row2-label", default="CL32")
    parser.add_argument("--title", default="IGF Green Max Projection")
    parser.add_argument("--keep-scale-bars", action="store_true")
    parser.add_argument("--tile-width", type=int, default=300)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--margin", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    builder = SelectedGreenPanelBuilder(
        row1_paths=args.row1_paths,
        row2_paths=args.row2_paths,
        output_path=args.output_path,
        row1_label=args.row1_label,
        row2_label=args.row2_label,
        title=args.title,
        crop_scale_bar=not args.keep_scale_bars,
        tile_width=args.tile_width,
        gap=args.gap,
        margin=args.margin,
        cli_args=(argv if argv is not None else sys.argv[1:]),
    )
    builder.build()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
