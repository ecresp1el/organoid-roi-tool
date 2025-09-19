"""Utilities for discovering ND2 files and building a metadata manifest."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import nd2  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - gracefully handled downstream
    nd2 = None  # type: ignore[assignment]


ChannelResolver = Callable[[Path], Sequence[str] | None]

DEFAULT_STAIN_ALIASES: Mapping[str, str] = {
    "dapi": "DAPI",
    "dcx": "DCX",
    "gfap": "GFAP",
    "map2": "MAP2",
    "nestin": "Nestin",
    "sox2": "SOX2",
    "tuj1": "TUJ1",
    "ki67": "Ki67",
    "gfp": "GFP",
    "mcherry": "mCherry",
    "cy3": "Cy3",
    "cy5": "Cy5",
}

_TOKEN_SPLIT_RE = re.compile(r"[_\W]+")
_CHANNEL_INDEX_RE = re.compile(r"^(?:ch|c|channel)(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ND2ManifestEntry:
    """Row of manifest information for a single ND2 file."""

    path: Path
    relative_path: Path
    filename: str
    stem: str
    num_channels: int
    channel_names: tuple[str, ...]
    channel_source: str
    stains: tuple[str, ...]
    metadata_error: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "nd2_path": str(self.path),
            "nd2_relpath": str(self.relative_path),
            "filename": self.filename,
            "stem": self.stem,
            "num_channels": self.num_channels,
            "channel_names": list(self.channel_names),
            "channel_source": self.channel_source,
            "stains": list(self.stains),
            "metadata_error": self.metadata_error,
        }


def discover_nd2_files(root: str | Path, *, recursive: bool = True) -> list[Path]:
    """Return ND2 files beneath ``root``.

    Parameters
    ----------
    root:
        Directory to scan.
    recursive:
        When true, traverse subdirectories.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"ND2 root does not exist: {root_path}")
    if root_path.is_file():
        if root_path.suffix.lower() != ".nd2":
            raise ValueError(f"Requested root {root_path} is a file but not an ND2")
        return [root_path]

    walker = root_path.rglob if recursive else root_path.glob
    nd2_files = [p for p in walker("*.nd2") if p.is_file()]

    def sort_key(path: Path):
        rel = path.relative_to(root_path)
        parts = rel.parts
        return (len(parts), parts)

    nd2_files.sort(key=sort_key)
    return nd2_files


def build_nd2_manifest(
    root: str | Path,
    *,
    recursive: bool = True,
    channel_resolver: ChannelResolver | None = None,
    stain_aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Construct a manifest for every ND2 file beneath ``root``.

    The resulting DataFrame contains one row per file with columns describing
    path, inferred stains, and channel metadata. Channel detection relies on the
    ``nd2`` package when available; callers may inject a custom
    ``channel_resolver`` for testing or specialised metadata extraction.
    """

    stain_aliases = {**DEFAULT_STAIN_ALIASES, **(stain_aliases or {})}
    resolver = channel_resolver or _default_channel_resolver

    root_path = Path(root).expanduser().resolve()
    root_dir = root_path if root_path.is_dir() else root_path.parent

    entries: list[ND2ManifestEntry] = []
    for path in discover_nd2_files(root_path, recursive=recursive):
        try:
            relative_path = path.relative_to(root_dir)
        except ValueError:
            relative_path = Path(path.name)
        channel_names: tuple[str, ...] = ()
        channel_source = "inferred"
        metadata_error: str | None = None

        if resolver is not None:
            try:
                resolved = resolver(path)
            except Exception as exc:  # pragma: no cover - defensive logging path
                metadata_error = f"{exc.__class__.__name__}: {exc}"
                logger.warning("Failed to resolve channels for %s: %s", path, exc)
            else:
                if resolved:
                    channel_names = _normalise_channel_names(resolved)
                    channel_source = "metadata"

        stains, channel_indices = infer_stains_from_name(
            path.stem, channel_names, stain_aliases=stain_aliases
        )
        num_channels = _infer_channel_count(channel_names, channel_indices, stains)

        entries.append(
            ND2ManifestEntry(
                path=path,
                relative_path=relative_path,
                filename=path.name,
                stem=path.stem,
                num_channels=num_channels,
                channel_names=channel_names,
                channel_source=channel_source,
                stains=stains,
                metadata_error=metadata_error,
            )
        )

    if not entries:
        return pd.DataFrame(
            columns=
            [
                "nd2_path",
                "nd2_relpath",
                "filename",
                "stem",
                "num_channels",
                "channel_names",
                "channel_source",
                "stains",
                "metadata_error",
            ]
        )

    records = [entry.to_record() for entry in entries]
    return pd.DataFrame.from_records(records)


def infer_stains_from_name(
    stem: str,
    channel_names: Sequence[str] | None = None,
    *,
    stain_aliases: Mapping[str, str] | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return inferred stains and any explicit channel indices from a filename stem."""

    aliases = stain_aliases or DEFAULT_STAIN_ALIASES

    tokens = _tokenise(stem)
    stains = _stains_from_channels(channel_names, aliases)
    channel_indices: list[int] = []

    for token in tokens:
        canonical = aliases.get(token)
        if canonical:
            stains.append(canonical)
            continue
        matches = _CHANNEL_INDEX_RE.match(token)
        if matches:
            try:
                channel_indices.append(int(matches.group(1)))
            except ValueError:
                continue

    if "DAPI" not in stains:
        stains.append("DAPI")

    stains = _unique_preserve_order(stains)
    channel_indices = tuple(sorted(set(channel_indices)))
    return stains, channel_indices


def _infer_channel_count(
    channel_names: Sequence[str] | None,
    channel_indices: Sequence[int],
    stains: Sequence[str],
) -> int:
    explicit_count = len(channel_names or [])
    max_index = max(channel_indices) if channel_indices else 0
    stain_count = len(stains)
    candidates = [explicit_count, max_index, stain_count]
    num = max(candidates)
    if num == 0:
        return 1
    return num


def _default_channel_resolver(path: Path) -> Sequence[str] | None:
    """Resolve channel names from ND2 metadata when the ``nd2`` package is installed."""

    if nd2 is None:  # pragma: no cover - exercised only when dependency is present
        return None

    try:
        with nd2.ND2File(path) as nd2_file:  # type: ignore[attr-defined]
            channels = getattr(nd2_file.metadata, "channels", None)
            names: list[str] = []
            if channels:
                for channel in channels:
                    channel_meta = None
                    name = getattr(channel, "name", None)
                    if not name:
                        channel_meta = getattr(channel, "channel", None)
                        if channel_meta is not None:
                            name = getattr(channel_meta, "name", None)
                    if not name and isinstance(channel, dict):
                        name = channel.get("name")
                    if not name and channel_meta is not None:
                        name = str(channel_meta)
                    if name:
                        names.append(str(name))
            if names:
                return names
    except Exception as exc:  # pragma: no cover - protective logging path
        logger.warning("Failed reading ND2 metadata for %s: %s", path, exc)
    return None


def _normalise_channel_names(channels: Sequence[str]) -> tuple[str, ...]:
    cleaned = [str(ch).strip() for ch in channels if str(ch).strip()]
    return _unique_preserve_order(cleaned)


def _tokenise(stem: str) -> list[str]:
    tokens = [token.lower() for token in _TOKEN_SPLIT_RE.split(stem) if token]
    return tokens


def _stains_from_channels(
    channel_names: Sequence[str] | None,
    aliases: Mapping[str, str],
) -> list[str]:
    stains: list[str] = []
    if not channel_names:
        return stains
    for name in channel_names:
        token = name.lower().strip()
        canonical = aliases.get(token)
        if canonical:
            stains.append(canonical)
        else:
            stains.append(name.strip())
    return stains


def _unique_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)
