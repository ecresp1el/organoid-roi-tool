"""
Utilities for working with Imaris ``.ims`` files.

Keep package import lightweight so CLI tools can start quickly without
triggering plotting/font-cache setup unless those features are used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .metadata import (
    ImarisChannelMetadata,
    ImarisMetadata,
    ImarisProjectionResult,
    default_channel_color,
    read_metadata,
)
from .projections import (
    colorize_projection,
    compute_max_projections,
    process_directory,
    process_file,
)

__all__ = [
    "ImarisChannelMetadata",
    "ImarisMetadata",
    "ImarisProjectionResult",
    "default_channel_color",
    "read_metadata",
    "compute_max_projections",
    "process_file",
    "process_directory",
    "colorize_projection",
    "export_directory",
    "plot_folder_projection_grid",
    "save_per_file_overview",
    "compute_statistics",
    "DEFAULT_STAT_FUNCS",
    "save_fluorescent_max_projections",
]


def __getattr__(name: str) -> Any:
    if name in {"plot_folder_projection_grid", "save_per_file_overview"}:
        module = import_module(".plotting", __name__)
        return getattr(module, name)
    if name == "export_directory":
        module = import_module(".export", __name__)
        return getattr(module, name)
    if name in {"compute_statistics", "DEFAULT_STAT_FUNCS"}:
        module = import_module(".stats", __name__)
        return getattr(module, name)
    if name == "save_fluorescent_max_projections":
        module = import_module(".quicklook", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
