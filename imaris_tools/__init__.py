"""
Utilities for working with Imaris ``.ims`` files.

The package exposes helpers to parse metadata, compute max intensity
projections per channel, and batch process stitched directories of
Imaris volumes. The core entry points are:

* :func:`process_directory` – iterate over ``.ims`` files in a folder
  and compute channel projections.
* :func:`process_file` – read a single file and yield projections and
  metadata.
* :func:`compute_max_projections` – low-level helper that generates the
  per-channel projection arrays.
"""

from .metadata import (
    ImarisChannelMetadata,
    ImarisMetadata,
    ImarisProjectionResult,
    read_metadata,
)
from .projections import (
    compute_max_projections,
    process_directory,
    process_file,
)

__all__ = [
    "ImarisChannelMetadata",
    "ImarisMetadata",
    "ImarisProjectionResult",
    "read_metadata",
    "compute_max_projections",
    "process_file",
    "process_directory",
]
