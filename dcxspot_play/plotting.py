"""Minimal plotting utilities for organoid ROI analyses."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, MutableMapping, Optional

import matplotlib as mpl
from cycler import cycler

# Shared, minimal aesthetic for all downstream figures.
_BASE_STYLE: Dict[str, object] = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#222222",
    "axes.titlecolor": "#222222",
    "axes.titlesize": 13,
    "axes.titlepad": 14,
    "axes.labelsize": 11,
    "axes.grid": False,
    "axes.grid.axis": "y",
    "axes.grid.which": "major",
    "grid.color": "#e1e1e1",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.prop_cycle": cycler(color=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]),
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "axes.titlelocation": "left",
}


def minimal_style(overrides: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """Return the default minimal plotting style dictionary."""
    style = dict(_BASE_STYLE)
    if overrides:
        style.update(overrides)
    return style


def apply_minimal_style(overrides: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """Apply the minimal plotting style globally and return the rcParams mapping used."""
    style = minimal_style(overrides)
    mpl.rcParams.update(style)
    return style


@contextmanager
def minimal_style_context(overrides: Optional[Mapping[str, object]] = None) -> Iterator[MutableMapping[str, object]]:
    """Temporarily apply the minimal plotting style within a context manager."""
    style = minimal_style(overrides)
    with mpl.rc_context(style):
        yield mpl.rcParams
