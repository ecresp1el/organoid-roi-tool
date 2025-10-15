from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np

ArrayLike = np.ndarray


def mean(array: ArrayLike) -> float:
    return float(np.mean(array))


def median(array: ArrayLike) -> float:
    return float(np.median(array))


def maximum(array: ArrayLike) -> float:
    return float(np.max(array))


def minimum(array: ArrayLike) -> float:
    return float(np.min(array))


def value_range(array: ArrayLike) -> float:
    data = np.asarray(array)
    return float(data.max() - data.min())


DEFAULT_STAT_FUNCS: Dict[str, Callable[[ArrayLike], float]] = {
    "mean": mean,
    "median": median,
    "max": maximum,
    "min": minimum,
    "range": value_range,
}


def compute_statistics(
    array: ArrayLike,
    *,
    metrics: Optional[Iterable[str]] = None,
    stat_funcs: Optional[Mapping[str, Callable[[ArrayLike], float]]] = None,
) -> Dict[str, float]:
    """
    Compute selected statistics for ``array``.

    Parameters
    ----------
    array:
        Input image (projection) array.
    metrics:
        Iterable of metric names to compute. Defaults to all available keys
        in ``stat_funcs`` (or :data:`DEFAULT_STAT_FUNCS`).
    stat_funcs:
        Mapping of metric name -> callable.
    """
    funcs = dict(DEFAULT_STAT_FUNCS)
    if stat_funcs:
        funcs.update(stat_funcs)

    selected = list(funcs.keys()) if metrics is None else list(metrics)
    data = np.asarray(array, dtype=np.float64)
    results: Dict[str, float] = {}
    for name in selected:
        func = funcs.get(name)
        if func is None:
            continue
        results[name] = float(func(data))
    return results


__all__ = [
    "mean",
    "median",
    "maximum",
    "minimum",
    "value_range",
    "DEFAULT_STAT_FUNCS",
    "compute_statistics",
]
