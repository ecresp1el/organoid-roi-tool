"""Quantification helpers for DCX spot measurement."""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


def measure_from_raw(raw_img: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Compute per-label intensity and geometry metrics from the raw image.

    Parameters
    ----------
    raw_img:
        The original (un-normalised) fluorescence image as a 2-D array.
    labels:
        Labeled integer array with 0 representing background and 1..N the clusters of interest.

    Returns
    -------
    pandas.DataFrame
        Table with one row per labeled cluster including centroid, area, equivalent diameter,
        and intensity statistics derived from ``raw_img``.
    """
    if raw_img.shape != labels.shape:
        raise ValueError(
            f"Raw image shape {raw_img.shape} must match labels shape {labels.shape}"
        )

    if labels.max() == 0:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "centroid_y",
                "centroid_x",
                "area_px",
                "equiv_diameter_px",
                "mean_intensity",
                "max_intensity",
                "sum_intensity",
            ]
        )

    props = regionprops_table(
        labels,
        intensity_image=raw_img,
        properties=(
            "label",
            "centroid",
            "area",
            "equivalent_diameter",
            "mean_intensity",
            "max_intensity",
        ),
    )
    df = pd.DataFrame(props)
    df.rename(
        columns={
            "label": "cluster_id",
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
            "area": "area_px",
            "equivalent_diameter": "equiv_diameter_px",
        },
        inplace=True,
    )

    # Compute integrated intensity using the boolean mask per label
    sum_intensity = []
    for cluster_id in df["cluster_id"].astype(int):
        mask = labels == cluster_id
        sum_intensity.append(float(raw_img[mask].sum()))
    df["sum_intensity"] = sum_intensity

    df = df[[
        "cluster_id",
        "centroid_y",
        "centroid_x",
        "area_px",
        "equiv_diameter_px",
        "mean_intensity",
        "max_intensity",
        "sum_intensity",
    ]]
    return df
