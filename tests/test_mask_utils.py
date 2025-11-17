import numpy as np
import pytest
import tifffile as tiff
from pathlib import Path

from simple_projection_tool.ihc_analyses.mask_utils import (
    discover_mask_path,
    load_mask_array,
)


def _prepare_run_folder(tmp_path: Path) -> Path:
    run = tmp_path / "sample_run"
    (run / "16bit").mkdir(parents=True)
    return run


def test_discover_mask_path_returns_none_when_missing(tmp_path):
    run = _prepare_run_folder(tmp_path)
    assert discover_mask_path(run) is None


def test_load_mask_array_returns_boolean_mask(tmp_path):
    run = _prepare_run_folder(tmp_path)
    mask_path = run / "16bit" / "mask.tif"
    tiff.imwrite(mask_path, np.array([[0, 1], [1, 0]], dtype=np.uint8))

    mask = load_mask_array(run)
    assert mask is not None
    assert mask.dtype == np.bool_
    assert mask.shape == (2, 2)
    assert mask.sum() == 2


def test_discover_mask_path_raises_when_multiple_masks(tmp_path):
    run = _prepare_run_folder(tmp_path)
    tiff.imwrite(run / "16bit" / "mask_one.tif", np.zeros((2, 2), dtype=np.uint8))
    tiff.imwrite(run / "16bit" / "mask_two.tif", np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError):
        discover_mask_path(run)
