from pathlib import Path

import h5py  # type: ignore
import numpy as np
import tifffile as tiff  # type: ignore

from prepare_volumetric_data_labtalk import VolumetricDataLabtalkPreparer


def _create_two_channel_ims(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        dataset_group = handle.create_group("DataSet/ResolutionLevel 0/TimePoint 0")
        red_data = np.array(
            [
                [[0, 2], [4, 6]],
                [[8, 10], [12, 14]],
            ],
            dtype=np.uint16,
        )
        green_data = np.array(
            [
                [[1, 3], [5, 7]],
                [[9, 11], [13, 15]],
            ],
            dtype=np.uint16,
        )

        dataset_group.create_group("Channel 0").create_dataset("Data", data=red_data)
        dataset_group.create_group("Channel 1").create_dataset("Data", data=green_data)

        dataset_info = handle.create_group("DataSetInfo")
        red_info = dataset_info.create_group("Channel 0")
        red_info.attrs["Name"] = np.array("RFP", dtype="S")
        red_info.attrs["Color"] = np.array("1 0 0", dtype="S")

        green_info = dataset_info.create_group("Channel 1")
        green_info.attrs["Name"] = np.array("GFP", dtype="S")
        green_info.attrs["Color"] = np.array("0 1 0", dtype="S")


def test_prepare_all_generates_triptych_and_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    ims_file = input_dir / "sample.ims"
    _create_two_channel_ims(ims_file)

    output_dir = tmp_path / "out"
    preparer = VolumetricDataLabtalkPreparer(input_dir=input_dir, output_dir=output_dir)
    records = preparer.prepare_all()

    assert len(records) == 1
    record = records[0]
    assert record.red_channel_index == 0
    assert record.green_channel_index == 1

    output_image = tiff.imread(record.output_path)
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    assert output_image.shape == (2, 6, 3)
=======
    # 3 panels of width 2 + (gap=4, bar=14) per panel => 6 + 54
    assert output_image.shape == (2, 60, 3)
>>>>>>> theirs
=======
    # 3 panels of width 2 + (gap=4, bar=14) per panel => 6 + 54
    assert output_image.shape == (2, 60, 3)
>>>>>>> theirs
=======
    # 3 panels of width 2 + (gap=4, bar=14) per panel => 6 + 54
    assert output_image.shape == (2, 60, 3)
>>>>>>> theirs
=======
    # 3 panels of width 2 + (gap=4, bar=14) per panel => 6 + 54
    assert output_image.shape == (2, 60, 3)
>>>>>>> theirs
=======
    # 3 panels of width 2 + (gap=4, bar=14) per panel => 6 + 54
    assert output_image.shape == (2, 60, 3)
>>>>>>> theirs

    # red-only panel
    np.testing.assert_array_equal(output_image[:, :2, 1], 0)
    assert output_image[:, :2, 0].max() > 0

<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    # green-only panel
    np.testing.assert_array_equal(output_image[:, 2:4, 0], 0)
    assert output_image[:, 2:4, 1].max() > 0

    # merged panel contains both channels
    assert output_image[:, 4:6, 0].max() > 0
    assert output_image[:, 4:6, 1].max() > 0
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    # green-only panel starts after red panel + gap + bar
    green_start = 2 + 4 + 14
    np.testing.assert_array_equal(output_image[:, green_start : green_start + 2, 0], 0)
    assert output_image[:, green_start : green_start + 2, 1].max() > 0

    # merged panel starts after second panel block
    merged_start = green_start + 2 + 4 + 14
    assert output_image[:, merged_start : merged_start + 2, 0].max() > 0
    assert output_image[:, merged_start : merged_start + 2, 1].max() > 0
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    manifest = output_dir / "prepared_manifest.csv"
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "sample.ims" in text
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
=======
    assert "red_min_value" in text
    assert "green_max_value" in text
>>>>>>> theirs
=======
    assert "red_min_value" in text
    assert "green_max_value" in text
>>>>>>> theirs
=======
    assert "red_min_value" in text
    assert "green_max_value" in text
>>>>>>> theirs
=======
    assert "red_min_value" in text
    assert "green_max_value" in text
>>>>>>> theirs
=======
    assert "red_min_value" in text
    assert "green_max_value" in text
>>>>>>> theirs


def test_prepare_all_skips_without_overwrite(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    ims_file = input_dir / "sample.ims"
    _create_two_channel_ims(ims_file)

    output_dir = tmp_path / "out"
    preparer = VolumetricDataLabtalkPreparer(input_dir=input_dir, output_dir=output_dir)
    first = preparer.prepare_all()
    second = preparer.prepare_all()

    assert len(first) == 1
    assert len(second) == 0
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs


def test_prepare_without_scale_bars_retains_original_triptych_width(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    ims_file = input_dir / "sample.ims"
    _create_two_channel_ims(ims_file)

    output_dir = tmp_path / "out"
    preparer = VolumetricDataLabtalkPreparer(
        input_dir=input_dir,
        output_dir=output_dir,
        include_scale_bars=False,
    )
    records = preparer.prepare_all()
    assert len(records) == 1

    output_image = tiff.imread(records[0].output_path)
    assert output_image.shape == (2, 6, 3)
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
