from pathlib import Path

import numpy as np
import h5py  # type: ignore
import pytest

from imaris_tools import (
    compute_max_projections,
    process_directory,
    process_file,
    read_metadata,
)


def _create_mock_ims(path: Path) -> None:
    xml = """
    <DataSetInfo>
        <Image>
            <Name>Mock Imaris Volume</Name>
            <Description>Synthetic test data</Description>
            <Dimensions>
                <X>2</X>
                <Y>2</Y>
                <Z>2</Z>
                <C>3</C>
                <T>1</T>
            </Dimensions>
            <VoxelSize>
                <X>0.5</X>
                <Y>0.6</Y>
                <Z>1.0</Z>
            </VoxelSize>
        </Image>
        <Channels>
            <Channel Id="0">
                <Name>DNA</Name>
                <Color>
                    <Red>1</Red>
                    <Green>0</Green>
                    <Blue>0</Blue>
                </Color>
            </Channel>
            <Channel Id="1">
                <Name>Membrane</Name>
                <Color>
                    <Red>0</Red>
                    <Green>1</Green>
                    <Blue>0</Blue>
                </Color>
            </Channel>
            <Channel Id="2">
                <Name>Marker</Name>
                <Color>
                    <Red>0</Red>
                    <Green>0</Green>
                    <Blue>1</Blue>
                </Color>
            </Channel>
        </Channels>
    </DataSetInfo>
    """.strip()

    with h5py.File(path, "w") as handle:
        handle.create_dataset("DataSetInfo", data=np.string_(xml))
        dataset_group = handle.create_group("DataSet/ResolutionLevel 0/TimePoint 0")

        channel_data = {
            0: np.array(
                [
                    [[0, 1], [2, 3]],
                    [[4, 5], [6, 7]],
                ],
                dtype=np.uint16,
            ),
            1: np.array(
                [
                    [[1, 3], [5, 7]],
                    [[9, 11], [13, 15]],
                ],
                dtype=np.uint16,
            ),
            2: np.array(
                [
                    [[2, 4], [6, 8]],
                    [[10, 12], [14, 16]],
                ],
                dtype=np.uint16,
            ),
        }

        for idx, data in channel_data.items():
            channel_group = dataset_group.create_group(f"Channel {idx}")
            channel_group.create_dataset("Data", data=data)


@pytest.fixture()
def ims_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.ims"
    _create_mock_ims(file_path)
    return file_path


def test_read_metadata_extracts_channel_info(ims_file: Path) -> None:
    metadata = read_metadata(ims_file)
    assert metadata.name == "Mock Imaris Volume"
    assert metadata.voxel_size_um == (0.5, 0.6, 1.0)
    assert metadata.dimensions_xyzct == (2, 2, 2, 3, 1)

    channel_names = [channel.name for channel in metadata.channels]
    assert channel_names == ["DNA", "Membrane", "Marker"]


def test_compute_max_projections_returns_expected_arrays(ims_file: Path) -> None:
    metadata, projections = compute_max_projections(ims_file)
    assert len(projections) == 3
    assert metadata.dimensions_xyzct[:3] == (2, 2, 2)

    # Verify channel 0 projection.
    expected = np.array([[4, 5], [6, 7]], dtype=np.uint16)
    np.testing.assert_array_equal(projections[0], expected)


def test_process_file_generates_rgb_composite(ims_file: Path) -> None:
    result = process_file(ims_file, composite_dtype=np.uint16)
    composite = result.composite_rgb

    assert composite.shape == (2, 2, 3)
    # Each channel should map entirely to its colour.
    red_channel = composite[..., 0]
    green_channel = composite[..., 1]
    blue_channel = composite[..., 2]

    expected_red = np.array([[4, 5], [6, 7]], dtype=np.float32) / 7.0
    expected_green = np.array([[9, 11], [13, 15]], dtype=np.float32) / 15.0
    expected_blue = np.array([[10, 12], [14, 16]], dtype=np.float32) / 16.0

    np.testing.assert_allclose(
        red_channel.astype(np.float32), np.round(expected_red * 65535), atol=1
    )
    np.testing.assert_allclose(
        green_channel.astype(np.float32), np.round(expected_green * 65535), atol=1
    )
    np.testing.assert_allclose(
        blue_channel.astype(np.float32), np.round(expected_blue * 65535), atol=1
    )


def test_process_directory_handles_multiple_files(tmp_path: Path, ims_file: Path) -> None:
    extra_file = tmp_path / "extra.ims"
    _create_mock_ims(extra_file)

    results = process_directory(tmp_path)
    assert len(results) == 2
    for result in results:
        assert result.composite_rgb.shape == (2, 2, 3)
