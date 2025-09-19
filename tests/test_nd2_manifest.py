from pathlib import Path

from cellcount_tools.nd2_manifest import build_nd2_manifest, discover_nd2_files


def test_discover_nd2_files_recursive(tmp_path):
    top_level = tmp_path / "sample.nd2"
    top_level.touch()

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "inner.nd2"
    nested_file.touch()

    (tmp_path / "ignore.txt").touch()

    result = discover_nd2_files(tmp_path)
    assert result == [top_level, nested_file]


def test_build_nd2_manifest_with_channel_resolver(tmp_path):
    gfap = tmp_path / "plate1_GFAP.nd2"
    gfap.touch()

    nested = tmp_path / "nested"
    nested.mkdir()
    dcx = nested / "day1_DCX.nd2"
    dcx.touch()

    control = tmp_path / "control.nd2"
    control.touch()

    def fake_resolver(path: Path):
        if path.name.endswith("GFAP.nd2"):
            return ["DAPI", "GFAP"]
        if path.name.endswith("DCX.nd2"):
            return ["DAPI", "DCX"]
        return None

    manifest = build_nd2_manifest(tmp_path, channel_resolver=fake_resolver)

    assert set(manifest["filename"]) == {"plate1_GFAP.nd2", "day1_DCX.nd2", "control.nd2"}

    gfap_row = manifest.loc[manifest["filename"] == "plate1_GFAP.nd2"].iloc[0]
    assert gfap_row["channel_names"] == ["DAPI", "GFAP"]
    assert gfap_row["stains"] == ["DAPI", "GFAP"]
    assert gfap_row["num_channels"] == 2
    assert gfap_row["channel_source"] == "metadata"

    dcx_row = manifest.loc[manifest["filename"] == "day1_DCX.nd2"].iloc[0]
    assert dcx_row["channel_names"] == ["DAPI", "DCX"]
    assert dcx_row["stains"] == ["DAPI", "DCX"]
    assert dcx_row["num_channels"] == 2

    control_row = manifest.loc[manifest["filename"] == "control.nd2"].iloc[0]
    assert control_row["channel_names"] == []
    assert control_row["stains"] == ["DAPI"]
    assert control_row["num_channels"] == 1
    assert control_row["channel_source"] == "inferred"

