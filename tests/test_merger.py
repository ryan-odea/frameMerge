import os
import tempfile
import numpy as np
import h5py
import pytest

from frameMerge.merger import Merger
from frameMerge.helpers import _merge_chunk_sq


def create_test_hdf5(num_frames=6, shape=(2, 2), dtype=np.int32):
    """Create a temporary HDF5 file with predictable test data."""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    with h5py.File(tmpfile.name, "w") as f:
        entry = f.create_group("entry")
        data_grp = entry.create_group("data")
        dset = data_grp.create_dataset("data", shape=(num_frames, *shape), dtype=dtype)
        # Fill frames with increasing integers per frame for clarity
        for i in range(num_frames):
            dset[i] = np.full(shape, i, dtype=dtype)
    return tmpfile.name


def test_merge_pattern_skip_1():
    """
    n_merged_frames = 3
    skip_pattern = [1]  → merge frames 0 and 2 (skip middle)
    """
    file_name = create_test_hdf5(num_frames=6)
    merger = Merger(
        file_name=file_name,
        output_file="test_merged.h5",
        n_frames=6,
        n_merged_frames=3,
        skip_pattern=[1],
        data_location="entry/data",
        data_name="data",
    )

    merger._open_and_load()
    merged = _merge_chunk_sq(
        merger.data_array,
        merger.n_frames,
        merger.n_merged_frames,
        merger.frame_shape,
        merger.skip_pattern,
        merger.dtype,
    )

    # There should be 2 merged frames: (0,1,2) and (3,4,5)
    # Each merged frame = sum of first and third (skip middle)
    expected_0 = merger.data_array[0] + merger.data_array[2]
    expected_1 = merger.data_array[3] + merger.data_array[5]

    np.testing.assert_array_equal(merged[0], expected_0)
    np.testing.assert_array_equal(merged[1], expected_1)

    os.remove(file_name)


def test_merge_pattern_skip_2():
    """
    n_merged_frames = 3
    skip_pattern = [2]  → merge frames 0 and 1 (skip last)
    """
    file_name = create_test_hdf5(num_frames=6)
    merger = Merger(
        file_name=file_name,
        output_file="test_merged.h5",
        n_frames=6,
        n_merged_frames=3,
        skip_pattern=[2],
        data_location="entry/data",
        data_name="data",
    )

    merger._open_and_load()
    merged = _merge_chunk_sq(
        merger.data_array,
        merger.n_frames,
        merger.n_merged_frames,
        merger.frame_shape,
        merger.skip_pattern,
        merger.dtype,
    )

    # Each merged frame = sum of first two frames in each block
    expected_0 = merger.data_array[0] + merger.data_array[1]
    expected_1 = merger.data_array[3] + merger.data_array[4]

    np.testing.assert_array_equal(merged[0], expected_0)
    np.testing.assert_array_equal(merged[1], expected_1)

    os.remove(file_name)


def test_end_to_end(tmp_path):
    """Full integration test: write merged file and check results."""
    file_name = create_test_hdf5(num_frames=6)
    output_file = tmp_path / "merged.h5"

    merger = Merger(
        file_name=file_name,
        output_file=str(output_file),
        n_frames=6,
        n_merged_frames=3,
        skip_pattern=[1],
        data_location="entry/data",
        data_name="data",
        n_workers=1,
    )

    merger.process(parallel=False)

    with h5py.File(output_file, "r") as f:
        merged_data = f["entry/data/data"][:]
        assert merged_data.shape[0] == 2
        # verify sum matches expected skip pattern [1]
        assert np.all(merged_data[0] == 0 + 2)
        assert np.all(merged_data[1] == 3 + 5)

    os.remove(file_name)
