import os
from typing import Optional, Tuple, List

import h5py
import numpy as np
from bitshuffle.h5 import H5FILTER, H5_COMPRESS_LZ4

############ IO ##################
def _validate(file_name: str,
              n_merged_frames: int,
              skip_frames: Optional[List[int]] = None) -> None:
    """
    Validate input arguments for the frame merger.
    """
    if not file_name or not os.path.isfile(file_name):
        raise ValueError(f"Input file {file_name} does not exist.")

    if n_merged_frames <= 0:
        raise ValueError("n_merged_frames must be a positive integer.")

    if skip_frames is not None:
        if not isinstance(skip_frames, list):
            raise ValueError("skip_frames must be a list of integers.")
        for skip_idx in skip_frames:
            if not isinstance(skip_idx, int) or skip_idx < 0:
                raise ValueError(f"Invalid skip index: {skip_idx}")
            if skip_idx >= n_merged_frames:
                raise ValueError(
                    f"Skip frame index {skip_idx} not in valid range [0, {n_merged_frames - 1}]"
                )
        if len(skip_frames) >= n_merged_frames:
            raise ValueError(f"Cannot skip all {n_merged_frames} frames in each group.")


def _open_file(file_name: str, 
               data_location: str,
               data_name: str) -> Tuple[h5py.File, h5py.Dataset]:
    """
    Open an HDF5 file and retrieve the specified dataset.

    Args:
        file_name (str): Path to the HDF5 file.
        data_location (str): Group path within the HDF5 file (e.g. ``entry/data``).
        data_name (str): Name of the dataset containing frame data.

    Returns:
        Tuple[h5py.File, h5py.Dataset]: The open file handle and dataset object.

    Raises:
        IOError: If the file cannot be opened.
        KeyError: If the specified dataset is missing.
    """
    try:
        data_file = h5py.File(file_name, 'r')
    except Exception as e:
        raise IOError(f"Could not open file {file_name}: {e}")

    data_path = f"{data_location}/{data_name}"
    if data_path not in data_file:
        raise KeyError(f"Dataset {data_path} not found in file {file_name}.")

    data = data_file[data_path]
    return data_file, data

########## WRANGLING ##############
def _create_merge_indices(n_frames: int,
                       n_merged_frames: int) -> List[int]:
    """
    Generate starting indices for each merged frame group.

    Args:
        n_frames (int): Total number of frames available.
        n_merged_frames (int): Number of frames to merge in each group.

    Returns:
        List[int]: List of start indices for each merge group.
    """
    indices = []
    for i in range(0, n_frames, n_merged_frames):
        if n_frames - i >= n_merged_frames:
            indices.append(i)
    return indices

def _merge_chunk_mp(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Merge a chunk of frames (used for multiprocessing).

    Args:
        args (Tuple): Contains:
            - start_idx (int): Start index of the merge block.
            - data_subset (np.ndarray): Subset of frames to merge.
            - n_merged_frames (int): Number of frames in the group.
            - skip_frames (List[int]): Indices to skip within the group.
            - dtype: Data type for merged output.

    Returns:
        Tuple[int, np.ndarray]: Start index and merged frame (summed array).
    """
    start_idx, data_subset, n_merged_frames, skip_frames, dtype = args
    frame_shape = data_subset.shape[1:]
    merged = np.zeros(frame_shape, dtype=dtype)
    
    skip_set = set(skip_frames) if skip_frames else set()
    
    for i in range(n_merged_frames):
        if i not in skip_set:
            merged += data_subset[i]
    
    return start_idx, merged

def _merge_chunk_sq(data_array: np.ndarray,
                    n_frames: int,
                    n_merged_frames: int,
                    frame_shape: Tuple[int],
                    skip_frames: Optional[List[int]] = None,
                    dtype=None) -> np.ndarray:
    """
    Merge frames sequentially (single-process execution).

    Args:
        data_array (np.ndarray): Input frame array.
        n_frames (int): Number of frames to consider.
        n_merged_frames (int): Number of frames to merge in each group.
        frame_shape (Tuple[int]): Shape of a single frame (H, W).
        skip_frames (Optional[List[int]]): List of frame indices to skip in each group.
        dtype: Output data type.

    Returns:
        np.ndarray: Array of merged frames.
    """
    merge_indices = _create_merge_indices(n_frames, n_merged_frames)
    merged_data = np.zeros((len(merge_indices), *frame_shape), dtype=dtype)
    
    skip_set = set(skip_frames) if skip_frames else set()
    
    for i, start_idx in enumerate(merge_indices):
        frame_merged = np.zeros(frame_shape, dtype=dtype)
        for j in range(n_merged_frames):
            if j not in skip_set:
                frame_merged += data_array[start_idx + j]
        merged_data[i] = frame_merged
    
    return merged_data

########## OUTPUT ##############
def _write_output(output_file: str,
                  data_location: str,
                  data_name: str,
                  merged_data: np.ndarray,
                  dtype,
                  compression: Optional[Tuple[str, Tuple]] = None) -> None:
    """
    Write merged frame data to an output HDF5 file.

    Args:
        output_file (str): Path to the output HDF5 file.
        data_location (str): HDF5 group path where the data should be written.
        data_name (str): Name of the dataset for merged frames.
        merged_data (np.ndarray): Array of merged frames to write.
        dtype: Data type of the merged dataset.
        compression (Optional[Tuple[str, Tuple]]): Compression method and options.
            Defaults to Bitshuffle with LZ4 compression.

    Raises:
        IOError: If writing fails.
    """
    compression_opts = (0, H5_COMPRESS_LZ4)
    with h5py.File(output_file, "w") as f:
        grp = f.create_group(data_location)
        dset = grp.create_dataset(
            data_name,
            merged_data.shape,
            chunks=(1, merged_data.shape[1], merged_data.shape[2]),
            compression=H5FILTER,
            compression_opts=compression_opts,
            dtype=dtype,
        )
        dset[:] = merged_data
        