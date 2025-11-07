import os
from typing import Optional, Tuple, List

import h5py
import numpy as np
from bitshuffle.h5 import H5FILTER, H5_COMPRESS_LZ4

############ IO ##################

def _validate(file_name: str,
             n_merged_frames: int,
             skip_pattern: Optional[List[int]] = None) -> None:
    """
    Validate input arguments for the frame merger.

    Args:
        file_name (str): Path to the HDF5 input file.
        n_merged_frames (int): Number of frames to merge per group.
        skip_pattern (Optional[List[int]]): Optional list of integers
            defining skip intervals between merged groups.

    Raises:
        ValueError: If the input file does not exist, or parameters are invalid.
    """
    if not file_name or not os.path.isfile(file_name):
        raise ValueError(f"Input file {file_name} does not exist.")
    
    if n_merged_frames <= 0:
        raise ValueError("n_merged_frames must be a positive integer.")

    if skip_pattern is not None:
        if not isinstance(skip_pattern, list) or not all(isinstance(x, int) and x >= 0 for x in skip_pattern):
            raise ValueError("skip_pattern must be a list of non-negative integers.")

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
    except: Exception as e:
        raise IOError(f"Could not open file {file_name}: {e}")

    data_path = f"{data_location}/{data_name}"
    if data_path not in data_file:
        raise KeyError(f"Dataset {data_path} not found in file {file_name}.")

    data = data_file[data_path]
    return data_file, data

########## WRANGLING ##############
def _create_merge_indices(n_frames: int,
                          n_merged_frames: int,
                          skip_pattern: Optional[List[int]] = None):
    """
    Generate starting indices for each merged frame group.

    Args:
        n_frames (int): Total number of frames available.
        n_merged_frames (int): Number of frames to merge in each group.
        skip_pattern (Optional[List[int]]): Optional list of skip intervals.
            If provided, pattern is cycled over each merge iteration.

    Yields:
        int: Start index of each merge group.
    """
    skip_pattern = skip_pattern or [0]
    pattern_length = len(skip_pattern)
    frame_idx = 0
    i = 0

    while frame_idx + n_merged_frames <= n_frames:
        yield frame_idx
        frame_idx += n_merged_frames + skip_pattern[i % pattern_length]
        i += 1

def _merge_chunk_mp(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Merge a chunk of frames (used for multiprocessing).

    Args:
        args (Tuple): Contains:
            - start_idx (int): Start index of the merge block.
            - data_subset (np.ndarray): Subset of frames to merge.
            - n_merged_frames (int): Number of frames to merge.
            - dtype: Data type for merged output.

    Returns:
        Tuple[int, np.ndarray]: Start index and merged frame (summed array).
    """
    start_idx, data_subset, n_merged_frames, dtype = args
    merged = np.sum(data_subset[:n_merged_frames], axis=0, dtype=dtype)
    return start_idx, merged

def _merge_chunk_sq(data_array: np.ndarray,
                    n_frames: int,
                    n_frames_merged: int,
                    frame_shape: Tuple[int],
                    skip_pattern: Optional[List[int]] = None,
                    dtype) -> np.ndarray:
    """
    Merge frames sequentially (single-process execution).

    Args:
        data_array (np.ndarray): Input frame array.
        n_frames (int): Number of frames to consider.
        n_frames_merged (int): Number of frames to merge in each group.
        frame_shape (Tuple[int]): Shape of a single frame (H, W).
        skip_pattern (Optional[List[int]]): List of skip intervals between merges.
        dtype: Output data type.

    Returns:
        np.ndarray: Array of merged frames.
    """
    merge_idx = list(generate_merge_indices(n_frames, n_frames_merged, skip_pattern))
    merge_data = np.zeros((len(merge_idx), *frame_shape), dtype=dtype)

    for i, start_idx in enumerate(merge_idx):
        merge_data[i] = np.sum(data_array[start_idx:start_idx + n_frames_merged], axis=0, dtype=dtype)
    return merge_data

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
    with h5py.File(output_file, 'w') as f:
        data_output = f.create_group(data_location)
        data_dset_output = data_output.create_dataset(
            data_name,
            merged_data.shape,
            chunks=(1, merged_data.shape[1], merged_data.shape[2]),
            compression=H5FILTER,
            compression_opts=(0, H5_COMPRESS_LZ4),
            dtype=dtype
        )
        data_dset_output[:] = merged_data
        