import os
from typing import Optional, Tuple, List

import h5py
import numpy as np
from bitshuffle.h5 import H5FILTER, H5_COMPRESS_LZ4

############ IO ##################

def _validate(file_name: str,
             n_merged_frames: int,
             skip_pattern: Optional[List[int]] = None):
    """
    Input validator for primary class
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
               data_name: str):
    """
    Opens HDF5 file and retrieves dataset handle
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
    Creates start indices for merge groups based on the skip pattern provided
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
    Merge a chunk of frames (multiprocessing)
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
    Merge a chunk of frames (sequentially)
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
                  compression: Optional[Tuple[str, Tuple]] = None):
    """
    Write merged data
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
        