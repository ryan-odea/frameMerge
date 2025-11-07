"""
frameMerge.merger
=======================

Main high-level class interface for frame merging of crystallographic HDF5 datasets.

This module defines the :class:`merger` class, which wraps helper utilities from
:mod:`frameMerge.helpers` to perform validation, file access, frame merging,
and writing of the merged dataset.

Example:
    ```python
    from frameMerge.frame_merger import merger

    m = merger(
        file_name="input.h5",
        output_file="merged.h5",
        n_frames=5000,
        n_merged_frames=10,
        skip_pattern=[1, 2],
        data_location="entry/data",
        data_name="data"
    )
    m.process(parallel=True)
    ```
"""

import os
from multiprocessing import Pool, cpu_count
from typing import Optional, List

import numpy as np
import h5py

from .helpers import (
    _validate,
    _open_file,
    _create_merge_indices,
    _merge_chunk_mp,
    _merge_chunk_sq,
    _write_output
)

class merger:
    """
    High-level interface for merging crystallographic HDF5 frames.

    This class supports both sequential and parallel (multiprocessing) execution.
    Frames are grouped and summed in chunks of ``n_merged_frames``, with an optional
    skip pattern defining variable intervals between merged blocks.

    Attributes:
        file_name (str): Input HDF5 file path.
        output_file (str): Output HDF5 file path.
        n_frames (int): Number of frames to read and process.
        n_merged_frames (int): Number of frames to merge per group.
        skip_pattern (Optional[List[int]]): Pattern of skips between groups.
        data_location (str): HDF5 group path containing the frame dataset.
        data_name (str): Name of dataset containing frames.
        n_workers (int): Number of parallel workers (defaults to CPU count).
    """
    def __init__(self,
                 file_name: str,
                 output_file: "merged.h5",
                 n_frames: int = 10000,
                 n_merged_frames: int = 10,
                 skip_pattern: Optional[int] = None,
                 data_location: str,
                 data_name: str,
                 n_workers: Optional[int] = None):
        self.file_name = file_name
        self.output_file = output_file
        self.n_frames = n_frames
        self.n_merged_frames = n_merged_frames
        self.skip_pattern = skip_pattern
        self.data_location = data_location
        self.data_name = data_name
        self.n_workers = n_workers or cpu_count()

        # Runtime
        self.data_file = None
        self.data = None
        self.data_array = None
        self.n_total_frames = None
        self.frame_shape = None
        self.dtype = None

    def process(self, parallel: bool = False) -> None:
        """
        Execute the full merge pipeline:
        validation → data loading → merging → output writing.

        Args:
            parallel (bool): If True, perform merging using multiprocessing.
                             Defaults to sequential mode.
        """
        try:
            _validate(self.file_name, self.n_merged_frames, self.skip_pattern)
            self._open_and_load()

            if parallel and self.n_workers > 1:
                self._merge_chunk_mp()
            else:
                self._merge_chunk_sq()
            
            _write_output(self.output_file,
                          self.data_location,
                          self.data_name,
                          self.merged_data,
                          self.dtype)
        finally:
            if self.data_file is not None:
                self.data_file.close()

    def _open_and_load(self) -> None:
        """
        Open the HDF5 file and load a subset of frames into memory.

        Automatically adjusts the number of frames if fewer than ``n_frames`` exist.
        """
        self.data_file, self.data = _open_file(self.file_name,
                                               self.data_location,
                                               self.data_name)
        self.n_total_frames = len(self.data)
        self.frame_shape = self.data.shape[1:]
        self.dtype = self.data.dtype

        if self.n_total_frames < self.n_frames:
            print(f"Warning: Requested {self.n_frames} frames, but only {self.n_total_frames} available. Adjusting n_frames.")
            self.n_frames = self.n_total_frames
        
        self.data_array = self.data[:self.n_frames]
    
    def _merge_parallel(self) -> None:
        """
        Merge frames using multiprocessing.

        Divides data into independent chunks according to the skip pattern,
        and merges each group in parallel using :func:`_merge_chunk_mp`.
        """
        chunks = []

        for start_idx in _create_merge_indices(self.n_frames,
                                               self.n_merged_frames,
                                               self.skip_pattern):

            subset = self.data_array[start_idx:start_idx + self.n_merged_frames]
            chunks.append((start_idx, subset, self.n_merged_frames, self.dtype))

        with Pool(self.n_workers) as pool:
            results = pool.map(_merge_chunk_mp, chunks)
        
        results.sort(key=lambda x: x[0])
        self.merged_data = np.array([r[1] for r in results])

    def _merge_sequential(self) -> None:
        """
        Merge frames sequentially (single-process).

        Uses :func:`_merge_chunk_sq` to merge groups in order.
        """
        self.merged_data = _merge_chunk_sq(self.data_array,
                                           self.n_frames,
                                           self.n_merged_frames,
                                           self.frame_shape,
                                           self.skip_pattern,
                                           self.dtype)
        