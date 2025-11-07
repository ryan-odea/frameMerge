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
    TODO
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

    def process(self, parallel: bool = False):
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

    def _open_and_load(self):
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
    
    def _merge_parallel(self):
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

    def _merge_sequential(self):
        self.merged_data = _merge_chunk_sq(self.data_array,
                                           self.n_frames,
                                           self.n_merged_frames,
                                           self.frame_shape,
                                           self.skip_pattern,
                                           self.dtype)
        