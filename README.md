# frameMerge
`frameMerge` is a lightweight Python package for merging crystallographic detector frames within HDF5 files - supporting both parallel (multiprocessing) and sequential execution.

## Installation
You can pull from pypi with pip
```bash
[TODO]
```
Or you can clone the repo and install it locally:
```bash
git clone https://github.com/ryan-odea/frameMerge.git
cd frameMerge 
pip install .
```

## Usage
You can use frameMerge directly from the command line as:
```
frameMerge --file-name input.h5 \
           --output-file merged.h5 \
           --n-frames 5000 \
           --n-merged-frames 10 \
           --skip-pattern 1 2 \
           --data-location entry/data \
           --data-name data \
           --parallel
```
or through the Python API
```python
from frameMerge import merger

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