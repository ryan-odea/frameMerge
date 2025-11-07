#!/usr/bin/env python3
import argparse
from .merger import merger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge HDF5 frames with optional skip patterns."
    )
    parser.add_argument("-f", "--file_name", required=True, help="Input HDF5 file")
    parser.add_argument("-o", "--output_file", default="merged.h5", help="Output HDF5 file")
    parser.add_argument("--n_frames", type=int, default=10000, help="Number of frames to read")
    parser.add_argument("--n_merged_frames", type=int, default=10, help="Number of frames to merge")
    parser.add_argument("--skip_pattern", type=str, default=None, help="Comma-separated indices to skip")
    parser.add_argument("--data_location", type=str, default="entry/data", help="HDF5 group path")
    parser.add_argument("--data_name", type=str, default="data", help="Dataset name")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially instead of parallel")
    return parser.parse_args()

def main():
    args = parse_args()
    skip_pattern = None
    if args.skip_pattern:
        skip_pattern = [int(x.strip()) for x in args.skip_pattern.split(",")]

    m = merger(
        file_name=args.file_name,
        output_file=args.output_file,
        n_frames=args.n_frames,
        n_merged_frames=args.n_merged_frames,
        skip_pattern=skip_pattern,
        data_location=args.data_location,
        data_name=args.data_name,
        n_workers=args.n_workers
    )

    m.process(parallel=not args.sequential)
    print(f"Finished merging. Output written to {args.output_file}")

if __name__ == "__main__":
    main()
