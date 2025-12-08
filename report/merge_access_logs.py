#!/usr/bin/env python3
"""Merge multiple access log directories into a combined directory.

This allows testing cache warmup from combined access patterns of multiple prompts.
"""

import argparse
import csv
import sys
from pathlib import Path


def merge_access_logs(input_dirs: list[Path], output_dir: Path, verbose: bool = False):
    """Merge access logs from multiple directories.

    Args:
        input_dirs: List of directories containing layer_*_access.csv files
        output_dir: Directory to write merged access logs
        verbose: Print progress
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all unique layer files across all input directories
    layer_files: dict[int, list[Path]] = {}

    for input_dir in input_dirs:
        for csv_file in input_dir.glob("layer_*_access.csv"):
            # Extract layer index from filename
            layer_idx = int(csv_file.stem.split("_")[1])
            if layer_idx not in layer_files:
                layer_files[layer_idx] = []
            layer_files[layer_idx].append(csv_file)

    if verbose:
        print(f"Found {len(layer_files)} layers across {len(input_dirs)} directories")

    # Merge each layer's access logs
    for layer_idx in sorted(layer_files.keys()):
        files = layer_files[layer_idx]
        output_file = output_dir / f"layer_{layer_idx}_access.csv"

        all_rows = []
        fieldnames = None

        for csv_file in files:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    all_rows.append(row)

        if fieldnames and all_rows:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

            if verbose:
                print(f"  Layer {layer_idx}: merged {len(all_rows)} records from {len(files)} files")

    print(f"Merged access logs written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge access logs from multiple directories"
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="Input directories containing access logs",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for merged access logs",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress",
    )

    args = parser.parse_args()

    # Validate input directories
    for input_dir in args.input_dirs:
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
            sys.exit(1)

    merge_access_logs(args.input_dirs, args.output, args.verbose)


if __name__ == "__main__":
    main()
