#!/usr/bin/env python3
"""Visualize cache hit rate by token position and by layer for multiple cache sizes."""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# Color palette for consistent colors across charts
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']


def load_access_logs(log_dir: Path) -> dict:
    """Load all layer access logs from directory.

    Returns:
        Dict mapping (layer_idx, token_idx) -> (hits, total)
    """
    stats = defaultdict(lambda: [0, 0])  # [hits, total]

    for csv_path in sorted(log_dir.glob("layer_*_access.csv")):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer_idx"])
                token = int(row["token_idx"])
                is_miss = int(row["is_miss"])

                stats[(layer, token)][1] += 1  # total
                if not is_miss:
                    stats[(layer, token)][0] += 1  # hits

    return stats


def aggregate_by_token(stats: dict) -> tuple[list[int], list[float]]:
    """Aggregate hit rates by token index (across all layers)."""
    token_stats = defaultdict(lambda: [0, 0])

    for (layer, token), (hits, total) in stats.items():
        token_stats[token][0] += hits
        token_stats[token][1] += total

    tokens = sorted(token_stats.keys())
    hit_rates = [token_stats[t][0] / token_stats[t][1] if token_stats[t][1] > 0 else 0
                 for t in tokens]

    return tokens, hit_rates


def aggregate_by_layer(stats: dict) -> tuple[list[int], list[float]]:
    """Aggregate hit rates by layer index (across all tokens)."""
    layer_stats = defaultdict(lambda: [0, 0])

    for (layer, token), (hits, total) in stats.items():
        layer_stats[layer][0] += hits
        layer_stats[layer][1] += total

    layers = sorted(layer_stats.keys())
    hit_rates = [layer_stats[layer][0] / layer_stats[layer][1] if layer_stats[layer][1] > 0 else 0
                 for layer in layers]

    return layers, hit_rates


def extract_cache_size(dir_name: str) -> int | None:
    """Extract cache size from directory name like 'cold_c128_n1024' or 'c128_n1024'."""
    match = re.search(r'_?c(\d+)_', dir_name)
    return int(match.group(1)) if match else None


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Apply moving average smoothing."""
    if window <= 1 or len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    # Use 'same' mode and handle edges by padding
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def plot_hitrate_by_token(data: list[tuple[str, list[int], list[float]]],
                          output_path: Path, title: str, smooth_window: int = 20):
    """Line chart of hit rate vs token index for multiple cache sizes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, tokens, hit_rates) in enumerate(data):
        color = COLORS[i % len(COLORS)]
        smoothed = smooth(hit_rates, smooth_window)
        ax.plot(tokens, smoothed, linewidth=1.5, color=color, label=label)

    ax.set_xlabel("Token Index")
    ax.set_ylabel("Hit Rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_hitrate_by_layer(data: list[tuple[str, list[int], list[float]]],
                          output_path: Path, title: str):
    """Line chart of hit rate per layer for multiple cache sizes."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (label, layers, hit_rates) in enumerate(data):
        color = COLORS[i % len(COLORS)]
        ax.plot(layers, hit_rates, linewidth=1.5, color=color, label=label, marker='', markersize=3)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Hit Rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dirs", type=Path, nargs='+',
                        help="Directories containing layer_*_access.csv files")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: parent of first log_dir)")
    parser.add_argument("-p", "--prefix", type=str, default="",
                        help="Prefix for output filenames (e.g., 'cold' -> 'cold_hitrate_by_token.png')")
    args = parser.parse_args()

    for d in args.log_dirs:
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")

    output_dir = args.output_dir or args.log_dirs[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from all directories
    token_data = []  # [(label, tokens, hit_rates), ...]
    layer_data = []  # [(label, layers, hit_rates), ...]

    # Sort by cache size for consistent ordering
    sorted_dirs = sorted(args.log_dirs, key=lambda d: extract_cache_size(d.name) or 0)

    for log_dir in sorted_dirs:
        cache_size = extract_cache_size(log_dir.name)
        label = f"c={cache_size}" if cache_size else log_dir.name

        print(f"Loading {log_dir}...")
        stats = load_access_logs(log_dir)
        print(f"  {len(stats)} (layer, token) pairs")

        tokens, token_hit_rates = aggregate_by_token(stats)
        layers, layer_hit_rates = aggregate_by_layer(stats)

        total_hits = sum(s[0] for s in stats.values())
        total_accesses = sum(s[1] for s in stats.values())
        overall = total_hits / total_accesses
        print(f"  Overall hit rate: {overall:.1%}")

        token_data.append((label, tokens, token_hit_rates))
        layer_data.append((label, layers, layer_hit_rates))

    # Plot
    prefix = f"{args.prefix}_" if args.prefix else ""
    plot_hitrate_by_token(token_data, output_dir / f"{prefix}hitrate_by_token.png",
                          "Cache Hit Rate by Token Position")
    plot_hitrate_by_layer(layer_data, output_dir / f"{prefix}hitrate_by_layer.png",
                          "Cache Hit Rate by Layer")


if __name__ == "__main__":
    main()
