#!/usr/bin/env python3
"""Compare expert usage between two prompts, analyzing overlap per layer."""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_experts_by_layer(log_dir: Path) -> dict[int, set[int]]:
    """Load expert IDs accessed per layer.

    Returns:
        Dict mapping layer_idx -> set of expert_ids
    """
    experts = defaultdict(set)

    for csv_path in sorted(log_dir.glob("layer_*_access.csv")):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer_idx"])
                expert = int(row["expert_id"])
                experts[layer].add(expert)

    return dict(experts)


def compute_overlap_stats(experts_a: dict[int, set[int]],
                          experts_b: dict[int, set[int]]) -> list[dict]:
    """Compute per-layer overlap statistics.

    Returns:
        List of dicts with layer stats, sorted by layer index
    """
    all_layers = sorted(set(experts_a.keys()) | set(experts_b.keys()))
    stats = []

    for layer in all_layers:
        set_a = experts_a.get(layer, set())
        set_b = experts_b.get(layer, set())

        shared = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a
        union = set_a | set_b

        jaccard = len(shared) / len(union) if union else 0

        stats.append({
            "layer": layer,
            "total_a": len(set_a),
            "total_b": len(set_b),
            "shared": len(shared),
            "only_a": len(only_a),
            "only_b": len(only_b),
            "union": len(union),
            "jaccard": jaccard,
        })

    return stats


def plot_overlap_bars(stats: list[dict], output_path: Path,
                      label_a: str = "Prompt A", label_b: str = "Prompt B"):
    """Stacked bar chart showing shared vs unique experts per layer."""
    fig, ax = plt.subplots(figsize=(14, 5))

    layers = [s["layer"] for s in stats]
    shared = [s["shared"] for s in stats]
    only_a = [s["only_a"] for s in stats]
    only_b = [s["only_b"] for s in stats]

    x = np.arange(len(layers))
    width = 0.8

    # Stacked bars: shared (bottom), only_a (middle), only_b (top)
    ax.bar(x, shared, width, label="Shared", color="#4daf4a")
    ax.bar(x, only_a, width, bottom=shared, label=f"Only {label_a}", color="#377eb8")
    ax.bar(x, only_b, width, bottom=np.array(shared) + np.array(only_a),
           label=f"Only {label_b}", color="#e41a1c")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Number of Experts")
    ax.set_title("Expert Overlap by Layer")
    ax.legend(loc="upper right")

    # X-axis ticks
    if len(layers) > 30:
        tick_step = max(1, len(layers) // 15)
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), tick_step)])
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(stats: list[dict], label_a: str, label_b: str):
    """Print summary statistics."""
    total_shared = sum(s["shared"] for s in stats)
    total_only_a = sum(s["only_a"] for s in stats)
    total_only_b = sum(s["only_b"] for s in stats)
    total_union = sum(s["union"] for s in stats)

    mean_jaccard = np.mean([s["jaccard"] for s in stats])

    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Layers analyzed: {len(stats)}")
    print("Total unique experts accessed:")
    print(f"  {label_a}: {sum(s['total_a'] for s in stats)}")
    print(f"  {label_b}: {sum(s['total_b'] for s in stats)}")
    print("\nExpert breakdown (summed across layers):")
    print(f"  Shared:        {total_shared:5d} ({total_shared/total_union:.1%})")
    print(f"  Only {label_a}: {total_only_a:5d} ({total_only_a/total_union:.1%})")
    print(f"  Only {label_b}: {total_only_b:5d} ({total_only_b/total_union:.1%})")
    print(f"\nMean Jaccard similarity: {mean_jaccard:.1%}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_a", type=Path, help="Access log directory for prompt A")
    parser.add_argument("dir_b", type=Path, help="Access log directory for prompt B")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: parent of dir_a)")
    parser.add_argument("--label-a", type=str, default="Prompt A",
                        help="Label for prompt A")
    parser.add_argument("--label-b", type=str, default="Prompt B",
                        help="Label for prompt B")
    args = parser.parse_args()

    for d in [args.dir_a, args.dir_b]:
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")

    output_dir = args.output_dir or args.dir_a.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dir_a}...")
    experts_a = load_experts_by_layer(args.dir_a)
    print(f"  {len(experts_a)} layers")

    print(f"Loading {args.dir_b}...")
    experts_b = load_experts_by_layer(args.dir_b)
    print(f"  {len(experts_b)} layers")

    stats = compute_overlap_stats(experts_a, experts_b)

    print_summary(stats, args.label_a, args.label_b)

    plot_overlap_bars(stats, output_dir / "expert_overlap_bars.png",
                      args.label_a, args.label_b)


if __name__ == "__main__":
    main()
