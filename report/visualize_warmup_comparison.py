#!/usr/bin/env python3
"""Visualize cross-prompt warmup strategy comparison as grouped bar chart."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def calculate_hitrate(access_log_dir: Path) -> float:
    """Calculate overall hit rate from access logs."""
    total = 0
    misses = 0

    for csv_file in access_log_dir.glob("layer_*_access.csv"):
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row["is_miss"] == "1":
                    misses += 1

    return (total - misses) / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poetry-cold", type=Path, required=True)
    parser.add_argument("--poetry-random", type=Path, required=True)
    parser.add_argument("--poetry-same", type=Path, required=True)
    parser.add_argument("--poetry-other", type=Path, required=True)
    parser.add_argument("--poetry-merged", type=Path, required=True)
    parser.add_argument("--code-cold", type=Path, required=True)
    parser.add_argument("--code-random", type=Path, required=True)
    parser.add_argument("--code-same", type=Path, required=True)
    parser.add_argument("--code-other", type=Path, required=True)
    parser.add_argument("--code-merged", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Calculate hit rates
    strategies = ["Cold", "Random", "Same Prompt", "Other Prompt", "Merged"]

    poetry_rates = [
        calculate_hitrate(args.poetry_cold),
        calculate_hitrate(args.poetry_random),
        calculate_hitrate(args.poetry_same),
        calculate_hitrate(args.poetry_other),
        calculate_hitrate(args.poetry_merged),
    ]

    code_rates = [
        calculate_hitrate(args.code_cold),
        calculate_hitrate(args.code_random),
        calculate_hitrate(args.code_same),
        calculate_hitrate(args.code_other),
        calculate_hitrate(args.code_merged),
    ]

    # Print summary
    print("Cross-Prompt Warmup Comparison:")
    print(f"{'Strategy':<15} {'Poetry':>10} {'Code':>10}")
    print("-" * 37)
    for strat, p, c in zip(strategies, poetry_rates, code_rates):
        print(f"{strat:<15} {p:>9.1%} {c:>9.1%}")

    # Create grouped bar chart
    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [r * 100 for r in poetry_rates], width,
                   label='Poetry', color='#377eb8')
    bars2 = ax.bar(x + width/2, [r * 100 for r in code_rates], width,
                   label='Code', color='#e41a1c')

    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Cache Hit Rate by Warmup Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.set_ylim(90, 100)  # Focus on the relevant range
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = args.output_dir / "warmup_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
