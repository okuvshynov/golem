#!/usr/bin/env python3
"""Compare cached expert model vs full baseline model.

Tests:
1. Correctness: Do outputs match?
2. Performance: How does caching affect speed?
3. Cache statistics: Hit rates, miss patterns, etc.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add golem to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from golem.logging_config import setup_logging

# Import generation functions from both implementations
from baseline import generate_baseline
from generate import generate_with_cached_experts, parse_layer_spec

logger = logging.getLogger("golem")


def compare_outputs(baseline_output: str, cached_output: str, show_diff: bool = True) -> bool:
    """Compare two outputs and return whether they match.

    Args:
        baseline_output: Output from baseline generation
        cached_output: Output from cached generation
        show_diff: Whether to show the difference if outputs don't match

    Returns:
        True if outputs match, False otherwise
    """
    match = baseline_output == cached_output

    if not match and show_diff:
        logger.warning("\nOutputs differ!")
        logger.info(f"\nBaseline output ({len(baseline_output)} chars):")
        logger.info(f"  {baseline_output[:200]}...")
        logger.info(f"\nCached output ({len(cached_output)} chars):")
        logger.info(f"  {cached_output[:200]}...")

        # Find first difference
        for i, (c1, c2) in enumerate(zip(baseline_output, cached_output)):
            if c1 != c2:
                logger.info(f"\nFirst difference at position {i}:")
                logger.info(f"  Baseline: '{baseline_output[max(0,i-20):i+20]}'")
                logger.info(f"  Cached:   '{cached_output[max(0,i-20):i+20]}'")
                break

    return match


def main():
    """Run validation comparing baseline vs cached generation."""
    parser = argparse.ArgumentParser(
        description="Validate cached expert generation against baseline"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to model directory (local path)",
    )
    parser.add_argument(
        "-w", "--weights-path",
        type=str,
        default=None,
        help="Path to exported expert weights. If not specified, uses ~/.golem/<model> and auto-exports if needed.",
    )
    parser.add_argument(
        "-c", "--cache-size",
        type=int,
        default=32,
        help="Number of experts to cache per layer (default: 32)",
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="Write a poem",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "-n", "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "-f", "--fully-load-layers",
        type=str,
        default=None,
        help="Layer indices to fully load (skip caching). Supports ranges and comma-separated values. "
             "Examples: -f 0,1,2 or -f 3-20,89-91,45,56 or -f 0-5",
    )

    args = parser.parse_args()

    # Parse fully-load-layers specification
    fully_load_layers = None
    if args.fully_load_layers:
        try:
            fully_load_layers = parse_layer_spec(args.fully_load_layers)
        except ValueError as e:
            parser.error(f"Invalid --fully-load-layers specification: {e}")

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    model_path = Path(args.model)
    weights_path = Path(args.weights_path) if args.weights_path else None

    # Validate model path
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    # Ensure experts are exported (auto-export if needed)
    from golem.model import ensure_experts_exported

    try:
        weights_path = ensure_experts_exported(
            model_path=model_path,
            weights_path=weights_path,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Failed to ensure expert export: {e}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Cached vs Baseline Model Validation")
    logger.info("=" * 80)
    logger.info(f"\nModel: {model_path}")
    logger.info(f"Expert weights: {weights_path}")
    logger.info(f"Cache size: {args.cache_size} experts per layer")
    if fully_load_layers:
        logger.info(f"Fully load layers: {fully_load_layers}")
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Max tokens: {args.max_tokens}\n")

    # Run baseline generation
    logger.info("=" * 80)
    logger.info("Test 1: Baseline Model (Standard MLX-LM)")
    logger.info("=" * 80)
    try:
        baseline_result = generate_baseline(
            model_path=model_path,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        logger.info(f"Baseline generation time: {baseline_result['generation_time']:.2f}s")
    except Exception as e:
        logger.error(f"Baseline generation failed: {e}")
        sys.exit(1)

    # Run cached generation
    logger.info("\n" + "=" * 80)
    logger.info(f"Test 2: Cached Model ({args.cache_size} experts per layer)")
    logger.info("=" * 80)
    try:
        cached_result = generate_with_cached_experts(
            model_path=model_path,
            weights_path=weights_path,
            prompt=args.prompt,
            cache_size=args.cache_size,
            max_tokens=args.max_tokens,
            fully_load_layers=fully_load_layers,
            verbose=args.verbose,
        )
        logger.info(f"Cached generation time: {cached_result['generation_time']:.2f}s")
    except Exception as e:
        logger.error(f"Cached generation failed: {e}")
        sys.exit(1)

    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("Validation Results")
    logger.info("=" * 80)

    # Correctness check
    outputs_match = compare_outputs(
        baseline_result['output'],
        cached_result['output'],
        show_diff=not args.verbose
    )

    logger.info(f"\nCorrectness: {'✓ PASS' if outputs_match else '✗ FAIL'}")
    if outputs_match:
        logger.info("  Outputs match exactly")
    else:
        logger.warning("  Outputs differ - this may indicate a bug or numerical instability")

    # Performance comparison
    logger.info("\nPerformance:")
    baseline_time = baseline_result['generation_time']
    cached_time = cached_result['generation_time']
    speedup = baseline_time / cached_time if cached_time > 0 else 0

    logger.info(f"  Baseline: {baseline_time:.2f}s")
    logger.info(f"  Cached:   {cached_time:.2f}s")
    logger.info(f"  Speedup:  {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

    # Cache statistics
    expert_caches = cached_result['expert_caches']
    total_accesses = sum(cache.total_accesses for cache in expert_caches)
    total_hits = sum(cache.cache_hits for cache in expert_caches)
    total_misses = sum(cache.cache_misses for cache in expert_caches)
    overall_hit_rate = total_hits / total_accesses if total_accesses > 0 else 0

    logger.info("\nCache Statistics:")
    logger.info(f"  Total accesses: {total_accesses:,}")
    logger.info(f"  Cache hits:     {total_hits:,} ({overall_hit_rate:.1%})")
    logger.info(f"  Cache misses:   {total_misses:,}")
    logger.info(f"  Experts cached: {len(expert_caches)} layers × {args.cache_size} = {len(expert_caches) * args.cache_size}")

    # Memory savings estimate
    model_config = cached_result['model_config']
    total_experts = len(expert_caches) * model_config.num_experts
    cached_experts = len(expert_caches) * args.cache_size
    memory_savings = 1 - (cached_experts / total_experts) if total_experts > 0 else 0

    logger.info("\nMemory Usage:")
    logger.info(f"  Total experts in model:  {total_experts}")
    logger.info(f"  Experts in cache:        {cached_experts}")
    logger.info(f"  Memory savings estimate: ~{memory_savings:.1%} of MoE weights")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)

    if outputs_match:
        logger.info("✓ Validation PASSED")
        logger.info("  - Correctness: Outputs match exactly")
        logger.info(f"  - Performance: {speedup:.2f}x speedup")
        logger.info(f"  - Cache hit rate: {overall_hit_rate:.1%}")
        logger.info(f"  - Memory savings: ~{memory_savings:.1%}")
        return 0
    else:
        logger.error("✗ Validation FAILED")
        logger.error("  - Outputs do not match")
        return 1


if __name__ == "__main__":
    sys.exit(main())
