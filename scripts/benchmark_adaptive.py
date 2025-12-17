#!/usr/bin/env python3
"""
Benchmark script comparing adaptive (entropy-based) vs static speculative decoding.

Runs both modes with identical seeds and verifies:
1. Generated tokens are identical (deterministic)
2. Measures end-to-end time for each mode
3. Compares accept rates and timing metrics
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx

from mlx_lm.utils import load
from mlx_lm.sample_utils import make_sampler

# Import from our stats script
from speculative_stats import generate_with_stats, SpeculativeStats


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    prompt_name: str
    mode: str  # "static" or "adaptive"
    entropy_threshold: Optional[float]
    generated_text: str
    num_tokens: int
    e2e_time_s: float
    tokens_per_sec: float
    accept_rate: float
    total_steps: int
    avg_drafted: float
    avg_draft_time_ms: float
    avg_verify_time_ms: float


def run_single_benchmark(
    model,
    draft_model,
    tokenizer,
    prompt: str,
    prompt_name: str,
    max_tokens: int,
    num_draft_tokens: int,
    seed: int,
    entropy_threshold: Optional[float] = None,
) -> BenchmarkResult:
    """Run a single benchmark with the given configuration."""
    # Reset random state
    mx.random.seed(seed)

    # Clear cache for fair comparison
    mx.clear_cache()

    sampler = make_sampler(temp=0.0)  # Greedy for determinism

    mode = "adaptive" if entropy_threshold is not None else "static"

    start_time = time.perf_counter()
    text, stats = generate_with_stats(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        sampler=sampler,
        live_stats=False,
        entropy_threshold=entropy_threshold,
    )
    e2e_time = time.perf_counter() - start_time

    # Calculate metrics
    num_tokens = stats.accepted_draft_tokens + stats.total_steps  # accepted + bonus tokens
    tokens_per_sec = num_tokens / e2e_time if e2e_time > 0 else 0

    # Calculate average drafted per step
    avg_drafted = stats.total_draft_tokens / stats.total_steps if stats.total_steps > 0 else 0

    # Calculate timing averages from step times (rough estimate)
    avg_step_time = sum(stats.step_times) / len(stats.step_times) if stats.step_times else 0
    # We don't have separate draft/verify times in stats, estimate based on ratio
    avg_draft_time_ms = avg_step_time * 1000 * 0.2  # rough estimate
    avg_verify_time_ms = avg_step_time * 1000 * 0.8  # rough estimate

    return BenchmarkResult(
        prompt_name=prompt_name,
        mode=mode,
        entropy_threshold=entropy_threshold,
        generated_text=text,
        num_tokens=num_tokens,
        e2e_time_s=e2e_time,
        tokens_per_sec=tokens_per_sec,
        accept_rate=stats.accept_rate,
        total_steps=stats.total_steps,
        avg_drafted=avg_drafted,
        avg_draft_time_ms=avg_draft_time_ms,
        avg_verify_time_ms=avg_verify_time_ms,
    )


def compare_results(static: BenchmarkResult, adaptive: BenchmarkResult) -> dict:
    """Compare static vs adaptive results."""
    tokens_match = static.generated_text == adaptive.generated_text

    return {
        "tokens_match": tokens_match,
        "e2e_speedup": (static.e2e_time_s - adaptive.e2e_time_s) / static.e2e_time_s * 100,
        "tps_improvement": (adaptive.tokens_per_sec - static.tokens_per_sec) / static.tokens_per_sec * 100,
        "accept_rate_delta": adaptive.accept_rate - static.accept_rate,
        "steps_delta": adaptive.total_steps - static.total_steps,
        "avg_drafted_delta": adaptive.avg_drafted - static.avg_drafted,
    }


def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    print(f"  Mode: {result.mode}" + (f" (threshold={result.entropy_threshold})" if result.entropy_threshold else ""))
    print(f"  Tokens: {result.num_tokens}")
    print(f"  Time: {result.e2e_time_s:.2f}s")
    print(f"  Speed: {result.tokens_per_sec:.1f} tok/s")
    print(f"  Accept rate: {result.accept_rate:.1%}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Avg drafted/step: {result.avg_drafted:.2f}")


def print_comparison(comparison: dict, static: BenchmarkResult, adaptive: BenchmarkResult):
    """Print comparison between static and adaptive."""
    print(f"\n  {'='*50}")
    print(f"  COMPARISON:")
    print(f"  {'='*50}")

    if comparison["tokens_match"]:
        print(f"  ✓ Tokens MATCH (deterministic)")
    else:
        print(f"  ✗ Tokens DIFFER!")
        print(f"    Static length: {len(static.generated_text)}")
        print(f"    Adaptive length: {len(adaptive.generated_text)}")
        # Find first difference
        for i, (c1, c2) in enumerate(zip(static.generated_text, adaptive.generated_text)):
            if c1 != c2:
                print(f"    First diff at char {i}: '{c1}' vs '{c2}'")
                print(f"    Context: ...{static.generated_text[max(0,i-20):i+20]}...")
                break

    speedup = comparison["e2e_speedup"]
    speedup_sign = "+" if speedup > 0 else ""
    print(f"  E2E time: {static.e2e_time_s:.2f}s → {adaptive.e2e_time_s:.2f}s ({speedup_sign}{speedup:.1f}%)")

    tps_imp = comparison["tps_improvement"]
    tps_sign = "+" if tps_imp > 0 else ""
    print(f"  Throughput: {static.tokens_per_sec:.1f} → {adaptive.tokens_per_sec:.1f} tok/s ({tps_sign}{tps_imp:.1f}%)")

    ar_delta = comparison["accept_rate_delta"]
    ar_sign = "+" if ar_delta > 0 else ""
    print(f"  Accept rate: {static.accept_rate:.1%} → {adaptive.accept_rate:.1%} ({ar_sign}{ar_delta:.1%})")

    steps_delta = comparison["steps_delta"]
    steps_sign = "+" if steps_delta > 0 else ""
    print(f"  Steps: {static.total_steps} → {adaptive.total_steps} ({steps_sign}{steps_delta})")

    drafted_delta = comparison["avg_drafted_delta"]
    drafted_sign = "+" if drafted_delta > 0 else ""
    print(f"  Avg drafted/step: {static.avg_drafted:.2f} → {adaptive.avg_drafted:.2f} ({drafted_sign}{drafted_delta:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark adaptive vs static speculation")
    parser.add_argument("--model", type=str, required=True, help="Main model path")
    parser.add_argument("--draft-model", type=str, required=True, help="Draft model path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per prompt")
    parser.add_argument("--num-draft-tokens", type=int, default=8, help="Draft tokens per step")
    parser.add_argument("--entropy-threshold", type=float, default=0.5, help="Entropy threshold for adaptive mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")

    args = parser.parse_args()

    # Test prompts of varying complexity
    PROMPTS = [
        ("simple_code", "Write a Python function to check if a number is prime."),
        ("explanation", "Explain the difference between concurrency and parallelism."),
        ("creative", "Write a short poem about artificial intelligence."),
        ("technical", "Explain how transformer attention mechanism works step by step."),
        ("multilang", "Show me hello world in Python, JavaScript, and Rust."),
    ]

    print(f"Loading main model: {args.model}")
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
    )

    print(f"Loading draft model: {args.draft_model}")
    draft_model, _ = load(
        args.draft_model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
    )

    # Prepare prompts with chat template
    def format_prompt(prompt_text: str) -> str:
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt_text}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt_text

    all_results: List[Tuple[BenchmarkResult, BenchmarkResult, dict]] = []

    print(f"\n{'='*70}")
    print("BENCHMARK: Adaptive vs Static Speculative Decoding")
    print(f"{'='*70}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Draft tokens/step: {args.num_draft_tokens}")
    print(f"Entropy threshold: {args.entropy_threshold}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    for prompt_name, prompt_text in PROMPTS:
        formatted_prompt = format_prompt(prompt_text)

        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt_name}")
        print(f"{'='*70}")
        print(f"Text: {prompt_text[:80]}...")

        # Run static (no entropy threshold)
        print(f"\n--- Running STATIC mode ---")
        static_result = run_single_benchmark(
            model, draft_model, tokenizer, formatted_prompt, prompt_name,
            args.max_tokens, args.num_draft_tokens, args.seed,
            entropy_threshold=None,
        )
        print_result(static_result)

        # Run adaptive (with entropy threshold)
        print(f"\n--- Running ADAPTIVE mode ---")
        adaptive_result = run_single_benchmark(
            model, draft_model, tokenizer, formatted_prompt, prompt_name,
            args.max_tokens, args.num_draft_tokens, args.seed,
            entropy_threshold=args.entropy_threshold,
        )
        print_result(adaptive_result)

        # Compare
        comparison = compare_results(static_result, adaptive_result)
        print_comparison(comparison, static_result, adaptive_result)

        all_results.append((static_result, adaptive_result, comparison))

    # Summary
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    all_match = all(c["tokens_match"] for _, _, c in all_results)
    print(f"\nToken consistency: {'✓ ALL MATCH' if all_match else '✗ SOME DIFFER'}")

    avg_speedup = sum(c["e2e_speedup"] for _, _, c in all_results) / len(all_results)
    avg_tps_imp = sum(c["tps_improvement"] for _, _, c in all_results) / len(all_results)
    avg_ar_delta = sum(c["accept_rate_delta"] for _, _, c in all_results) / len(all_results)

    print(f"\nAverage across all prompts:")
    print(f"  E2E speedup: {avg_speedup:+.1f}%")
    print(f"  Throughput improvement: {avg_tps_imp:+.1f}%")
    print(f"  Accept rate change: {avg_ar_delta:+.1%}")

    print(f"\nPer-prompt results:")
    print(f"{'Prompt':<15} {'Static(s)':<10} {'Adaptive(s)':<12} {'Speedup':<10} {'Match':<6}")
    print("-" * 55)
    for static, adaptive, comp in all_results:
        match_str = "✓" if comp["tokens_match"] else "✗"
        print(f"{static.prompt_name:<15} {static.e2e_time_s:<10.2f} {adaptive.e2e_time_s:<12.2f} {comp['e2e_speedup']:>+6.1f}%    {match_str}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
