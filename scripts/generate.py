#!/usr/bin/env python3
"""Generate text using a model with offloaded expert weights.

Expert weights are auto-exported to ~/.golem/ on first run.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

# Add golem to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm.generate import stream_generate

from golem.logging_config import setup_logging
from golem.model import (
    install_expert_caches,
    install_fully_loaded_layers,
    load_model_without_experts,
)

logger = logging.getLogger("golem")


def parse_layer_spec(spec: str) -> List[int]:
    """Parse layer specification supporting ranges and comma-separated values.

    Args:
        spec: Layer specification string, e.g., "3-20,89-91,45,56"
              Supports:
              - Individual layers: "45,56"
              - Ranges (inclusive): "3-20"
              - Mixed: "3-20,89-91,45,56"

    Returns:
        Sorted list of unique layer indices

    Examples:
        >>> parse_layer_spec("0,1,2")
        [0, 1, 2]
        >>> parse_layer_spec("3-5,10")
        [3, 4, 5, 10]
        >>> parse_layer_spec("89-91,45,56")
        [45, 56, 89, 90, 91]
    """
    if not spec or not spec.strip():
        return []

    layers = set()

    # Split by comma
    parts = spec.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if it's a range (contains '-')
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())

                if start_idx > end_idx:
                    raise ValueError(f"Invalid range {part}: start > end")

                # Add all layers in range (inclusive)
                layers.update(range(start_idx, end_idx + 1))
            except ValueError as e:
                raise ValueError(f"Invalid range specification '{part}': {e}")
        else:
            # Single layer index
            try:
                layers.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid layer index '{part}': must be an integer")

    return sorted(layers)


def generate_with_latency_tracking(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
    timings_file: Path | None = None,
    output_file: Path | None = None,
    enable_context: bool = False,
):
    """Generate text with latency tracking and streaming output.

    Args:
        model: The MLX model
        tokenizer: The tokenizer
        prompt: Text prompt for generation
        max_tokens: Maximum tokens to generate
        verbose: Print detailed progress
        timings_file: Optional path to save per-token timings (in milliseconds)
        output_file: Optional path to append generated text (suppresses stdout)
        enable_context: Whether to update generation context for access logging

    Returns:
        tuple: (output_text, latency_stats_dict)
    """
    from golem.context import get_context

    latency_stats = {
        'total_time': 0.0,  # Total generation time (ms)
        'prompt_time': 0.0,  # Prompt processing time (ms)
        'generation_time': 0.0,  # Pure token generation time (ms)
        'prompt_tps': 0.0,  # Prompt tokens per second
        'generation_tps': 0.0,  # Generation tokens per second
        'prompt_tokens': 0,  # Number of prompt tokens
        'generation_tokens': 0,  # Number of generated tokens
    }

    # Initialize generation context
    if enable_context:
        ctx = get_context()
        ctx.reset(token_idx=0)

    # Start timing
    total_start = time.time()

    # Use stream_generate to get per-token output and timing
    full_text = ""

    # Initialize timing tracking
    token_timings = []  # List of inter-token times in milliseconds
    last_token_time = None

    # Open output file if specified (append mode)
    out_handle = open(output_file, 'a', encoding='utf-8') if output_file else None

    try:
        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, prefill_step_size=8192):
            # Output token as it's generated
            if out_handle:
                out_handle.write(response.text)
                out_handle.flush()
            else:
                print(response.text, end="", flush=True)

            # Measure time after output; output forces MLX evaluation
            current_time = time.time()

            # Track inter-token timing (skip first token as it includes prompt processing)
            if last_token_time is not None:
                inter_token_ms = (current_time - last_token_time) * 1000
                token_timings.append(inter_token_ms)

            last_token_time = current_time
            full_text += response.text

            # Update stats from the response
            latency_stats['prompt_tokens'] = response.prompt_tokens
            latency_stats['prompt_tps'] = response.prompt_tps
            latency_stats['generation_tokens'] = response.generation_tokens
            latency_stats['generation_tps'] = response.generation_tps

            # Increment token index for next iteration
            if enable_context:
                ctx.increment("token_idx")

        if not out_handle:
            print()  # New line after streaming output
    finally:
        if out_handle:
            out_handle.close()

    # Save timings to file if requested
    if timings_file and token_timings:
        with open(timings_file, 'w') as f:
            for timing in token_timings:
                f.write(f"{timing:.2f}\n")
        logger.info(f"Saved {len(token_timings)} token timings to {timings_file}")

    total_end = time.time()

    # Calculate timings
    latency_stats['total_time'] = (total_end - total_start) * 1000  # Convert to ms

    # Calculate prompt and generation times from tokens/tps
    if latency_stats['prompt_tps'] > 0:
        latency_stats['prompt_time'] = (latency_stats['prompt_tokens'] / latency_stats['prompt_tps']) * 1000
    if latency_stats['generation_tps'] > 0:
        latency_stats['generation_time'] = (latency_stats['generation_tokens'] / latency_stats['generation_tps']) * 1000

    return full_text, latency_stats


def generate_with_cached_experts(
    model_path: Path,
    prompt: str,
    weights_path: Path | None = None,
    cache_size: int = 32,
    cache_size_overrides: dict[int, int] | None = None,
    max_tokens: int = 100,
    fully_load_layers: List[int] | None = None,
    warmup_from: Path | None = None,
    random_init_seed: int | None = None,
    verbose: bool = False,
    timings_file: Path | None = None,
    access_log_dir: Path | None = None,
    output_file: Path | None = None,
    prompt_handling: str = "full",
) -> dict:
    """Generate text using cached expert weights.

    Args:
        model_path: Path to model directory (local path)
        prompt: Text prompt for generation
        weights_path: Optional path to exported expert weights.
                     If None, uses default cache in ~/.golem/
                     and auto-exports if needed.
        cache_size: Number of experts to cache per layer (default)
        cache_size_overrides: Optional dict mapping layer indices to custom cache sizes
        max_tokens: Maximum tokens to generate
        fully_load_layers: Layer indices to fully load (skip caching)
        warmup_from: Optional path to access log directory for frequency-based cache warming
        random_init_seed: Optional seed for random cache initialization (baseline comparison)
        verbose: Print detailed progress
        timings_file: Optional path to save per-token timings (in milliseconds)
        access_log_dir: Optional directory to save cache access logs (enables cache logging)
        output_file: Optional path to append generated text (suppresses stdout streaming)
        prompt_handling: How to handle prompt processing ('full' or 'adaptive')

    Returns:
        dict with keys:
            - 'output': generated text
            - 'latency_stats': dict with timing information
            - 'expert_caches': list of ExpertCache objects
            - 'model_config': model configuration info
            - 'generation_time': total generation time in seconds
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    # Validate model path
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Ensure experts are exported (auto-export if needed)
    from golem.model import ensure_experts_exported

    weights_path = ensure_experts_exported(
        model_path=model_path,
        weights_path=weights_path,
        verbose=verbose,
    )

    # Load model without expert weights
    logger.info("Loading model...")

    model, tokenizer, config = load_model_without_experts(
        model_path,
        verbose=verbose,
    )

    # Get model-specific config for printing stats
    from golem.model_configs import get_model_config
    model_config = get_model_config(model_path, config_dict=config, verbose=False)

    logger.info(f"Model loaded ({model_config.num_hidden_layers} layers, {model_config.num_experts} experts/layer)")

    # Load fully-loaded layers if specified
    if fully_load_layers:
        logger.info(f"Loading all experts for {len(fully_load_layers)} layers...")

        install_fully_loaded_layers(
            model,
            weights_path,
            fully_load_layers=fully_load_layers,
            verbose=verbose,
        )

        logger.info("Fully loaded layers")

    # Install expert caches
    logger.info("Installing expert caches...")

    # Enable cache logging only when exporting access logs
    enable_cache_logging = access_log_dir is not None

    expert_caches = install_expert_caches(
        model,
        weights_path,
        cache_size=cache_size,
        cache_size_overrides=cache_size_overrides,
        verbose=verbose,
        fully_load_layers=fully_load_layers,
        enable_logging=enable_cache_logging,
        prompt_handling=prompt_handling,
    )

    # Count MoE layers from model config
    from golem.model import get_moe_layer_indices
    total_moe_layers = len(get_moe_layer_indices(model))
    fully_loaded_count = len(fully_load_layers) if fully_load_layers else 0

    # Calculate total experts in memory (accounting for per-layer cache sizes)
    cached_experts = sum(cache.cache_size for cache in expert_caches)
    fully_loaded_experts = fully_loaded_count * model_config.num_experts
    total_experts_in_memory = cached_experts + fully_loaded_experts
    total_possible_experts = total_moe_layers * model_config.num_experts
    memory_ratio = total_experts_in_memory / total_possible_experts if total_possible_experts > 0 else 0

    logger.info(f'{cached_experts} + {fully_loaded_experts} < {total_possible_experts}')

    logger.info(f"Caches installed ({len(expert_caches)} cached, {fully_loaded_count} fully loaded, ~{memory_ratio:.0%} of experts)")

    # Warm caches from access logs or random initialization
    if warmup_from and random_init_seed is not None:
        logger.warning("Both --warmup-from and --random-init-seed specified; using warmup-from")

    if warmup_from:
        warmup_path = Path(warmup_from)
        if warmup_path.exists():
            logger.info(f"Warming caches from access logs in {warmup_path}...")
            for cache in expert_caches:
                cache.warmup_from_access_log(warmup_path, verbose=verbose)
            logger.info("Cache warming completed")
        else:
            logger.warning(f"Warmup directory not found at {warmup_path}, starting with cold cache")
    elif random_init_seed is not None:
        logger.info(f"Warming caches with random experts (seed={random_init_seed})...")
        for cache in expert_caches:
            cache.warmup_random(random_init_seed, verbose=verbose)
        logger.info("Random cache warming completed")

    # Format prompt
    if output_file:
        logger.info(f"Generating to {output_file}...")
    else:
        logger.info("Generating...")
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted_prompt = prompt

    # Generate with latency tracking
    output, latency_stats = generate_with_latency_tracking(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=verbose,
        timings_file=timings_file,
        output_file=output_file,
        enable_context=enable_cache_logging,
    )

    gen_time = latency_stats['total_time'] / 1000  # Convert back to seconds for compatibility

    # Print summary stats
    logger.info(f"Prompt: {latency_stats['prompt_tokens']} tokens, {latency_stats['prompt_tps']:.1f} t/s")
    logger.info(f"Generation: {latency_stats['generation_tokens']} tokens, {latency_stats['generation_tps']:.1f} t/s")

    # Cache statistics (only available when logging is enabled)
    if enable_cache_logging:
        total_accesses = sum(cache.total_accesses for cache in expert_caches)
        total_hits = sum(cache.cache_hits for cache in expert_caches)
        total_misses = sum(cache.cache_misses for cache in expert_caches)
        overall_hit_rate = total_hits / total_accesses if total_accesses > 0 else 0

        logger.info(f"Cache: {overall_hit_rate:.1%} hit rate ({total_hits:,} hits, {total_misses:,} misses)")

        # Per-layer breakdown if verbose
        if verbose:
            logger.info("Per-layer cache statistics:")
            for cache in expert_caches:
                moe_layer_idx = cache.layer_idx
                stats_line = f"  Layer {moe_layer_idx}: {cache.cache_hit_rate:.1%} hit rate, {len(cache.policy)} experts in cache"
                logger.info(stats_line)

    # Save access logs if requested
    if access_log_dir:
        access_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving access logs to {access_log_dir}...")
        for cache in expert_caches:
            log_file = access_log_dir / f"layer_{cache.layer_idx}_access.csv"
            cache.save_access_log(log_file)

    # Return results for use by calling code
    return {
        'output': output,
        'latency_stats': latency_stats,
        'expert_caches': expert_caches,
        'model_config': model_config,
        'generation_time': gen_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with offloaded expert weights"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to model directory (local path, not HuggingFace name)",
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
        "--cache-size-override",
        type=str,
        nargs=2,
        action="append",
        metavar=("LAYERS", "SIZE"),
        help="Override cache size for specific layers. Can be specified multiple times. "
             "Examples: --cache-size-override 40-50 150 --cache-size-override 30 200",
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "-P", "--prompt-file",
        type=str,
        default=None,
        help="Path to text file containing prompt",
    )
    parser.add_argument(
        "-n", "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "-T", "--timings",
        type=str,
        default=None,
        help="Save per-token timing data to file (in milliseconds)",
    )
    parser.add_argument(
        "-A", "--access-log-dir",
        type=str,
        default=None,
        help="Directory to save cache access logs (enables cache logging)",
    )
    parser.add_argument(
        "-W", "--warmup-from",
        type=str,
        default=None,
        help="Warm cache from access logs directory (uses frequency to prioritize experts)",
    )
    parser.add_argument(
        "-R", "--random-init-seed",
        type=int,
        default=None,
        help="Warm cache with random experts using this seed (baseline comparison)",
    )
    parser.add_argument(
        "-f", "--fully-load-layers",
        type=str,
        default=None,
        help="Layer indices to fully load (skip caching). Supports ranges and comma-separated values. "
             "Examples: -f 0,1,2 or -f 3-20,89-91,45,56 or -f 0-5",
    )
    parser.add_argument(
        "--prompt-handling",
        type=str,
        default="full",
        choices=["full", "adaptive"],
        help="How to handle prompt processing: 'full' loads all experts temporarily (default), "
             "'adaptive' uses cache if sufficient capacity",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Append generated text to file (suppresses stdout streaming)",
    )

    args = parser.parse_args()

    # Parse fully-load-layers specification
    fully_load_layers = None
    if args.fully_load_layers:
        try:
            fully_load_layers = parse_layer_spec(args.fully_load_layers)
        except ValueError as e:
            parser.error(f"Invalid --fully-load-layers specification: {e}")

    # Parse cache-size-override specifications
    cache_size_overrides = None
    if args.cache_size_override:
        cache_size_overrides = {}
        for layer_spec, size_str in args.cache_size_override:
            try:
                layers = parse_layer_spec(layer_spec)
                size = int(size_str)
                if size <= 0:
                    parser.error(f"Cache size must be positive, got: {size}")
                for layer_idx in layers:
                    cache_size_overrides[layer_idx] = size
            except ValueError as e:
                parser.error(f"Invalid --cache-size-override specification '{layer_spec} {size_str}': {e}")

    # Validate prompt arguments
    if args.prompt and args.prompt_file:
        parser.error("Cannot specify both --prompt and --prompt-file")

    # Read prompt from file or use command line argument
    if args.prompt_file:
        prompt_file_path = Path(args.prompt_file)
        if not prompt_file_path.exists():
            parser.error(f"Prompt file does not exist: {args.prompt_file}")
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except Exception as e:
            parser.error(f"Failed to read prompt file: {e}")
    elif args.prompt:
        prompt = args.prompt
    else:
        # Default prompt if neither is specified
        prompt = "The capital of France is"

    model_path = Path(args.model)
    weights_path = Path(args.weights_path) if args.weights_path else None

    try:
        generate_with_cached_experts(
            model_path=model_path,
            prompt=prompt,
            weights_path=weights_path,
            cache_size=args.cache_size,
            cache_size_overrides=cache_size_overrides,
            max_tokens=args.max_tokens,
            fully_load_layers=fully_load_layers,
            warmup_from=Path(args.warmup_from) if args.warmup_from else None,
            random_init_seed=args.random_init_seed,
            verbose=args.verbose,
            timings_file=Path(args.timings) if args.timings else None,
            access_log_dir=Path(args.access_log_dir) if args.access_log_dir else None,
            output_file=Path(args.output) if args.output else None,
            prompt_handling=args.prompt_handling,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
