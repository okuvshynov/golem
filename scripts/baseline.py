#!/usr/bin/env python3
"""Baseline generation using standard mlx_lm without expert caching."""

from mlx_lm import load, generate
import sys
import time
from pathlib import Path


def generate_baseline(
    model_path: Path,
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
) -> dict:
    """Generate text using standard mlx_lm (no expert caching).

    Args:
        model_path: Path to model directory (can be HuggingFace name or local path)
        prompt: Text prompt for generation
        max_tokens: Maximum tokens to generate
        verbose: Print detailed progress

    Returns:
        dict with keys:
            - 'output': generated text
            - 'generation_time': total generation time in seconds
    """
    # Load model
    print(f"Loading model: {model_path}")
    start_time = time.time()
    model, tokenizer = load(str(model_path))
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    # Format prompt
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted_prompt = prompt

    # Generate
    print("\nGenerating...")
    start_time = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=verbose
    )
    gen_time = time.time() - start_time

    print(f"\nGeneration completed in {gen_time:.2f}s")

    return {
        'output': response,
        'generation_time': gen_time,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python baseline.py <model_path> [prompt] [max_tokens]")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a poem"
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    result = generate_baseline(
        model_path=model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Output:")
    print("=" * 80)
    print(result['output'])


if __name__ == "__main__":
    main()