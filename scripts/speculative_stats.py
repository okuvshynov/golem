#!/usr/bin/env python3
"""
Wrapper script for mlx_lm.generate with detailed speculative decoding statistics.

Usage:
    python speculative_stats.py --model ./mlx-community/Devstral-2-123B-Instruct-2512-8bit \
        -p "Your prompt" --max-tokens 1024 \
        --draft-model mlx-community/mistralai_Devstral-Small-2-24B-Instruct-2512-MLX-4Bit \
        --num-draft-tokens 16
"""

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any, Generator, TextIO

import mlx.core as mx
import mlx.nn as nn

# Import the original module (use mlx_lm.generate not mlx_lm)
from mlx_lm.generate import (
    wired_limit,
    generation_stream,
    maybe_quantize_kv_cache,
)
from mlx_lm.utils import load
from mlx_lm.sample_utils import make_sampler


def compute_entropy(logprobs: mx.array) -> float:
    """Compute entropy from logprobs (fast version for early stopping)."""
    if logprobs.ndim > 1:
        logprobs = logprobs.squeeze()
    probs = mx.exp(logprobs)
    return float(-mx.sum(probs * logprobs))


def compute_confidence_metrics(logprobs: mx.array, sampled_token: int) -> Tuple[float, float, float, float]:
    """
    Compute entropy and distribution stats from logprobs.

    Returns: (entropy, sampled_prob, top1_prob, top5_prob)
    """
    # Ensure logprobs is 1D (vocab_size,)
    if logprobs.ndim > 1:
        logprobs = logprobs.squeeze()
    probs = mx.exp(logprobs)
    entropy = float(-mx.sum(probs * logprobs))
    sampled_prob = float(probs[sampled_token])
    sorted_probs = mx.sort(probs)[::-1]  # descending
    top1_prob = float(sorted_probs[0])
    top5_prob = float(mx.sum(sorted_probs[:5]))
    return entropy, sampled_prob, top1_prob, top5_prob


@dataclass
class SpeculativeStats:
    """Detailed statistics for speculative decoding."""
    total_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    total_steps: int = 0
    step_history: List[Tuple[int, int]] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)
    csv_file: Optional[TextIO] = field(default=None, repr=False)
    csv_writer: Optional[csv.writer] = field(default=None, repr=False)

    def init_csv(self, path: str):
        """Initialize CSV logging to the given path (per-token format)."""
        self.csv_file = open(path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'step', 'position', 'draft_token', 'main_token', 'accepted',
            'draft_entropy', 'draft_prob', 'draft_top1_prob', 'draft_top5_prob',
            'main_entropy', 'main_prob', 'main_top1_prob', 'main_top5_prob',
            'draft_time_ms', 'verify_time_ms', 'num_drafted',
        ])

    def close_csv(self):
        """Close the CSV file if open."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def record_step(self, accepted: int, attempted: int, step_time: float = 0.0):
        """Record step-level statistics (in-memory only, for summary)."""
        self.total_draft_tokens += attempted
        self.accepted_draft_tokens += accepted
        self.total_steps += 1
        self.step_history.append((accepted, attempted))
        self.step_times.append(step_time)

    def record_draft_token(
        self,
        step: int,
        position: int,
        draft_token: int,
        main_token: int,
        accepted: bool,
        draft_metrics: Tuple[float, float, float, float],
        main_metrics: Tuple[float, float, float, float],
        draft_time_ms: float,
        verify_time_ms: float,
        num_drafted: int,
    ):
        """Record per-draft-token statistics to CSV."""
        if self.csv_writer:
            draft_entropy, draft_prob, draft_top1, draft_top5 = draft_metrics
            main_entropy, main_prob, main_top1, main_top5 = main_metrics
            self.csv_writer.writerow([
                step,
                position,
                draft_token,
                main_token,
                1 if accepted else 0,
                f"{draft_entropy:.6f}",
                f"{draft_prob:.6f}",
                f"{draft_top1:.6f}",
                f"{draft_top5:.6f}",
                f"{main_entropy:.6f}",
                f"{main_prob:.6f}",
                f"{main_top1:.6f}",
                f"{main_top5:.6f}",
                f"{draft_time_ms:.3f}",
                f"{verify_time_ms:.3f}",
                num_drafted,
            ])

    @property
    def accept_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.total_draft_tokens

    def rolling_accept_rate(self, window: int = 10) -> List[float]:
        """Calculate rolling accept rate over generation."""
        rates = []
        for i in range(len(self.step_history)):
            start = max(0, i - window + 1)
            window_data = self.step_history[start:i+1]
            accepted = sum(a for a, _ in window_data)
            attempted = sum(t for _, t in window_data)
            rates.append(accepted / attempted if attempted > 0 else 0.0)
        return rates

    def print_summary(self):
        print(f"\n{'='*60}")
        print("SPECULATIVE DECODING STATISTICS")
        print(f"{'='*60}")
        print(f"Total verification steps:    {self.total_steps}")
        print(f"Draft tokens attempted:      {self.total_draft_tokens}")
        print(f"Draft tokens accepted:       {self.accepted_draft_tokens}")
        print(f"Overall accept rate:         {self.accept_rate:.2%}")

        if self.step_history:
            # Per-step breakdown
            accepts_per_step = [a for a, _ in self.step_history]
            avg_accepts = sum(accepts_per_step) / len(accepts_per_step)
            max_accepts = max(accepts_per_step)
            min_accepts = min(accepts_per_step)

            print(f"\nPer-step statistics:")
            print(f"  Avg accepted per step:     {avg_accepts:.2f}")
            print(f"  Max accepted in a step:    {max_accepts}")
            print(f"  Min accepted in a step:    {min_accepts}")

            # Show accept rate evolution
            rolling = self.rolling_accept_rate(window=20)
            if len(rolling) >= 5:
                print(f"\nAccept rate evolution:")
                print(f"  First 20%:                 {sum(rolling[:len(rolling)//5])/(len(rolling)//5):.2%}")
                print(f"  Last 20%:                  {sum(rolling[-len(rolling)//5:])/(len(rolling)//5):.2%}")

            # Distribution of accepts per step
            from collections import Counter
            dist = Counter(accepts_per_step)
            print(f"\nDistribution of accepted tokens per step:")
            for k in sorted(dist.keys()):
                bar = '#' * min(dist[k], 40)
                print(f"  {k:2d} accepted: {dist[k]:4d} times {bar}")

        print(f"{'='*60}\n")

    def print_live(self, step_num: int):
        """Print live statistics during generation."""
        if step_num % 10 == 0 and step_num > 0:
            recent_window = min(20, len(self.step_history))
            recent = self.step_history[-recent_window:]
            recent_accepted = sum(a for a, _ in recent)
            recent_attempted = sum(t for _, t in recent)
            recent_rate = recent_accepted / recent_attempted if recent_attempted > 0 else 0

            print(f"\r[Step {step_num:4d}] "
                  f"Overall: {self.accept_rate:.1%} | "
                  f"Recent({recent_window}): {recent_rate:.1%} | "
                  f"Accepted: {self.accepted_draft_tokens}/{self.total_draft_tokens}",
                  end="", flush=True)


def speculative_generate_step_with_stats(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    stats: SpeculativeStats,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    live_stats: bool = True,
    entropy_threshold: Optional[float] = None,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    Wrapper around speculative_generate_step that collects detailed statistics.

    Args:
        entropy_threshold: If set, stop drafting early when entropy exceeds this value.
                          Recommended range: 0.5-1.0 based on empirical analysis.
    """
    from mlx_lm.models import cache
    import functools

    y = prompt.astype(mx.uint32)
    prev_tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
        draft_cache = cache.make_prompt_cache(draft_model)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers) :]

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _step(model, cache, y, n_predict=1):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=cache)
            logits = logits[:, -n_predict:, :]

            quantize_cache_fn(cache)
            if logits_processors:
                nonlocal prev_tokens
                out_y, out_logprobs = [], []
                if n_predict > 1:
                    y = y[: -(n_predict - 1)]
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concat([prev_tokens, y]) if prev_tokens is not None else y
                    )
                    y, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(y)
                    out_logprobs.append(logprobs)
                return mx.concatenate(out_y, axis=0), mx.concatenate(
                    out_logprobs, axis=0
                )
            else:
                return _process_and_sample(None, logits.squeeze(0))

    def _prefill(model, cache, y):
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=cache)
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            y = y[prefill_step_size:]
            mx.clear_cache()
        return y

    def _rewind_cache(num_draft, num_accept):
        cache.trim_prompt_cache(model_cache, num_draft - num_accept)
        cache.trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate(y, num_draft, entropy_threshold=None):
        if num_draft == 0:
            return mx.array([], mx.uint32), []
        ys = []
        logprobs_list = []
        for _ in range(num_draft):
            y, logprobs = _step(draft_model, draft_cache, y)
            mx.async_eval(y)
            ys.append(y)
            logprobs_list.append(logprobs)

            # Early stopping based on entropy threshold
            if entropy_threshold is not None:
                mx.eval(y)  # Need actual value for entropy computation
                entropy = compute_entropy(logprobs)
                if entropy > entropy_threshold:
                    break

        return mx.concatenate(ys), logprobs_list

    with mx.stream(generation_stream):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    ntoks = 0
    num_draft = 0
    n = 0
    step_count = 0

    try:
        while True:
            step_start = time.perf_counter()
            num_draft = min(max_tokens - ntoks, num_draft_tokens)

            # Time draft generation
            draft_start = time.perf_counter()
            draft_tokens, draft_logprobs_list = _draft_generate(draft_y, num_draft, entropy_threshold)
            mx.eval(draft_tokens)  # Ensure draft generation is complete
            draft_time = time.perf_counter() - draft_start

            # Actual number drafted (may be less due to early stopping)
            actual_drafted = len(draft_logprobs_list)

            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - actual_drafted + 1]
            y = mx.concatenate([y, draft_tokens])

            # Time main model verification
            verify_start = time.perf_counter()
            tokens, main_logprobs = _step(model, model_cache, y, actual_drafted + 1)
            mx.eval(tokens)  # Ensure verification is complete
            verify_time = time.perf_counter() - verify_start

            step_time = time.perf_counter() - step_start

            draft_tokens_list = draft_tokens.tolist()
            tokens_list = tokens.tolist()
            n = 0

            # Count accepted tokens and log per-token metrics
            while n < actual_drafted:
                tn, dtn = tokens_list[n], draft_tokens_list[n]
                accepted = (tn == dtn)

                # Log per-token confidence metrics to CSV
                if stats.csv_writer and n < len(draft_logprobs_list):
                    draft_metrics = compute_confidence_metrics(draft_logprobs_list[n], dtn)
                    main_metrics = compute_confidence_metrics(main_logprobs[n], tn)
                    stats.record_draft_token(
                        step_count + 1, n, dtn, tn, accepted, draft_metrics, main_metrics,
                        draft_time * 1000, verify_time * 1000, actual_drafted,
                    )

                if not accepted:
                    break
                n += 1
                ntoks += 1
                yield tn, main_logprobs[n - 1], True
                if ntoks == max_tokens:
                    break

            # Record step-level statistics (for summary)
            stats.record_step(accepted=n, attempted=actual_drafted, step_time=step_time)
            step_count += 1

            if live_stats:
                stats.print_live(step_count)

            if ntoks < max_tokens:
                ntoks += 1
                yield tokens_list[n], main_logprobs[n], False

            if ntoks == max_tokens:
                break

            y = mx.array([tokens_list[n]], mx.uint32)
            draft_y = y

            if n == actual_drafted:
                draft_y = mx.concatenate(
                    [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                )

            if prev_tokens is not None:
                prev_tokens = prev_tokens[: -max(actual_drafted - n, 1)]
            _rewind_cache(actual_drafted, n)
    finally:
        _rewind_cache(actual_drafted, n)
        if live_stats:
            print()  # Newline after live stats


def generate_with_stats(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    num_draft_tokens: int = 3,
    live_stats: bool = True,
    csv_path: Optional[str] = None,
    entropy_threshold: Optional[float] = None,
    **kwargs,
) -> Tuple[str, SpeculativeStats]:
    """
    Generate text with speculative decoding and return detailed statistics.

    Args:
        entropy_threshold: If set, stop drafting early when entropy exceeds this value.
    """
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Encode prompt
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    else:
        prompt_tokens = prompt
    prompt_array = mx.array(prompt_tokens)

    stats = SpeculativeStats()
    if csv_path:
        stats.init_csv(csv_path)
    detokenizer = tokenizer.detokenizer

    sampler = kwargs.pop('sampler', None)

    token_generator = speculative_generate_step_with_stats(
        prompt_array,
        model,
        draft_model,
        stats,
        num_draft_tokens=num_draft_tokens,
        max_tokens=max_tokens,
        sampler=sampler,
        live_stats=live_stats,
        entropy_threshold=entropy_threshold,
        **kwargs,
    )

    text = ""
    total_start = time.perf_counter()
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt_array.size / prompt_time
                tic = time.perf_counter()

            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

        detokenizer.finalize()
        text = detokenizer.text

        gen_time = time.perf_counter() - tic
        gen_tps = (n + 1) / gen_time if gen_time > 0 else 0

    total_time = time.perf_counter() - total_start
    total_tokens = prompt_array.size + n + 1
    total_tps = total_tokens / total_time if total_time > 0 else 0

    print(f"\nPrompt: {prompt_array.size} tokens, {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {n + 1} tokens, {gen_tps:.3f} tokens-per-sec")
    print(f"Total: {total_time:.2f}s, {total_tps:.2f} tokens-per-sec (prompt+gen)")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.3f} GB")

    if csv_path:
        stats.close_csv()
        print(f"Speculation log saved to: {csv_path}")

    return text, stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM generation with detailed speculative decoding statistics"
    )
    parser.add_argument("--model", type=str, required=True, help="Main model path")
    parser.add_argument("--draft-model", type=str, required=True, help="Draft model path")
    parser.add_argument("-p", "--prompt", type=str, default="Hello", help="Input prompt")
    parser.add_argument("--max-tokens", "-m", type=int, default=256, help="Max tokens")
    parser.add_argument("--num-draft-tokens", type=int, default=4, help="Draft tokens per step")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-live", action="store_true", help="Disable live stats")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file for logging speculation results")
    parser.add_argument("--entropy-threshold", type=float, default=None,
                        help="Stop drafting early when entropy exceeds this value (recommended: 0.5-1.0)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    parser.add_argument("--ignore-chat-template", action="store_true", help="Ignore chat template")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt")

    args = parser.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    print(f"Loading main model: {args.model}")
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
    )

    print(f"Loading draft model: {args.draft_model}")
    draft_model, draft_tokenizer = load(
        args.draft_model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
    )

    # Prepare prompt with chat template
    prompt = args.prompt
    if not args.ignore_chat_template and tokenizer.chat_template is not None:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    sampler = make_sampler(
        args.temp,
        args.top_p,
        args.min_p,
        min_tokens_to_keep=1,
        top_k=args.top_k,
    )

    print(f"\n{'='*60}")
    print("Starting generation...")
    print(f"{'='*60}\n")

    text, stats = generate_with_stats(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        draft_model=draft_model,
        num_draft_tokens=args.num_draft_tokens,
        sampler=sampler,
        live_stats=not args.no_live,
        csv_path=args.csv,
        entropy_threshold=args.entropy_threshold,
    )

    print(f"\n{'='*60}")
    print("GENERATED TEXT:")
    print(f"{'='*60}")
    print(text)

    stats.print_summary()

    return stats


if __name__ == "__main__":
    main()
