# Adaptive Speculative Decoding

Tools and experiments for analyzing and optimizing speculative decoding with mlx_lm.

## Speculative Stats Tool

The `scripts/speculative_stats.py` script provides detailed statistics for speculative decoding. This is useful for analyzing draft model performance and tuning speculative decoding parameters.

### Basic Usage

```bash
python scripts/speculative_stats.py \
    --model ./path/to/main-model \
    --draft-model ./path/to/draft-model \
    -p "Your prompt" \
    --max-tokens 256 \
    --num-draft-tokens 8
```

### Entropy-Based Early Stopping

Stop drafting early when the draft model is uncertain (high entropy):

```bash
python scripts/speculative_stats.py \
    --model ./path/to/main-model \
    --draft-model ./path/to/draft-model \
    -p "Your prompt" \
    --max-tokens 256 \
    --num-draft-tokens 8 \
    --entropy-threshold 0.5
```

### CSV Logging

Log per-token metrics for analysis:

```bash
python scripts/speculative_stats.py \
    --model ./path/to/main-model \
    --draft-model ./path/to/draft-model \
    -p "Your prompt" \
    --max-tokens 256 \
    --csv /tmp/speculation.csv
```

### Features

- Live statistics during generation (acceptance rate, rolling averages)
- Overall accept rate and per-step statistics
- Distribution of accepted tokens per verification step
- Accept rate evolution (comparing first vs last 20% of generation)
- Per-token confidence metrics (entropy, top-k probabilities)
- Timing breakdown (draft time, verify time, total throughput)

## Benchmark Tool

Compare adaptive vs static speculation:

```bash
python scripts/benchmark_adaptive.py \
    --model ./path/to/main-model \
    --draft-model ./path/to/draft-model \
    --max-tokens 256 \
    --num-draft-tokens 8 \
    --entropy-threshold 0.5
```

## Key Findings

### Entropy Predicts Acceptance

Analysis with Devstral-2-123B + Devstral-Small-24B showed strong correlation between draft model entropy and token acceptance:

| Metric | Accepted Tokens | Rejected Tokens |
|--------|-----------------|-----------------|
| Mean entropy | 0.098 | 1.028 |
| Mean top-1 prob | 0.969 | 0.616 |

**Correlation**: -0.63 (strong negative correlation between entropy and acceptance)

### Entropy Threshold Results

| Threshold | Tokens Affected | Rejection Rate |
|-----------|-----------------|----------------|
| entropy > 0.5 | 194 | 41.8% |
| entropy > 1.0 | 73 | 60.3% |
| entropy > 1.5 | 21 | 71.4% |
| entropy > 2.0 | 4 | 100% |

### Performance (threshold=0.5)

Benchmark across 5 diverse prompts:

| Metric | Improvement |
|--------|-------------|
| E2E speedup | +17.5% avg |
| Throughput | +24.1% avg |
| Accept rate | +26.0% avg |
| Best case (creative) | +39.8% speedup |

### Token Divergence with Quantized Models

**Important finding**: With quantized models (e.g., 8-bit), different `num_draft_tokens` values can produce different outputs even with greedy sampling. This is NOT a bug but a property of how KV cache computations interact with quantization:

- Different batch sizes during verification cause slightly different attention computations
- At low-confidence decision points, numerical differences can flip predictions
- Both outputs are semantically valid

This affects any change to step boundaries, whether from:
- Different `num_draft_tokens` settings
- Entropy-based early stopping
- Any adaptive n_draft strategy

For exact reproducibility with quantized models, use consistent speculation settings or non-speculative generation.

## Implementation Details

See [adaptive_draft_tokens_plan.md](adaptive_draft_tokens_plan.md) for detailed implementation notes and future directions.
