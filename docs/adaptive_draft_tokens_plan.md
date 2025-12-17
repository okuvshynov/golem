# Plan: Adaptive Draft Token Count - Phase 1 (Logging)

## Goal
Add per-draft-token confidence logging to understand the correlation between draft model confidence and acceptance rate. This "dark launch" will inform the adaptive n_draft implementation.

## Key Insight
In `scripts/speculative_stats.py`, draft logprobs are computed but discarded:
```python
# Line 242 in _draft_generate()
y, _ = _step(draft_model, draft_cache, y)  # logprobs discarded
```

We need to keep them and log alongside acceptance outcomes.

## CSV Format
**Single expanded file** with one row per draft token (not per step).

## Metrics to Log (per draft token)

| Column | Description |
|--------|-------------|
| `step` | Verification step number |
| `position` | Position within draft sequence (0 to n_draft-1) |
| `draft_token` | The token ID drafted |
| `main_token` | The token ID from main model |
| `accepted` | 1 if draft == main, 0 otherwise |
| `draft_entropy` | Entropy of draft distribution |
| `draft_prob` | P(draft_token) from draft model |
| `draft_top1_prob` | Probability of top-1 token from draft |
| `draft_top5_prob` | Sum of top-5 token probabilities from draft |
| `main_entropy` | Entropy of main model distribution |
| `main_prob` | P(main_token) from main model |
| `main_top1_prob` | Probability of top-1 token from main |
| `main_top5_prob` | Sum of top-5 token probabilities from main |

## Implementation Steps

### Step 1: Add helper function for confidence metrics
```python
def compute_confidence_metrics(logprobs, sampled_token):
    """Compute entropy and distribution stats from logprobs."""
    probs = mx.exp(logprobs)
    entropy = float(-mx.sum(probs * logprobs))
    sampled_prob = float(probs[sampled_token])
    sorted_probs = mx.sort(probs)[::-1]  # descending
    top1_prob = float(sorted_probs[0])
    top5_prob = float(mx.sum(sorted_probs[:5]))
    return entropy, sampled_prob, top1_prob, top5_prob
```

### Step 2: Modify `_draft_generate()` to return logprobs
Change line ~242 from discarding logprobs to collecting them:
```python
def _draft_generate(y, num_draft):
    if num_draft == 0:
        return mx.array([], mx.uint32), []
    ys = []
    logprobs_list = []
    for _ in range(num_draft):
        y, logprobs = _step(draft_model, draft_cache, y)
        mx.async_eval(y)
        ys.append(y)
        logprobs_list.append(logprobs)
    return mx.concatenate(ys), logprobs_list
```

### Step 3: Update SpeculativeStats CSV to per-token format
Replace current CSV logic with expanded per-token rows:
- Update `init_csv()` header to include all confidence columns
- Replace `record_step()` with `record_draft_token()` that writes one row per token
- Remove step-level aggregation from CSV (keep in-memory for summary)

### Step 4: Update main generation loop
After verification (line ~264), log each draft position up to (and including) first rejection:
```python
draft_tokens, draft_logprobs_list = _draft_generate(draft_y, num_draft)
# ... run main model ...
tokens, main_logprobs = _step(model, model_cache, y, num_draft + 1)

# Log draft tokens up to first rejection only
for pos in range(num_draft):
    dtn = draft_tokens[pos]
    tn = tokens[pos]
    accepted = (dtn == tn)

    draft_metrics = compute_confidence_metrics(draft_logprobs_list[pos], dtn)
    main_metrics = compute_confidence_metrics(main_logprobs[pos], tn)

    stats.record_draft_token(step, pos, dtn, tn, accepted, draft_metrics, main_metrics)

    if not accepted:
        break  # stop logging at rejection point
```

**Note**: We only log positions 0..N where N is the first rejection. Positions after rejection are not logged (they weren't actually used for generation).

### Step 5: Handle main model logprobs shape
The main model returns logprobs for all `num_draft + 1` positions in one call.
Need to ensure we index correctly: `main_logprobs[pos]` for each draft position.

## Files to Modify
- `scripts/speculative_stats.py` - All changes in this file (~50-80 lines changed)

## Testing
After implementation, run:
```bash
python scripts/speculative_stats.py \
    --model ./main-model \
    --draft-model ./draft-model \
    -p "Write a Python function" \
    --max-tokens 100 \
    --csv /tmp/speculation_detailed.csv
```

Then analyze with pandas:
```python
import pandas as pd
df = pd.read_csv('/tmp/speculation_detailed.csv')
# Correlation: draft_entropy vs accepted
df.groupby('accepted')['draft_entropy'].describe()
```

## Future Steps (not in this PR)
1. Analyze logged data to find confidence thresholds
2. Implement early stopping based on entropy threshold
3. Implement adaptive n_draft based on rolling accept rate
4. Combine both approaches with sanity guards [2, 16]
