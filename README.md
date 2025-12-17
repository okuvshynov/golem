# golem

A set of experiments on running LLMs locally, mostly GLM 4.6 for now, therefore, GoLeM.

## Experiments

### [MoE Expert Offloading](docs/moe_expert_offloading.md)

Dynamic expert reallocation for MoE models, storing a subset of experts on slower storage (SSD). Inspired by [Cerebras REAP](https://www.cerebras.ai/blog/reap).

Key results with Qwen3-235B-A22B on Mac Studio M2 Ultra:
- Baseline (4-bit, fits in memory): 28.7 t/s
- Cached 6-bit with warm start: 14.6 t/s

### [Adaptive Speculative Decoding](docs/adaptive_speculative_decoding.md)

Tools for analyzing and optimizing speculative decoding with entropy-based early stopping.

Key results:
- +17.5% average e2e speedup
- +24.1% throughput improvement
- Strong correlation (-0.63) between draft entropy and acceptance

## Hardware

All experiments run on:
- Apple Mac Studio M2 Ultra
- 72 GPU Cores
- 192GB unified memory
- SSD as secondary storage
