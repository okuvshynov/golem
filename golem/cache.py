"""Expert weight cache with LRU eviction and flat access logging."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import mlx.core as mx
import numpy as np

from .context import get_context
from .lru_policy import LRUPolicy
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AccessRecord:
    """A single cache access event for flat logging.

    This format captures cache behavior with optional context fields
    that can be analyzed post-hoc (hit rates, per-token patterns, etc.).
    """

    layer_idx: int
    expert_id: int
    is_miss: bool
    evicted_id: Optional[int]  # Expert evicted to make room (None if hit or cache had space)
    context: Dict[str, Any] = field(default_factory=dict)  # Snapshot of generation context


class ExpertCache:
    """Cache for expert weights with LRU eviction.

    Uses LRU (Least Recently Used) eviction with frequency-based warmup.
    Access patterns are recorded as flat AccessRecord tuples that can
    be analyzed post-hoc to compute hit rates, eviction patterns, etc.
    """

    def __init__(
        self,
        num_experts: int,
        cache_size: int,
        weights_path: Path,
        layer_idx: int = 0,
        enable_logging: bool = True,
    ):
        """Initialize expert cache.

        Args:
            num_experts: Total number of experts
            cache_size: Maximum number of experts to keep in memory
            weights_path: Path to disk-backed weights directory
            layer_idx: Layer index for disk-backed loading
            enable_logging: Whether to log access records (disable for performance)
        """
        self.num_experts = num_experts
        self.cache_size = min(cache_size, num_experts)
        self.enable_logging = enable_logging

        # Disk-backed mode parameters
        self.weights_path = Path(weights_path)
        self.layer_idx = layer_idx

        # Load metadata to get weight names and shapes
        from .io import load_metadata
        metadata = load_metadata(weights_path, layer_idx)
        self.weight_names = metadata["weight_names"]

        # Store per-expert shapes (without the expert dimension)
        self.expert_shapes = metadata["shapes"]

        # LRU eviction policy for cache management
        self.policy = LRUPolicy(self.cache_size)

        # Mapping: original expert ID -> position in stack [0, cache_size)
        self.expert_to_stack_pos: Dict[int, int] = {}

        # Track free positions in the compact stack
        self.free_positions: Set[int] = set(range(self.cache_size))

        # Pre-allocate compact weight stacks of size [cache_size, ...]
        # We'll load one expert first to get the dtype
        from .io import load_expert_from_disk
        first_expert = load_expert_from_disk(
            expert_id=0,
            weights_path=self.weights_path,
            layer_idx=self.layer_idx,
            weight_names=self.weight_names,
        )

        # Pre-allocate stacks
        self.weight_stacks: Dict[str, mx.array] = {}
        for weight_name in self.weight_names:
            expert_shape = self.expert_shapes[weight_name][1:]  # Remove expert dim
            dtype = first_expert[weight_name].dtype
            # Shape: [cache_size, ...expert_shape]
            stack_shape = [self.cache_size] + list(expert_shape)
            self.weight_stacks[weight_name] = mx.zeros(stack_shape, dtype=dtype)

        # Load the first expert we already fetched into position 0
        stack_pos = self.free_positions.pop()
        for weight_name in self.weight_names:
            self.weight_stacks[weight_name][stack_pos] = first_expert[weight_name]
        mx.eval(list(self.weight_stacks.values()))

        self.policy.add(0)
        self.expert_to_stack_pos[0] = stack_pos

        # Flat access log: list of AccessRecord tuples
        # Format: (layer_idx, expert_id, is_miss, evicted_id)
        self.access_log: List[AccessRecord] = []

    def _load_expert_into_cache(self, expert_id: int):
        """Load expert weights into cache from disk.

        NOTE: Eviction is now handled in get_experts() to ensure transactional loading.
        This method should only be called when there's space in the cache.
        """
        if not self.free_positions:
            raise RuntimeError("No free positions in cache - eviction should have been called")
        stack_pos = self.free_positions.pop()

        from .io import load_expert_from_disk
        expert_weights = load_expert_from_disk(
            expert_id=expert_id,
            weights_path=self.weights_path,
            layer_idx=self.layer_idx,
            weight_names=self.weight_names,
        )

        for weight_name in self.weight_names:
            self.weight_stacks[weight_name][stack_pos] = expert_weights[weight_name]

        mx.eval(list(self.weight_stacks.values()))

        self.policy.add(expert_id)
        self.expert_to_stack_pos[expert_id] = stack_pos

    def _evict_expert(self, expert_id: int):
        """Evict expert from memory cache (low-level: frees stack position).

        Args:
            expert_id: Expert to evict
        """
        if expert_id in self.expert_to_stack_pos:
            stack_pos = self.expert_to_stack_pos[expert_id]
            self.free_positions.add(stack_pos)
            del self.expert_to_stack_pos[expert_id]

    def _evict_if_full(self, exclude: Set[int] = None) -> Optional[int]:
        """Evict one expert if cache is full, otherwise do nothing.

        Args:
            exclude: Set of expert IDs that cannot be evicted (currently needed)

        Returns:
            The evicted expert ID, or None if cache not full or no evictable expert.
        """
        if exclude is None:
            exclude = set()

        if self.policy.is_full():
            evicted_id = self.policy.evict_one(exclude)
            if evicted_id is not None:
                self._evict_expert(evicted_id)
            return evicted_id
        return None

    def get_experts(self, expert_ids: mx.array) -> tuple[Dict[str, mx.array], mx.array]:
        """Get expert weights for given expert IDs.

        Args:
            expert_ids: [batch, seq_len, top_k] expert indices in range [0, num_experts)

        Returns:
            Tuple of:
            - weight_stacks: Dictionary of compact weight tensors with shape [cache_size, ...]
            - remapped_indices: Array with same shape as expert_ids, but values in range [0, cache_size)
        """
        # TODO: This forces mx.eval()
        expert_ids_np = np.array(expert_ids).flatten()
        unique_ids = np.unique(expert_ids_np)

        # Identify hits and misses
        current_needed_set = set(int(eid) for eid in unique_ids)
        experts_to_load = [eid for eid in current_needed_set if eid not in self.policy]
        experts_to_load_set = set(experts_to_load)

        # TRANSACTIONAL LOADING: Evict first, then load
        # Step 1: Evict until we have room (exclude set prevents evicting needed experts)
        evicted_ids = []
        while len(self.policy) + len(experts_to_load) > self.cache_size:
            evicted_id = self.policy.evict_one(exclude=current_needed_set)
            if evicted_id is None:
                break  # Cache too small - all cached experts are needed
            evicted_ids.append(evicted_id)
            self._evict_expert(evicted_id)

        # Step 2: Load missing experts
        for expert_id in experts_to_load:
            if expert_id not in self.policy:
                self._load_expert_into_cache(expert_id)

        # Step 3: Update policy state for all accessed experts (after loading complete)
        for expert_id in current_needed_set:
            if expert_id in self.policy:
                self.policy.access(expert_id)

        # Step 4: Log access records (flat format for post-hoc analysis)
        if self.enable_logging:
            # Capture current generation context
            ctx_snapshot = get_context().get_all()

            # Pair evictions with misses in order; remaining misses have no eviction
            miss_idx = 0
            for eid in unique_ids:
                eid = int(eid)
                is_miss = eid in experts_to_load_set
                if is_miss:
                    evicted_id = evicted_ids[miss_idx] if miss_idx < len(evicted_ids) else None
                    miss_idx += 1
                else:
                    evicted_id = None
                self.access_log.append(AccessRecord(
                    layer_idx=self.layer_idx,
                    expert_id=eid,
                    is_miss=is_miss,
                    evicted_id=evicted_id,
                    context=ctx_snapshot,
                ))

        # Remap expert IDs to stack positions
        original_shape = expert_ids.shape
        expert_ids_flat = np.array(expert_ids).flatten()
        remapped_flat = np.array([self.expert_to_stack_pos[int(eid)] for eid in expert_ids_flat])
        remapped_indices = mx.array(remapped_flat).reshape(original_shape)

        return self.weight_stacks, remapped_indices

    # -------------------------------------------------------------------------
    # Stats derived from access_log (computed on demand)
    # -------------------------------------------------------------------------

    @property
    def total_accesses(self) -> int:
        """Total number of expert accesses."""
        return len(self.access_log)

    @property
    def cache_hits(self) -> int:
        """Number of cache hits."""
        return sum(1 for r in self.access_log if not r.is_miss)

    @property
    def cache_misses(self) -> int:
        """Number of cache misses."""
        return sum(1 for r in self.access_log if r.is_miss)

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        if not self.access_log:
            return 0.0
        return self.cache_hits / len(self.access_log)

    def save_access_log(self, output_path: Path):
        """Save access log to CSV file for analysis.

        Args:
            output_path: Path to save the access log CSV file

        Format:
            CSV with columns: layer_idx, expert_id, is_miss, evicted_id, [context fields...]
            Context fields are added dynamically based on what was logged.
        """
        if not self.access_log:
            logger.warning(f"Layer {self.layer_idx}: Access log is empty, nothing to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all unique context keys across all records
        context_keys = set()
        for record in self.access_log:
            context_keys.update(record.context.keys())
        context_keys = sorted(context_keys)  # Consistent column order

        with open(output_path, 'w') as f:
            # Write header
            header = ["layer_idx", "expert_id", "is_miss", "evicted_id"] + context_keys
            f.write(",".join(header) + "\n")

            # Write records
            for record in self.access_log:
                evicted_str = str(record.evicted_id) if record.evicted_id is not None else ""
                row = [
                    str(record.layer_idx),
                    str(record.expert_id),
                    str(int(record.is_miss)),
                    evicted_str,
                ]
                # Add context values in consistent order
                for key in context_keys:
                    value = record.context.get(key, "")
                    row.append(str(value) if value != "" else "")
                f.write(",".join(row) + "\n")

        logger.info(f"Layer {self.layer_idx}: Saved {len(self.access_log)} access records to {output_path}")

    def warmup_from_access_log(self, access_log_dir: Path, verbose: bool = False):
        """Warm cache using frequency data from access logs.

        Reads the access log for this layer, counts expert frequencies,
        and pre-populates the cache with the most frequent experts.
        Also reorders cache by frequency so least frequent are evicted first.

        Args:
            access_log_dir: Directory containing layer_*_access.csv files
            verbose: If True, log warmup progress
        """
        import csv
        from collections import Counter

        access_log_dir = Path(access_log_dir)
        access_file = access_log_dir / f"layer_{self.layer_idx}_access.csv"

        if not access_file.exists():
            if verbose:
                logger.warning(f"  Layer {self.layer_idx}: No access log found at {access_file}")
            return

        # Count expert frequencies from access log
        frequencies: Counter = Counter()
        with open(access_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                expert_id = int(row["expert_id"])
                frequencies[expert_id] += 1

        if not frequencies:
            return

        # Get top experts by frequency (up to cache_size)
        top_experts = [expert_id for expert_id, _ in frequencies.most_common(self.cache_size)]

        # Load top experts into cache
        loaded_count = 0
        for expert_id in top_experts:
            if expert_id not in self.policy:
                self._evict_if_full()
                self._load_expert_into_cache(expert_id)
                loaded_count += 1

        # Reorder cache by frequency (lowest frequency = evicted first)
        self.policy.reorder_by_priority(frequencies)

        if verbose and loaded_count > 0:
            logger.info(f"  Layer {self.layer_idx}: Warmed cache with {loaded_count} experts from access log")

    def warmup_random(self, seed: int, verbose: bool = False):
        """Warm cache with randomly selected experts.

        Useful as a baseline comparison against frequency-based warmup.
        Each layer uses a different seed derived from the base seed to
        ensure different random selections per layer.

        Args:
            seed: Random seed (combined with layer_idx for per-layer variation)
            verbose: If True, log warmup progress
        """
        import random

        # Use layer-specific seed for reproducibility with variation across layers
        layer_seed = seed + self.layer_idx
        rng = random.Random(layer_seed)

        # Randomly select experts (up to cache_size, excluding expert 0 which is pre-loaded)
        available_experts = [i for i in range(self.num_experts) if i not in self.policy]
        num_to_load = min(len(available_experts), self.cache_size - len(self.policy))

        if num_to_load <= 0:
            return

        selected_experts = rng.sample(available_experts, num_to_load)

        # Load selected experts into cache
        loaded_count = 0
        for expert_id in selected_experts:
            if expert_id not in self.policy:
                self._evict_if_full()
                self._load_expert_into_cache(expert_id)
                loaded_count += 1

        if verbose and loaded_count > 0:
            logger.info(f"  Layer {self.layer_idx}: Warmed cache with {loaded_count} random experts (seed={layer_seed})")


class CachedSwitchMLPWrapper:
    """Wrapper that uses cached expert weights for runtime execution."""

    PROJECTION_NAMES = ["gate_proj", "up_proj", "down_proj"]

    @staticmethod
    def _assign_projection_weights(proj, weights: dict, proj_name: str):
        """Assign weight, scales, and biases to a projection from a weights dict."""
        proj.weight = weights[proj_name]
        scales_key = f"{proj_name}_scales"
        if scales_key in weights:
            proj.scales = weights[scales_key]
        biases_key = f"{proj_name}_biases"
        if biases_key in weights:
            proj.biases = weights[biases_key]

    def __init__(self, original, cache, layer_idx, prompt_handling: str = "full"):
        """Initialize the cached wrapper.

        Args:
            original: Original switch_mlp layer
            cache: ExpertCache instance for this layer
            layer_idx: Layer index for logging
            prompt_handling: How to handle prompt (multi-token) processing:
                - "full": Always load all experts temporarily (default)
                - "adaptive": Use cache if sufficient, otherwise fall back to full
        """
        self.original = original
        self.cache = cache
        self.layer_idx = layer_idx
        self.prompt_handling = prompt_handling

        # The original switch_mlp has empty/uninitialized expert weights
        # We'll replace them dynamically in __call__

    def _forward_with_all_experts(self, x, indices, num_tokens: int):
        """Forward pass loading all experts temporarily.

        Used during prompt processing when multiple tokens are being
        processed. Loads all experts, runs forward pass, then unloads
        to free memory.

        Args:
            x: Input tensor
            indices: Expert indices to use
            num_tokens: Number of tokens being processed (for logging)

        Returns:
            Output tensor from switch_mlp
        """
        from .io import load_all_layer_experts

        logger.info(f"Layer {self.layer_idx}: Prompt processing ({num_tokens} tokens). "
                   f"Temporarily loading all experts.")

        # Load all experts stacked into [num_experts, ...] tensors
        temp_weights = load_all_layer_experts(self.cache.weights_path, self.layer_idx)

        # Save original dummy weights before replacing them
        saved_weights = {}
        for proj_name in self.PROJECTION_NAMES:
            proj = getattr(self.original, proj_name)
            saved_weights[proj_name] = proj.weight
            if hasattr(proj, 'scales'):
                saved_weights[f"{proj_name}_scales"] = proj.scales
            if hasattr(proj, 'biases'):
                saved_weights[f"{proj_name}_biases"] = proj.biases

        # Set weights on the projection layer objects
        for proj_name in self.PROJECTION_NAMES:
            proj = getattr(self.original, proj_name)
            self._assign_projection_weights(proj, temp_weights, proj_name)

        # Forward with original indices (no remapping needed)
        result = self.original(x, indices)

        # Force evaluation before offloading weights
        mx.eval(result)

        # Restore original dummy weights to release references
        for proj_name in self.PROJECTION_NAMES:
            proj = getattr(self.original, proj_name)
            self._assign_projection_weights(proj, saved_weights, proj_name)

        # Explicitly delete temporary weights to free memory
        del temp_weights
        del saved_weights
        mx.clear_cache()

        logger.info(f"Layer {self.layer_idx}: Prompt processing complete, experts unloaded.")

        return result

    def _cache_sufficient_for_indices(self, indices) -> bool:
        """Check if the cache can hold all unique experts needed for these indices."""
        import numpy as np
        expert_ids_np = np.array(indices).flatten()
        unique_count = len(np.unique(expert_ids_np))
        return unique_count <= self.cache.cache_size

    def __call__(self, x, indices):
        """Forward pass with cached expert weights.

        Args:
            x: Input tensor
            indices: Expert indices to use

        Returns:
            Output tensor from switch_mlp
        """
        # Check if processing multiple tokens (prompt processing)
        # indices shape is (batch_size, seq_len, experts_per_token)
        num_tokens = indices.shape[0] * indices.shape[1]

        if num_tokens > 1:
            # Multi-token (prompt) processing
            if self.prompt_handling == "full":
                return self._forward_with_all_experts(x, indices, num_tokens)
            elif self.prompt_handling == "adaptive":
                # Use cache if it can hold all needed experts, otherwise fall back to full
                if self._cache_sufficient_for_indices(indices):
                    logger.debug(f"Layer {self.layer_idx}: Prompt ({num_tokens} tokens) fits in cache")
                else:
                    return self._forward_with_all_experts(x, indices, num_tokens)

        # Normal token generation (or adaptive prompt that fits) - use cache-based approach
        cached_weights, remapped_indices = self.cache.get_experts(indices)

        for proj_name in self.PROJECTION_NAMES:
            proj = getattr(self.original, proj_name)
            self._assign_projection_weights(proj, cached_weights, proj_name)

        return self.original(x, remapped_indices)
