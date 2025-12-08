"""LRU (Least Recently Used) eviction policy for expert weight caching."""

from collections import OrderedDict
from typing import Dict, Optional, Set


class LRUPolicy:
    """Least Recently Used eviction policy.

    Evicts the item that hasn't been accessed for the longest time.
    Uses OrderedDict for O(1) operations.
    """

    def __init__(self, capacity: int):
        """Initialize the policy.

        Args:
            capacity: Maximum number of items the cache can hold
        """
        self.capacity = capacity
        # OrderedDict maintains insertion order; we move accessed items to end
        self._cache: OrderedDict[int, None] = OrderedDict()

    def __contains__(self, key: int) -> bool:
        """Check if key is in cache."""
        return key in self._cache

    def __len__(self) -> int:
        """Number of items currently in cache."""
        return len(self._cache)

    def access(self, key: int) -> None:
        """Move key to MRU position (end of OrderedDict)."""
        self._cache.move_to_end(key)

    def add(self, key: int) -> None:
        """Add key at MRU position."""
        self._cache[key] = None

    def remove(self, key: int) -> None:
        """Remove key from cache."""
        del self._cache[key]

    def evict_one(self, exclude: Set[int]) -> Optional[int]:
        """Evict LRU item not in exclude set."""
        for key in self._cache:  # Iterates from oldest to newest
            if key not in exclude:
                del self._cache[key]
                return key
        return None

    def is_full(self) -> bool:
        """Check if cache is at capacity."""
        return len(self) >= self.capacity

    def reorder_by_priority(self, priority: Dict[int, float]) -> None:
        """Reorder cached keys by priority (lowest priority = evicted first).

        Used for frequency-based warmup: keys with higher frequency get
        higher priority and are placed at the end (MRU position).

        Args:
            priority: Dict mapping key to priority value (e.g., frequency count)
        """
        if not self._cache:
            return
        # Sort by priority (lowest first, so highest ends up at MRU position)
        sorted_keys = sorted(self._cache.keys(), key=lambda k: priority.get(k, 0))
        new_cache: OrderedDict[int, None] = OrderedDict()
        for k in sorted_keys:
            new_cache[k] = None
        self._cache = new_cache
