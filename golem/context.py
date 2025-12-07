"""Generation context for tracking state across cache layers."""

from typing import Any, Dict, Optional


class GenerationContext:
    """Shared context for tracking generation state across cache layers.

    This allows access records to be associated with specific tokens
    and include arbitrary metadata for post-hoc analysis.

    Usage:
        ctx = GenerationContext()
        ctx.set("token_idx", 0)
        ctx.set("prompt_phase", True)

        # In cache layer:
        token_idx = ctx.get("token_idx")

        # After each token:
        ctx.increment("token_idx")
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._data.get(key, default)

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric context value and return new value."""
        current = self._data.get(key, 0)
        new_value = current + amount
        self._data[key] = new_value
        return new_value

    def get_all(self) -> Dict[str, Any]:
        """Get all context values as a dict."""
        return dict(self._data)

    def clear(self) -> None:
        """Clear all context values."""
        self._data.clear()

    def reset(self, **initial_values) -> None:
        """Clear and set initial values."""
        self._data.clear()
        self._data.update(initial_values)


# Global context instance (can be replaced for testing)
_global_context: Optional[GenerationContext] = None


def get_context() -> GenerationContext:
    """Get the global generation context, creating if needed."""
    global _global_context
    if _global_context is None:
        _global_context = GenerationContext()
    return _global_context


def set_context(ctx: Optional[GenerationContext]) -> None:
    """Set the global generation context (for testing or custom contexts)."""
    global _global_context
    _global_context = ctx
