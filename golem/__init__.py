"""Golem-MLX: Expert offloading for memory-constrained MoE models."""

# Cache module: runtime caching of expert weights
from .cache import AccessRecord, CachedSwitchMLPWrapper, ExpertCache

# LRU eviction policy
from .lru_policy import LRUPolicy

# Generation context: shared state for access logging
from .context import GenerationContext, get_context, set_context

# I/O module: disk operations and path utilities
from .io import (
    get_cache_info,
    get_default_cache_path,
    get_default_state_path,
    load_all_layer_experts,
    load_expert_from_disk,
    load_metadata,
    save_expert_weights_to_disk,
)

# Model module: model loading and expert installation
from .model import (
    ensure_experts_exported,
    get_moe_layer_indices,
    install_expert_caches,
    install_fully_loaded_layers,
    load_all_experts_for_layer,
    load_model_without_experts,
)

__all__ = [
    # Cache classes
    "AccessRecord",
    "CachedSwitchMLPWrapper",
    "ExpertCache",
    # LRU eviction policy
    "LRUPolicy",
    # Generation context
    "GenerationContext",
    "get_context",
    "set_context",
    # I/O functions
    "get_cache_info",
    "get_default_cache_path",
    "get_default_state_path",
    "load_all_layer_experts",
    "load_expert_from_disk",
    "load_metadata",
    "save_expert_weights_to_disk",
    # Model functions
    "ensure_experts_exported",
    "get_moe_layer_indices",
    "install_expert_caches",
    "install_fully_loaded_layers",
    "load_all_experts_for_layer",
    "load_model_without_experts",
]
