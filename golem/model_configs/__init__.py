"""Model configuration registry for different MoE architectures.

This module provides a registry of model-specific configurations that handle
differences in MoE layer detection, config key mappings, and weight patterns
across different model architectures.

Supported models:
- Qwen3 MoE (qwen3_moe, qwen2moe)
- GLM-4 MoE (glm4_moe, chatglm)
- MiniMax MoE (minimax)

Usage:
    from golem.model_configs import get_model_config

    config = get_model_config(model_path)
    num_experts = config.num_experts
    moe_layers = config.detect_moe_layers(weight_map)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Type

from .glm4 import GLM4ModelConfig
from .minimax import MiniMaxModelConfig
from .qwen3 import Qwen3ModelConfig
from ..logging_config import get_logger

logger = get_logger(__name__)

# Registry mapping model_type to config class
_MODEL_CONFIG_REGISTRY: Dict[str, Type] = {
    "qwen3_moe": Qwen3ModelConfig,
    "qwen2moe": Qwen3ModelConfig,
    "glm4_moe": GLM4ModelConfig,
    "chatglm": GLM4ModelConfig,
    "minimax": MiniMaxModelConfig,
}


def register_model_config(model_type: str, config_class: Type):
    """Register a new model config class."""
    _MODEL_CONFIG_REGISTRY[model_type] = config_class


def get_model_config(
    model_path: Path,
    config_dict: Optional[Dict] = None,
    verbose: bool = False,
):
    """Get appropriate model config for the given model.

    Args:
        model_path: Path to model directory
        config_dict: Optional pre-loaded config dictionary
        verbose: Print detection information

    Returns:
        Model-specific config instance

    Raises:
        ValueError: If model_type is not supported
    """
    model_path = Path(model_path)

    if config_dict is None:
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No config.json found at {model_path}")
        with open(config_file) as f:
            config_dict = json.load(f)

    model_type = config_dict.get("model_type", "unknown")
    config_class = _MODEL_CONFIG_REGISTRY.get(model_type)

    if config_class:
        if verbose:
            logger.info(f"Using {config_class.__name__} for model_type='{model_type}'")
        return config_class(config_dict, model_path)
    else:
        supported = list(_MODEL_CONFIG_REGISTRY.keys())
        raise ValueError(f"Unsupported model_type='{model_type}'. Supported: {supported}")


__all__ = [
    "Qwen3ModelConfig",
    "GLM4ModelConfig",
    "MiniMaxModelConfig",
    "get_model_config",
    "register_model_config",
]
