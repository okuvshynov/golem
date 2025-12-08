"""Qwen3 MoE model configuration."""

from pathlib import Path
from typing import Dict, List, Optional



class Qwen3ModelConfig:
    """Configuration for Qwen3 MoE models (Qwen3-30B, Qwen3-235B, etc.)."""

    def __init__(self, config: Dict, model_path: Optional[Path] = None):
        self.config = config
        self.model_path = Path(model_path) if model_path else None
        self.model_type = config.get("model_type", "qwen3_moe")

    @property
    def num_experts(self) -> int:
        return self.config.get("num_experts", 0)

    @property
    def num_experts_per_tok(self) -> int:
        return self.config.get("num_experts_per_tok", 4)

    @property
    def num_hidden_layers(self) -> int:
        return self.config.get("num_hidden_layers", 0)

    def detect_moe_layers(self, weight_map: Dict[str, str]) -> List[int]:
        """Detect MoE layers using Qwen3's decoder_sparse_step pattern."""
        decoder_sparse_step = self.config.get("decoder_sparse_step", 1)
        mlp_only_layers = self.config.get("mlp_only_layers", [])

        moe_layers = []
        for layer_idx in range(self.num_hidden_layers):
            is_moe = (
                layer_idx not in mlp_only_layers
                and self.num_experts > 0
                and (layer_idx + 1) % decoder_sparse_step == 0
            )
            if is_moe:
                prefix = self.get_expert_weight_pattern(layer_idx)
                if any(k.startswith(prefix) for k in weight_map.keys()):
                    moe_layers.append(layer_idx)
        return moe_layers

    def get_expert_weight_pattern(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}.mlp.switch_mlp"

    def get_weight_components(self) -> List[str]:
        return ["gate_proj", "up_proj", "down_proj"]

    def get_weight_suffixes(self) -> List[str]:
        return ["weight", "scales", "biases"]

    def __repr__(self) -> str:
        return f"Qwen3ModelConfig(num_experts={self.num_experts}, layers={self.num_hidden_layers})"
