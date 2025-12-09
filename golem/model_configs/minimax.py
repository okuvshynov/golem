"""MiniMax MoE model configuration."""

from pathlib import Path
from typing import Dict, List, Optional


class MiniMaxModelConfig:
    """Configuration for MiniMax MoE models (MiniMax-M1, etc.)."""

    def __init__(self, config: Dict, model_path: Optional[Path] = None):
        self.config = config
        self.model_path = Path(model_path) if model_path else None
        self.model_type = config.get("model_type", "minimax")

    @property
    def num_experts(self) -> int:
        # MiniMax uses "num_local_experts" instead of "num_experts"
        return self.config.get("num_local_experts", 0)

    @property
    def num_experts_per_tok(self) -> int:
        return self.config.get("num_experts_per_tok", 8)

    @property
    def num_hidden_layers(self) -> int:
        return self.config.get("num_hidden_layers", 0)

    def detect_moe_layers(self, weight_map: Dict[str, str]) -> List[int]:
        """Detect MoE layers - all layers in MiniMax are MoE."""
        moe_layers = []
        for layer_idx in range(self.num_hidden_layers):
            prefix = self.get_expert_weight_pattern(layer_idx)
            if any(k.startswith(prefix) for k in weight_map.keys()):
                moe_layers.append(layer_idx)
        return moe_layers

    def get_expert_weight_pattern(self, layer_idx: int) -> str:
        # MiniMax uses block_sparse_moe.switch_mlp instead of mlp.switch_mlp
        return f"model.layers.{layer_idx}.block_sparse_moe.switch_mlp"

    def get_weight_components(self) -> List[str]:
        return ["gate_proj", "up_proj", "down_proj"]

    def get_weight_suffixes(self) -> List[str]:
        return ["weight", "scales", "biases"]

    def __repr__(self) -> str:
        return f"MiniMaxModelConfig(num_experts={self.num_experts}, layers={self.num_hidden_layers})"
