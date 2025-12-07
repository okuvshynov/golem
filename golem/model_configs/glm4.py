"""GLM-4 MoE model configuration."""

from pathlib import Path
from typing import Dict, List, Optional



class GLM4ModelConfig:
    """Configuration for GLM-4 MoE models (GLM-4-9B, GLM-4.6-5bit, etc.)."""

    def __init__(self, config: Dict, model_path: Optional[Path] = None):
        self.config = config
        self.model_path = Path(model_path) if model_path else None
        self.model_type = config.get("model_type", "glm4_moe")

    @property
    def num_experts(self) -> int:
        # GLM-4 uses "n_routed_experts" instead of "num_experts"
        return self.config.get("n_routed_experts", 0)

    @property
    def num_experts_per_tok(self) -> int:
        return self.config.get("num_experts_per_tok", 8)

    @property
    def num_hidden_layers(self) -> int:
        return self.config.get("num_hidden_layers", 0)

    def detect_moe_layers(self, weight_map: Dict[str, str]) -> List[int]:
        """Detect MoE layers using GLM-4's first_k_dense_replace pattern."""
        first_k_dense_replace = self.config.get("first_k_dense_replace", 0)

        moe_layers = []
        for layer_idx in range(self.num_hidden_layers):
            is_moe = layer_idx >= first_k_dense_replace
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
        return f"GLM4ModelConfig(num_experts={self.num_experts}, layers={self.num_hidden_layers})"
