"""Disk I/O utilities and path management for expert weights."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx

from .logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Path Utilities
# =============================================================================


def get_default_cache_path(model_path: Path) -> Path:
    """Get default cache directory for a model.

    Args:
        model_path: Path to the model directory

    Returns:
        Path to cache directory in ~/.golem/<model_hash>
    """
    # Resolve to absolute path for consistent hashing
    model_path = Path(model_path).resolve()

    # Create hash of model path for unique cache directory
    path_str = str(model_path)
    path_hash = hashlib.sha256(path_str.encode()).hexdigest()[:16]

    # Use model name if it's a directory, otherwise use hash
    if model_path.is_dir():
        model_name = model_path.name
    else:
        model_name = path_hash

    # Create cache path: ~/.golem/<model_name>_<hash>
    cache_dir = Path.home() / ".golem" / f"{model_name}_{path_hash}"

    return cache_dir


def get_default_state_path(model_path: Path) -> Path:
    """Get default cache state file path for a model.

    Args:
        model_path: Path to the model directory

    Returns:
        Path to cache state file in ~/.golem/<model_name>-expert-state.json
    """
    # Resolve to absolute path
    model_path = Path(model_path).resolve()

    # Use model name for state file
    if model_path.is_dir():
        model_name = model_path.name
    else:
        # Use hash if not a directory
        path_str = str(model_path)
        path_hash = hashlib.sha256(path_str.encode()).hexdigest()[:16]
        model_name = path_hash

    # Create state path: ~/.golem/<model_name>-expert-state.json
    state_dir = Path.home() / ".golem"
    state_dir.mkdir(parents=True, exist_ok=True)

    state_file = state_dir / f"{model_name}-expert-state.json"

    return state_file


def get_cache_info(weights_path: Path) -> dict:
    """Get information about cached expert weights.

    Args:
        weights_path: Path to exported weights directory

    Returns:
        Dictionary with cache information
    """
    weights_path = Path(weights_path)
    metadata_file = weights_path / "model_metadata.json"

    if not metadata_file.exists():
        return {
            "exists": False,
            "path": str(weights_path),
        }

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Calculate disk usage
    total_size = sum(f.stat().st_size for f in weights_path.rglob("*.safetensors"))

    return {
        "exists": True,
        "path": str(weights_path),
        "num_layers": len(metadata.get("moe_layer_indices", [])),
        "num_experts": metadata.get("num_experts"),
        "size_gb": total_size / (1024**3),
        "metadata": metadata,
    }


# =============================================================================
# Disk I/O for Expert Weights
# =============================================================================


def save_expert_weights_to_disk(
    expert_weights: Dict[str, mx.array],
    save_path: Path,
    layer_idx: int,
) -> Path:
    """Save expert weights to disk using MLX's built-in save.

    Args:
        expert_weights: Dictionary with keys like "gate_proj", "up_proj", "down_proj"
                       Each value is [num_experts, ...] shaped array
        save_path: Directory to save weights
        layer_idx: Layer index for filename

    Returns:
        Path to saved file
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create filename for this layer
    weights_file = save_path / f"layer_{layer_idx}_experts.safetensors"

    # Flatten the expert weights into individual expert tensors
    # Format: expert_0_gate_proj, expert_0_up_proj, etc.
    num_experts = expert_weights["gate_proj"].shape[0]
    flat_weights = {}

    for expert_id in range(num_experts):
        for weight_name, weights in expert_weights.items():
            key = f"expert_{expert_id}_{weight_name}"
            flat_weights[key] = weights[expert_id]

    # Save using MLX's safetensors save function
    mx.save_safetensors(str(weights_file), flat_weights)

    # Save metadata (including dtypes for quantized weights)
    metadata = {
        "num_experts": num_experts,
        "layer_idx": layer_idx,
        "weight_names": list(expert_weights.keys()),
        "shapes": {name: list(arr.shape) for name, arr in expert_weights.items()},
        "dtypes": {name: str(arr.dtype) for name, arr in expert_weights.items()},
    }
    metadata_file = save_path / f"layer_{layer_idx}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return weights_file


def load_expert_from_disk(
    expert_id: int,
    weights_path: Path,
    layer_idx: int,
    weight_names: Optional[List[str]] = None,
) -> Dict[str, mx.array]:
    """Load a single expert's weights from disk.

    Args:
        expert_id: Expert ID to load
        weights_path: Path to saved weights file or directory
        layer_idx: Layer index
        weight_names: List of weight names to load (e.g., ["gate_proj", "up_proj", "down_proj"])
                     If None, loads metadata to determine names

    Returns:
        Dictionary mapping weight names to arrays
    """
    weights_path = Path(weights_path)

    # If weights_path is a directory, construct the file path
    if weights_path.is_dir():
        weights_file = weights_path / f"layer_{layer_idx}_experts.safetensors"
        metadata_file = weights_path / f"layer_{layer_idx}_metadata.json"
    else:
        weights_file = weights_path
        metadata_file = weights_path.parent / f"layer_{layer_idx}_metadata.json"

    # Load metadata if weight_names not provided
    if weight_names is None:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        weight_names = metadata["weight_names"]

    # Load all weights (safetensors uses memory mapping, so this is efficient)
    all_weights = mx.load(str(weights_file))

    # Extract this expert's weights
    expert_weights = {}
    for weight_name in weight_names:
        key = f"expert_{expert_id}_{weight_name}"
        expert_weights[weight_name] = all_weights[key]

    return expert_weights


def load_metadata(weights_path: Path, layer_idx: int) -> Dict:
    """Load metadata for a saved expert weights file.

    Args:
        weights_path: Path to weights directory
        layer_idx: Layer index

    Returns:
        Metadata dictionary
    """
    weights_path = Path(weights_path)
    metadata_file = weights_path / f"layer_{layer_idx}_metadata.json"

    with open(metadata_file, "r") as f:
        return json.load(f)


def load_all_layer_experts(weights_path: Path, layer_idx: int) -> Dict[str, mx.array]:
    """Load all expert weights for a layer, stacked into [num_experts, ...] tensors.

    Args:
        weights_path: Path to weights directory
        layer_idx: Layer index

    Returns:
        Dictionary mapping weight names to stacked arrays with shape [num_experts, ...]
    """
    weights_path = Path(weights_path)
    metadata = load_metadata(weights_path, layer_idx)
    num_experts = metadata["num_experts"]
    weight_names = metadata["weight_names"]

    weights_file = weights_path / f"layer_{layer_idx}_experts.safetensors"
    all_weights = mx.load(str(weights_file))

    stacked = {}
    for weight_name in weight_names:
        expert_tensors = [all_weights[f"expert_{i}_{weight_name}"]
                          for i in range(num_experts)]
        stacked[weight_name] = mx.stack(expert_tensors, axis=0)
        mx.eval(stacked[weight_name])

    return stacked


# =============================================================================
# Expert Export
# =============================================================================


def export_experts_from_safetensors(
    model_path: Path,
    save_path: Path,
    verbose: bool = False,
) -> List[int]:
    """Export expert weights directly from model safetensors without loading full model.

    This is memory-efficient for large models - only loads weights for one layer at a time.
    Supports multiple MoE architectures (Qwen3, GLM-4, etc.) via model config system.

    Args:
        model_path: Path to model directory containing config.json and safetensors
        save_path: Directory to save expert weights
        verbose: Print detailed progress

    Returns:
        List of layer indices that were exported
    """
    from .model_configs import get_model_config

    model_path = Path(model_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Get model-specific config
    model_config = get_model_config(model_path, verbose=verbose)

    if verbose:
        logger.info(f"\nModel config: {model_config}")
        logger.info(f"  Layers: {model_config.num_hidden_layers}")
        logger.info(f"  Experts per layer: {model_config.num_experts}")
        logger.info(f"  Experts per token: {model_config.num_experts_per_tok}")

    # Load safetensors index
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        # Single file model
        safetensor_file = model_path / "model.safetensors"
        if safetensor_file.exists():
            weight_map = {k: "model.safetensors" for k in mx.load(str(safetensor_file)).keys()}
        else:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
    else:
        with open(index_file) as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]

    # Detect MoE layers using model-specific logic
    moe_layers = model_config.detect_moe_layers(weight_map)

    if verbose:
        logger.info(f"\nDetected MoE layers: {moe_layers} (total: {len(moe_layers)})")

    saved_layers = []

    # Process each MoE layer
    for layer_idx in moe_layers:
        if verbose:
            logger.info(f"\nProcessing layer {layer_idx}...")

        # Find which safetensor files contain this layer's expert weights
        layer_prefix = model_config.get_expert_weight_pattern(layer_idx)
        layer_weight_keys = [k for k in weight_map.keys() if k.startswith(layer_prefix)]

        if not layer_weight_keys:
            logger.warning(f"No expert weights found for layer {layer_idx}, skipping")
            continue

        # Get unique safetensor files needed for this layer
        needed_files = set(weight_map[k] for k in layer_weight_keys)

        if verbose:
            logger.info(f"  Loading from: {needed_files}")

        # Load weights from needed files
        all_weights = {}
        for filename in needed_files:
            filepath = model_path / filename
            if verbose:
                logger.debug(f"  Loading {filename}...")
            weights = mx.load(str(filepath))
            all_weights.update(weights)

        # Extract expert weights for this layer
        expert_weights = {}

        # Get weight components and suffixes from model config
        weight_components = model_config.get_weight_components()
        suffixes = model_config.get_weight_suffixes()

        for component in weight_components:
            for suffix in suffixes:
                key = f"{layer_prefix}.{component}.{suffix}"
                if key in all_weights:
                    # Store with simplified name (e.g., "gate_proj" or "gate_proj_scales")
                    if suffix == "weight":
                        store_key = component
                    else:
                        store_key = f"{component}_{suffix}"
                    expert_weights[store_key] = all_weights[key]

        if not expert_weights:
            logger.warning(f"No expert weights extracted for layer {layer_idx}, skipping")
            continue

        # Verify shapes
        if verbose:
            logger.debug("  Extracted weights:")
            for k, v in expert_weights.items():
                logger.debug(f"    {k}: shape={v.shape}, dtype={v.dtype}")

        # Save to disk
        save_expert_weights_to_disk(expert_weights, save_path, layer_idx)
        saved_layers.append(layer_idx)

        logger.info(f"  ✓ Saved layer {layer_idx} experts")

        # Clear loaded weights to free memory
        del all_weights
        del expert_weights

    # Save global metadata
    global_metadata = {
        "moe_layer_indices": saved_layers,
        "num_experts": model_config.num_experts,
        "num_experts_per_tok": model_config.num_experts_per_tok,
        "model_name": model_config.model_type,
        "model_config_class": model_config.__class__.__name__,
    }
    with open(save_path / "model_metadata.json", "w") as f:
        json.dump(global_metadata, f, indent=2)

    logger.info(f"\n✓ Exported {len(saved_layers)} MoE layers to {save_path}")
    return saved_layers
