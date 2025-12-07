"""Model loading and expert installation for MoE models."""

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model

from .logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Model Utilities
# =============================================================================


def get_moe_layer_indices(model) -> List[int]:
    """Get indices of MoE layers in model.

    Model-agnostic: inspects each layer directly to detect MoE.

    Args:
        model: MLX-LM model

    Returns:
        List of layer indices that have MoE
    """
    args = model.args
    moe_layers = []

    for layer_idx in range(args.num_hidden_layers):
        layer = model.model.layers[layer_idx]

        # Check if this layer has MoE by inspecting structure
        has_moe = hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp")

        if has_moe:
            moe_layers.append(layer_idx)

    return moe_layers


def ensure_experts_exported(
    model_path: Path,
    weights_path: Optional[Path] = None,
    verbose: bool = False,
) -> Path:
    """Ensure expert weights are exported, auto-exporting if needed.

    Args:
        model_path: Path to the model directory
        weights_path: Optional custom path for exported weights.
                     If None, uses default cache location.
        verbose: Print detailed progress

    Returns:
        Path to the exported weights directory

    Raises:
        FileNotFoundError: If model_path doesn't exist
    """
    from .io import get_default_cache_path

    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Determine weights path
    if weights_path is None:
        weights_path = get_default_cache_path(model_path)
        if verbose:
            logger.info(f"Using default cache path: {weights_path}")
    else:
        weights_path = Path(weights_path).resolve()

    # Check if already exported
    metadata_file = weights_path / "model_metadata.json"

    if metadata_file.exists():
        if verbose:
            logger.info(f"✓ Expert weights already exported to: {weights_path}")
        return weights_path

    # Need to export
    logger.info(f"Expert weights not found, exporting from {model_path}...")
    logger.info(f"Export destination: {weights_path}")

    from .io import export_experts_from_safetensors

    start_time = time.time()

    saved_layers = export_experts_from_safetensors(
        model_path=model_path,
        save_path=weights_path,
        verbose=verbose,
    )

    export_time = time.time() - start_time

    logger.info(f"✓ Export complete in {export_time:.2f}s ({len(saved_layers)} layers)")

    return weights_path


# =============================================================================
# Model Loading
# =============================================================================


def load_model_without_experts(
    model_path: Path,
    verbose: bool = False,
) -> Tuple[nn.Module, object, dict]:
    """Load MLX model without loading expert weights.

    This is memory-efficient for large MoE models - loads only ~26GB for Qwen3-235B
    instead of ~242GB. All expert weights are replaced with tiny dummy placeholders.
    Use load_all_experts_for_layer() or install_expert_caches() after loading to
    make experts available.

    Supports multiple MoE architectures via model config system.

    Args:
        model_path: Path to model directory
        verbose: Print progress

    Returns:
        (model, tokenizer, config) - model has dummy expert weights that need to be
        filled via load_all_experts_for_layer() or cached via install_expert_caches()
    """
    from .model_configs import get_model_config

    model_path = Path(model_path)

    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # Get model-specific config
    model_config = get_model_config(model_path, config_dict=config, verbose=False)

    if verbose:
        logger.info(f"Model: {model_config.model_type}")
        logger.info(f"Layers: {model_config.num_hidden_layers}")
        logger.info(f"Experts: {model_config.num_experts}")
        logger.info("Skipping all expert weights (use load_all_experts_for_layer or install_expert_caches after)")

    # Use selective loading to skip expert weights
    from safetensors import safe_open

    # Try to use PyTorch backend for bfloat16 support
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        framework = "pt"
        use_torch = True
    else:
        framework = "numpy"
        use_torch = False
        if verbose:
            logger.warning("PyTorch not available, using numpy backend (may fail on bfloat16)")

    # Store original mx.load
    original_mx_load = mx.load

    # Create filtered loading function that selectively loads non-expert weights
    def filtered_load(path, return_metadata=False):
        """Load safetensors but replace ALL expert weights with tiny dummy placeholders.

        Uses selective loading to NEVER load expert weights into memory during initial load.
        """
        weights = {}
        metadata = {}
        expert_count = 0
        loaded_count = 0

        with safe_open(str(path), framework=framework) as f:
            # Get metadata if requested
            if return_metadata:
                metadata = f.metadata() if hasattr(f, 'metadata') else {}

            # Process each tensor selectively
            for key in f.keys():
                is_expert = "switch_mlp" in key

                if not is_expert:
                    # Load non-expert weights only
                    tensor = f.get_tensor(key)

                    # Convert to MLX array
                    if use_torch:
                        import torch as torch_module
                        # Handle bfloat16: convert to float32 for numpy, then back to bfloat16
                        is_bfloat16 = tensor.dtype == torch_module.bfloat16
                        if is_bfloat16:
                            tensor = tensor.to(torch_module.float32)
                        numpy_tensor = tensor.cpu().numpy()
                        mlx_array = mx.array(numpy_tensor)
                        # Convert back to bfloat16 to preserve original dtype
                        if is_bfloat16:
                            mlx_array = mlx_array.astype(mx.bfloat16)
                        weights[key] = mlx_array
                    else:
                        weights[key] = mx.array(tensor)

                    loaded_count += 1
                else:
                    # Create minimal dummy placeholder (not full-sized zero array)
                    # This satisfies the model loader but uses negligible memory
                    # Will be replaced by load_all_experts_for_layer() or ExpertCache
                    weights[key] = mx.array([0.0])  # Tiny 1-element array
                    expert_count += 1

        if verbose:
            logger.debug(f"    Loaded {loaded_count} non-expert tensors, skipped {expert_count} expert tensors")

        if return_metadata:
            return weights, metadata
        return weights

    # Monkey-patch mx.load temporarily
    mx.load = filtered_load

    try:
        # Use load_model with strict=False to allow minimal placeholders
        if verbose:
            logger.info("Loading model with filtered weights...")

        model, _ = load_model(model_path, strict=False)

        # Load tokenizer separately
        tokenizer = load_tokenizer(model_path)

        if verbose:
            logger.info("Model loaded successfully (all expert weights use minimal placeholders)")

        return model, tokenizer, config

    finally:
        # Restore original mx.load
        mx.load = original_mx_load


# =============================================================================
# Expert Installation
# =============================================================================


def load_all_experts_for_layer(
    model: nn.Module,
    weights_path: Path,
    layer_idx: int,
    verbose: bool = False,
) -> None:
    """Load all expert weights for a specific MoE layer from disk.

    This replaces the dummy expert weight placeholders with actual weights
    loaded from exported expert files. Use this when you want a layer to have
    all experts resident in memory (no caching).

    Args:
        model: Model with dummy expert weights
        weights_path: Path to exported expert weights directory
        layer_idx: Layer index to fully load
        verbose: Print progress
    """
    from .io import load_all_layer_experts, load_metadata

    weights_path = Path(weights_path)

    # Load all experts stacked into [num_experts, ...] tensors
    stacked_weights = load_all_layer_experts(weights_path, layer_idx)

    # Replace the switch_mlp weights with the fully loaded weights
    layer = model.model.layers[layer_idx]
    switch_mlp = layer.mlp.switch_mlp

    # Set weights on the projection layer objects (not replace the layers themselves)
    # The projections (gate_proj, up_proj, down_proj) are layer objects with .weight attribute
    for weight_name, weight_array in stacked_weights.items():
        # Handle different weight types: main weights, scales (quantized), biases (quantized)
        if weight_name.endswith("_scales"):
            proj_name = weight_name.replace("_scales", "")
            proj = getattr(switch_mlp, proj_name)
            proj.scales = weight_array
        elif weight_name.endswith("_biases"):
            proj_name = weight_name.replace("_biases", "")
            proj = getattr(switch_mlp, proj_name)
            proj.biases = weight_array
        else:
            # Main weight (gate_proj, up_proj, down_proj)
            proj = getattr(switch_mlp, weight_name)
            proj.weight = weight_array

    if verbose:
        metadata = load_metadata(weights_path, layer_idx)
        num_experts = metadata["num_experts"]
        weight_names = metadata["weight_names"]
        total_tensors = len(weight_names) * num_experts
        logger.info(f"  Layer {layer_idx}: loaded {total_tensors} expert weight tensors ({num_experts} experts × {len(weight_names)} weights)")


def install_fully_loaded_layers(
    model: nn.Module,
    weights_path: Path,
    fully_load_layers: list[int],
    verbose: bool = False,
) -> None:
    """Load all expert weights for specified MoE layers.

    This is useful for layers where you want all experts resident in memory
    instead of using caching. Typically used for layers with diverse access
    patterns where LRU caching would be inefficient.

    Args:
        model: Model with dummy expert weights
        weights_path: Path to model directory with safetensors files
        fully_load_layers: List of layer indices to fully load
        verbose: Print progress
    """
    if not fully_load_layers:
        return

    moe_layers = get_moe_layer_indices(model)

    # Validate that all specified layers are actually MoE layers
    invalid_layers = [idx for idx in fully_load_layers if idx not in moe_layers]
    if invalid_layers:
        raise ValueError(f"Invalid layer indices (not MoE layers): {invalid_layers}")

    # Get num_experts model-agnostically
    num_experts = (
        getattr(model.args, 'n_routed_experts', None) or  # GLM-4
        getattr(model.args, 'num_experts', None) or       # Qwen3, Mixtral
        0
    )

    if verbose:
        logger.info(f"Loading all experts for {len(fully_load_layers)} layers: {fully_load_layers}")
        total_experts = len(fully_load_layers) * num_experts
        logger.info(f"  Total experts to load: {total_experts}")

    # Load all experts for each layer
    for layer_idx in fully_load_layers:
        load_all_experts_for_layer(model, weights_path, layer_idx, verbose)

    if verbose:
        logger.info(f"Successfully loaded all experts for {len(fully_load_layers)} layers")


def install_expert_caches(
    model: nn.Module,
    weights_path: Path,
    cache_size: int = 32,
    cache_size_overrides: Optional[dict[int, int]] = None,
    verbose: bool = False,
    fully_load_layers: Optional[list[int]] = None,
    enable_logging: bool = True,
    prompt_handling: str = "full",
) -> list:
    """Install expert caches for MoE layers.

    Creates caches that load expert weights on-demand from disk. This is
    memory-efficient for layers with focused expert usage patterns.

    Args:
        model: Model with dummy expert weights
        weights_path: Path to exported expert weights directory
        cache_size: Number of experts to cache per layer (default)
        cache_size_overrides: Optional dict mapping layer indices to custom cache sizes
        verbose: Print progress
        fully_load_layers: Optional list of layer indices to skip (because they
            were already fully loaded via install_fully_loaded_layers()).
        enable_logging: Whether to log cache access records (disable for performance)
        prompt_handling: How to handle prompt processing ('full' or 'adaptive')

    Returns:
        List of expert caches (one per cached MoE layer, excluding fully loaded layers)
    """
    from .cache import CachedSwitchMLPWrapper, ExpertCache

    fully_load_layers = fully_load_layers or []
    moe_layers = get_moe_layer_indices(model)

    # Skip cache installation for fully loaded layers
    cached_layers = [idx for idx in moe_layers if idx not in fully_load_layers]

    cache_size_overrides = cache_size_overrides or {}

    if verbose:
        if fully_load_layers:
            logger.info(f"Skipping cache installation for {len(fully_load_layers)} fully loaded layers: {fully_load_layers}")
        logger.info(f"Installing expert caches for {len(cached_layers)} MoE layers...")
        logger.info(f"Default cache size: {cache_size} experts per layer")
        if cache_size_overrides:
            logger.info(f"Cache size overrides: {cache_size_overrides}")

    expert_caches = []

    # Get num_experts model-agnostically
    num_experts = (
        getattr(model.args, 'n_routed_experts', None) or  # GLM-4
        getattr(model.args, 'num_experts', None) or       # Qwen3, Mixtral
        0
    )

    # Install cache for each MoE layer (except fully loaded ones)
    for layer_idx in cached_layers:
        layer = model.model.layers[layer_idx]

        # Use per-layer cache size if override specified, otherwise default
        layer_cache_size = cache_size_overrides.get(layer_idx, cache_size)

        # Create expert cache
        expert_cache = ExpertCache(
            num_experts=num_experts,
            cache_size=layer_cache_size,
            weights_path=weights_path,
            layer_idx=layer_idx,
            enable_logging=enable_logging,
        )

        expert_caches.append(expert_cache)

        # Replace switch_mlp with cached wrapper
        wrapper = CachedSwitchMLPWrapper(
            layer.mlp.switch_mlp,
            expert_cache,
            layer_idx,
            prompt_handling=prompt_handling,
        )
        layer.mlp.switch_mlp = wrapper

    return expert_caches
