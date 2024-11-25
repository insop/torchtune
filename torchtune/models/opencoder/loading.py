import json
import os
from typing import Dict, Optional, Union

import torch
from torch import nn

from . import OpenCoderConfig, OpenCoderModel


def load_config_from_json(config_file: str) -> OpenCoderConfig:
    """Load OpenCoder config from a JSON file."""
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    return OpenCoderConfig(**config_dict)


def save_config_to_json(config: OpenCoderConfig, config_file: str) -> None:
    """Save OpenCoder config to a JSON file."""
    config_dict = config.__dict__.copy()
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_checkpoint(
    checkpoint_path: str,
    config: Optional[Union[OpenCoderConfig, str]] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> OpenCoderModel:
    """
    Load an OpenCoder model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: OpenCoderConfig instance or path to config.json
        device: Device to load the model on
        dtype: Data type for model parameters
    
    Returns:
        OpenCoderModel instance
    """
    if config is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if not os.path.exists(config_path):
            raise ValueError(
                f"No config provided and no config.json found at {config_path}"
            )
        config = config_path

    if isinstance(config, str):
        config = load_config_from_json(config)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    model = OpenCoderModel(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def save_checkpoint(
    model: OpenCoderModel,
    save_dir: str,
    filename: str = "pytorch_model.bin",
    save_config: bool = True,
) -> None:
    """
    Save an OpenCoder model checkpoint.
    
    Args:
        model: OpenCoderModel instance to save
        save_dir: Directory to save the checkpoint in
        filename: Name of the checkpoint file
        save_config: Whether to save the config alongside the model
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    
    # Save model weights
    model_to_save = model.module if hasattr(model, "module") else model
    state_dict = model_to_save.state_dict()
    torch.save(state_dict, checkpoint_path)
    
    # Save config
    if save_config:
        config_path = os.path.join(save_dir, "config.json")
        save_config_to_json(model.config, config_path)


def convert_llama_weights(
    llama_model: nn.Module,
    config: OpenCoderConfig,
) -> Dict[str, torch.Tensor]:
    """
    Convert LLaMA weights to OpenCoder format.
    
    Args:
        llama_model: LLaMA model instance
        config: OpenCoder config for the target model
        
    Returns:
        Dictionary with converted weights
    """
    state_dict = {}
    llama_state = llama_model.state_dict()
    
    # Direct mappings
    mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
    }
    
    # Copy directly mapped weights
    for llama_key, opencoder_key in mapping.items():
        if llama_key in llama_state:
            state_dict[opencoder_key] = llama_state[llama_key]
    
    # Convert layer weights
    for i in range(config.num_hidden_layers):
        # Attention weights
        state_dict[f"layers.{i}.self_attn.q_proj.weight"] = llama_state[f"model.layers.{i}.self_attn.q_proj.weight"]
        state_dict[f"layers.{i}.self_attn.k_proj.weight"] = llama_state[f"model.layers.{i}.self_attn.k_proj.weight"]
        state_dict[f"layers.{i}.self_attn.v_proj.weight"] = llama_state[f"model.layers.{i}.self_attn.v_proj.weight"]
        state_dict[f"layers.{i}.self_attn.o_proj.weight"] = llama_state[f"model.layers.{i}.self_attn.o_proj.weight"]
        
        # Layer norms
        state_dict[f"layers.{i}.input_layernorm.weight"] = llama_state[f"model.layers.{i}.input_layernorm.weight"]
        state_dict[f"layers.{i}.post_attention_layernorm.weight"] = llama_state[f"model.layers.{i}.post_attention_layernorm.weight"]
        
        # Feed-forward weights
        state_dict[f"layers.{i}.mlp.gate_proj.weight"] = llama_state[f"model.layers.{i}.mlp.gate_proj.weight"]
        state_dict[f"layers.{i}.mlp.up_proj.weight"] = llama_state[f"model.layers.{i}.mlp.up_proj.weight"]
        state_dict[f"layers.{i}.mlp.down_proj.weight"] = llama_state[f"model.layers.{i}.mlp.down_proj.weight"]
    
    return state_dict