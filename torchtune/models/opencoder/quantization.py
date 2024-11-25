from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver


def create_dynamic_quant_config(
    dtype: str = "int8",
    symmetric: bool = False,
) -> QConfig:
    """Create a dynamic quantization config."""
    if dtype == "int8":
        weight_observer = MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine,
        )
    elif dtype == "int4":
        weight_observer = MinMaxObserver.with_args(
            dtype=torch.qint4,
            qscheme=torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine,
        )
    else:
        raise ValueError(f"Unsupported quantization dtype: {dtype}")
        
    return QConfig(
        activation=None,  # Dynamic quantization doesn't quantize activations
        weight=FakeQuantize.with_args(
            observer=weight_observer,
            dtype=torch.qint8,
        ),
    )


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer supporting different precisions and schemes.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: str = "int8",
        scheme: str = "symmetric",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create quantization parameters
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.weight_zero_point = nn.Parameter(torch.zeros(1))
        
        if scheme == "symmetric":
            self.weight_zero_point.requires_grad = False
        
        # Initialize quantized weights
        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
            
        self.dtype = dtype
        self.scheme = scheme

    @classmethod
    def from_float(cls, module: nn.Linear) -> "QuantizedLinear":
        """Convert a float Linear module to QuantizedLinear."""
        assert isinstance(module, nn.Linear)
        
        qmodule = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
        )
        
        # Compute quantization parameters
        weight_min = module.weight.min()
        weight_max = module.weight.max()
        
        if qmodule.scheme == "symmetric":
            weight_max = max(abs(weight_min), abs(weight_max))
            weight_min = -weight_max
            qmodule.weight_zero_point.data.zero_()
        else:
            qmodule.weight_zero_point.data.copy_(
                torch.tensor(-weight_min / (weight_max - weight_min) * 255)
            )
            
        qmodule.weight_scale.data.copy_(
            torch.tensor((weight_max - weight_min) / 255)
        )
        
        # Quantize weights
        qmodule.weight_packed.copy_(
            torch.clamp(
                torch.round(
                    module.weight / qmodule.weight_scale + qmodule.weight_zero_point
                ),
                -128,
                127,
            ).to(torch.int8)
        )
        
        if module.bias is not None:
            qmodule.bias.data.copy_(module.bias)
            
        return qmodule

    def forward(self, input: Tensor) -> Tensor:
        # Dequantize weights
        weight_float = (
            self.weight_packed.float() - self.weight_zero_point
        ) * self.weight_scale
        
        # Compute output
        output = torch.nn.functional.linear(input, weight_float, self.bias)
        return output


class Int8DynamicQuantizer:
    """
    Handles dynamic INT8 quantization of OpenCoder models.
    """
    def __init__(
        self,
        model: nn.Module,
        dtype: str = "int8",
        scheme: str = "symmetric",
    ):
        self.model = model
        self.dtype = dtype
        self.scheme = scheme
        
    def quantize(self) -> nn.Module:
        """
        Quantize the model using dynamic quantization.
        Only weights are quantized, activations remain in floating point.
        """
        def _quantize_module(module: nn.Module) -> nn.Module:
            if isinstance(module, nn.Linear):
                return QuantizedLinear.from_float(
                    module,
                    dtype=self.dtype,
                    scheme=self.scheme,
                )
            return module
            
        # Create a copy of the model to quantize
        model_to_quantize = type(self.model)(self.model.config)
        model_to_quantize.load_state_dict(self.model.state_dict())
        
        # Replace modules with quantized versions
        for name, module in model_to_quantize.named_children():
            if isinstance(module, nn.ModuleList):
                setattr(
                    model_to_quantize,
                    name,
                    nn.ModuleList([_quantize_module(m) for m in module])
                )
            else:
                setattr(model_to_quantize, name, _quantize_module(module))
                
        return model_to_quantize


def quantize_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
    scale_bits: int = 8,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """
    Quantize key-value cache to reduce memory usage.
    
    Args:
        key_states: Key states tensor
        value_states: Value states tensor
        scale_bits: Number of bits for scales (8 or 4)
        
    Returns:
        Tuple of (quantized_key_states, quantized_value_states, quantization_state)
    """
    # Compute scales and zero points
    k_max = torch.amax(torch.abs(key_states), dim=-1, keepdim=True)
    v_max = torch.amax(torch.abs(value_states), dim=-1, keepdim=True)
    
    k_scale = k_max / 127.0
    v_scale = v_max / 127.0
    
    # Quantize to int8
    k_quant = torch.clamp(
        torch.round(key_states / k_scale),
        -128,
        127,
    ).to(torch.int8)
    v_quant = torch.clamp(
        torch.round(value_states / v_scale),
        -128,
        127,
    ).to(torch.int8)
    
    # Save quantization state for dequantization
    quant_state = {
        "k_scale": k_scale,
        "v_scale": v_scale,
    }
    
    return k_quant, v_quant, quant_state


def dequantize_kv_cache(
    k_quant: Tensor,
    v_quant: Tensor,
    quant_state: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    Dequantize key-value cache.
    
    Args:
        k_quant: Quantized key states
        v_quant: Quantized value states
        quant_state: Quantization state from quantize_kv_cache
        
    Returns:
        Tuple of (key_states, value_states)
    """
    # Dequantize
    key_states = k_quant.float() * quant_state["k_scale"]
    value_states = v_quant.float() * quant_state["v_scale"]
    
    return key_states, value_states