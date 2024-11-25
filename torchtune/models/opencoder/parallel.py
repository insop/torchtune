from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from .flash_attention import FlashAttention


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> Tuple[int, int]:
    """
    Initialize model parallel groups.
    
    Args:
        tensor_model_parallel_size: Number of tensor parallel processes
        pipeline_model_parallel_size: Number of pipeline parallel processes
        
    Returns:
        Tuple of (tensor_model_parallel_rank, pipeline_model_parallel_rank)
    """
    if not dist.is_initialized():
        return 0, 0

    world_size = dist.get_world_size()
    if world_size < tensor_model_parallel_size * pipeline_model_parallel_size:
        raise ValueError(
            f"World size ({world_size}) must be >= tensor_parallel_size ({tensor_model_parallel_size}) "
            f"* pipeline_parallel_size ({pipeline_model_parallel_size})"
        )

    rank = dist.get_rank()
    
    # Create tensor model parallel group
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    tensor_model_parallel_group = None
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            tensor_model_parallel_group = group
            tensor_model_parallel_rank = rank % tensor_model_parallel_size

    # Create pipeline model parallel group
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    pipeline_model_parallel_group = None
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = dist.new_group(ranks)
        if rank in ranks:
            pipeline_model_parallel_group = group
            pipeline_model_parallel_rank = rank // num_pipeline_model_parallel_groups

    return tensor_model_parallel_rank, pipeline_model_parallel_rank


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    The linear layer is divided uniformly along the output dimension.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[callable] = None,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        super().__init__()
        
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        # Divide the weight matrix along the output dimension
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.output_size_per_partition = output_size // world_size
        
        # Parameters
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size,
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        if init_method is not None:
            init_method(self.weight)
            if self.bias is not None:
                init_method(self.bias)
                
        # For testing
        if keep_master_weight_for_test:
            self.master_weight = self.weight.data.clone()
            if self.bias is not None:
                self.master_bias = self.bias.data.clone()

    def forward(self, input_: Tensor) -> Tensor:
        # Matrix multiply
        output_parallel = F.linear(input_, self.weight, self.bias)
        
        if self.gather_output and dist.is_initialized():
            # All-gather across model parallel group
            world_size = dist.get_world_size()
            output_list = [torch.empty_like(output_parallel) for _ in range(world_size)]
            dist.all_gather(output_list, output_parallel)
            output = torch.cat(output_list, dim=-1)
        else:
            output = output_parallel
        
        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    The linear layer is divided uniformly along the input dimension.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[callable] = None,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        super().__init__()
        
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        
        # Divide the weight matrix along the input dimension
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.input_size_per_partition = input_size // world_size
        
        # Parameters
        self.weight = nn.Parameter(torch.empty(
            self.output_size,
            self.input_size_per_partition,
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        if init_method is not None:
            init_method(self.weight)
            if self.bias is not None:
                init_method(self.bias)
                
        # For testing
        if keep_master_weight_for_test:
            self.master_weight = self.weight.data.clone()
            if self.bias is not None:
                self.master_bias = self.bias.data.clone()

    def forward(self, input_: Tensor) -> Tensor:
        # Set up input tensor
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # Split input tensor along last dimension
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            input_parallel = input_.chunk(world_size, dim=-1)[dist.get_rank()]
            
        # Matrix multiply
        output_parallel = F.linear(input_parallel, self.weight)
        
        if dist.is_initialized():
            # All-reduce across model parallel group
            dist.all_reduce(output_parallel)
            
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        return output_parallel


class ParallelAttention(FlashAttention):
    """
    Attention module with tensor parallelism support.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        qkv_proj_bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        gather_output: bool = True,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_proj_bias=qkv_proj_bias,
            dropout=dropout,
            softmax_scale=softmax_scale,
        )
        
        # Replace linear projections with parallel versions
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * self.head_dim,
            bias=qkv_proj_bias,
            gather_output=False,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=qkv_proj_bias,
            gather_output=False,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=qkv_proj_bias,
            gather_output=False,
        )
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=qkv_proj_bias,
            input_is_parallel=True,
        )


class PipelineParallel(nn.Module):
    """
    Implements pipeline parallelism for transformer models.
    """
    def __init__(
        self,
        num_layers: int,
        num_pipeline_parallel: int,
        layer_fn: callable,
        *layer_args,
        **layer_kwargs,
    ):
        super().__init__()
        if not dist.is_initialized():
            self.layers = nn.ModuleList([
                layer_fn(*layer_args, **layer_kwargs)
                for _ in range(num_layers)
            ])
            return
            
        # Divide layers among pipeline stages
        self.pipeline_parallel_rank = dist.get_rank() // (
            dist.get_world_size() // num_pipeline_parallel
        )
        self.pipeline_parallel_size = num_pipeline_parallel
        
        layers_per_stage = num_layers // num_pipeline_parallel
        start_layer = self.pipeline_parallel_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage
        
        self.layers = nn.ModuleList([
            layer_fn(*layer_args, **layer_kwargs)
            for _ in range(start_layer, end_layer)
        ])
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tuple[Tensor, Tensor]]]]:
        # Handle non-distributed case
        if not dist.is_initialized():
            return self._forward_sequential(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
            )
            
        # Pipeline parallel forward
        if self.pipeline_parallel_rank == 0:
            # First stage
            output = self._forward_sequential(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
            )
            self._send_activations(output)
            return None
            
        elif self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            # Last stage
            hidden_states = self._receive_activations()
            return self._forward_sequential(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
            )
            
        else:
            # Middle stages
            hidden_states = self._receive_activations()
            output = self._forward_sequential(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
            )
            self._send_activations(output)
            return None
            
    def _forward_sequential(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tuple[Tensor, Tensor]]]]:
        """Sequential forward pass through layers."""
        all_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            if use_cache:
                hidden_states, past_key_value = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
                all_past_key_values.append(past_key_value)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
                
        if use_cache:
            return hidden_states, all_past_key_values
        return hidden_states
        
    def _send_activations(self, activations: Tensor):
        """Send activations to next pipeline stage."""
        next_rank = dist.get_rank() + 1
        dist.send(activations, next_rank)
        
    def _receive_activations(self) -> Tensor:
        """Receive activations from previous pipeline stage."""
        prev_rank = dist.get_rank() - 1
        activations = torch.empty_like(hidden_states)
        dist.recv(activations, prev_rank)
        return activations