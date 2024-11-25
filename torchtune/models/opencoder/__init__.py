from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .tokenizer import OpenCoderTokenizer
from .loading import (
    load_checkpoint,
    save_checkpoint,
    load_config_from_json,
    save_config_to_json,
    convert_llama_weights,
)
from .generation import (
    OpenCoderGenerationConfig,
    prepare_inputs_for_generation,
    adjust_logits_during_generation,
)
from .flash_attention import FlashAttention
from .kv_cache import KeyValueCache, KeyValueCacheConfig
from .parallel import (
    initialize_model_parallel,
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelAttention,
    PipelineParallel,
)
from .quantization import (
    Int8DynamicQuantizer,
    quantize_kv_cache,
    dequantize_kv_cache,
)

from torchtune.modules.attention import Attention
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.position_embeddings import RotaryEmbedding
from torchtune.modules.rms_norm import RMSNorm
from torchtune.modules.transformer import TransformerBlock, Transformer
from torchtune.modules.tied_linear import TiedLinear


class OpenCoderConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_scaling: Optional[dict] = None,
        # Parallel processing
        use_flash_attention: bool = True,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        # Memory optimization
        use_kv_cache: bool = True,
        max_batch_size: int = 32,
        kv_cache_dtype: torch.dtype = torch.float16,
        # Quantization
        quantization: Optional[dict] = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_scaling = rope_scaling
        
        # Parallel processing
        self.use_flash_attention = use_flash_attention
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        
        # Memory optimization
        self.use_kv_cache = use_kv_cache
        self.max_batch_size = max_batch_size
        self.kv_cache_dtype = kv_cache_dtype
        
        # Quantization
        self.quantization = {
            "enabled": False,
            "dtype": "int8",
            "scheme": "symmetric",
        }
        if quantization is not None:
            self.quantization.update(quantization)


class OpenCoderBlock(TransformerBlock):
    def __init__(self, config: OpenCoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Choose attention implementation based on config
        if config.use_flash_attention:
            if config.tensor_parallel_size > 1:
                self.self_attn = ParallelAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    rotary_embedding=RotaryEmbedding(
                        head_size=config.hidden_size // config.num_attention_heads,
                        max_position_embeddings=config.max_position_embeddings,
                        scaling=config.rope_scaling,
                    ),
                )
            else:
                self.self_attn = FlashAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    rotary_embedding=RotaryEmbedding(
                        head_size=config.hidden_size // config.num_attention_heads,
                        max_position_embeddings=config.max_position_embeddings,
                        scaling=config.rope_scaling,
                    ),
                )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rotary_embedding=RotaryEmbedding(
                    head_size=config.hidden_size // config.num_attention_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    scaling=config.rope_scaling,
                ),
            )
        
        # Initialize feed-forward network with tensor parallelism if enabled
        if config.tensor_parallel_size > 1:
            self.mlp = nn.Sequential(
                ColumnParallelLinear(
                    config.hidden_size,
                    config.intermediate_size,
                    bias=False,
                ),
                nn.SiLU(),
                RowParallelLinear(
                    config.intermediate_size,
                    config.hidden_size,
                    bias=False,
                ),
            )
        else:
            self.mlp = FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize KV cache if enabled
        self.kv_cache = None
        if config.use_kv_cache:
            self.kv_cache = KeyValueCache(
                KeyValueCacheConfig(
                    max_batch_size=config.max_batch_size,
                    max_sequence_length=config.max_position_embeddings,
                    num_layers=1,  # Each block manages its own cache
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                    dtype=config.kv_cache_dtype,
                )
            )


class OpenCoderModel(Transformer):
    def __init__(self, config: OpenCoderConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Initialize model parallel if enabled
        if config.tensor_parallel_size > 1 or config.pipeline_parallel_size > 1:
            self.tensor_model_rank, self.pipeline_model_rank = initialize_model_parallel(
                config.tensor_parallel_size,
                config.pipeline_parallel_size,
            )
        else:
            self.tensor_model_rank = 0
            self.pipeline_model_rank = 0
        
        # Initialize embeddings with tensor parallelism if enabled
        if config.tensor_parallel_size > 1:
            self.embed_tokens = ColumnParallelLinear(
                config.vocab_size,
                config.hidden_size,
                bias=False,
                gather_output=True,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Initialize transformer layers with pipeline parallelism if enabled
        if config.pipeline_parallel_size > 1:
            self.layers = PipelineParallel(
                num_layers=config.num_hidden_layers,
                num_pipeline_parallel=config.pipeline_parallel_size,
                layer_fn=OpenCoderBlock,
                config=config,
            )
        else:
            self.layers = nn.ModuleList(
                [OpenCoderBlock(config) for _ in range(config.num_hidden_layers)]
            )
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize output head with tensor parallelism if enabled
        if config.tensor_parallel_size > 1:
            self.head = RowParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        else:
            self.head = TiedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                tie_embeddings=config.tie_word_embeddings,
                embeddings=self.embed_tokens,
            )
        
        # Apply quantization if enabled
        if config.quantization["enabled"]:
            self.quantizer = Int8DynamicQuantizer(
                self,
                dtype=config.quantization["dtype"],
                scheme=config.quantization["scheme"],
            )
            self = self.quantizer.quantize()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], ...]]]:
        # Prepare inputs
        if isinstance(self.embed_tokens, nn.Embedding):
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = self.embed_tokens(
                F.one_hot(input_ids, num_classes=self.vocab_size).float()
            )
        
        # Forward through transformer layers
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if isinstance(self.layers, PipelineParallel):
            hidden_states = self.layers(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            # Early return for non-last pipeline stages
            if hidden_states is None:
                return None
        else:
            hidden_states = self._forward_transformer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        
        # Handle past key values
        if isinstance(hidden_states, tuple):
            hidden_states, past_key_values = hidden_states
        
        # Final layer norm and head
        hidden_states = self.norm(hidden_states)
        logits = self.head(hidden_states)
        
        if use_cache:
            return logits, past_key_values
        return logits
        
    def activate_kv_cache(self):
        """Activate KV cache for all layers."""
        if not self.config.use_kv_cache:
            return
        if isinstance(self.layers, PipelineParallel):
            for layer in self.layers.layers:
                if hasattr(layer, "kv_cache"):
                    layer.kv_cache.activate()
        else:
            for layer in self.layers:
                if hasattr(layer, "kv_cache"):
                    layer.kv_cache.activate()
                    
    def deactivate_kv_cache(self):
        """Deactivate KV cache for all layers."""
        if not self.config.use_kv_cache:
            return
        if isinstance(self.layers, PipelineParallel):
            for layer in self.layers.layers:
                if hasattr(layer, "kv_cache"):
                    layer.kv_cache.deactivate()
        else:
            for layer in self.layers:
                if hasattr(layer, "kv_cache"):
                    layer.kv_cache.deactivate()