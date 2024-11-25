from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


def _flash_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tensor:
    """
    Implements Flash Attention forward pass.
    
    Args:
        query: (batch_size, seqlen, num_heads, head_dim)
        key: (batch_size, seqlen, num_kv_heads, head_dim)
        value: (batch_size, seqlen, num_kv_heads, head_dim)
        attention_mask: Optional attention mask (batch_size, seqlen)
        dropout_p: Dropout probability
        softmax_scale: Optional scaling factor for attention scores
        causal: Whether to apply causal mask
    
    Returns:
        output: (batch_size, seqlen, num_heads, head_dim)
    """
    if not FLASH_AVAILABLE:
        raise ImportError("flash-attention package is required but not installed.")
    
    batch_size, seqlen, num_heads, head_dim = query.shape
    
    # Handle padding if attention mask is provided
    if attention_mask is not None:
        query, indices, cu_seqlens, max_seqlen = unpad_input(
            query, attention_mask.squeeze(1)
        )
        key, _, _, _ = unpad_input(key, attention_mask.squeeze(1))
        value, _, _, _ = unpad_input(value, attention_mask.squeeze(1))
        
        output_unpad = flash_attn_func(
            query, key, value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        
        output = pad_input(
            output_unpad, indices, batch_size, seqlen
        )
    else:
        # Reshape inputs for Flash Attention
        query = query.transpose(1, 2)  # (batch_size, num_heads, seqlen, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        output = flash_attn_func(
            query, key, value,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        
        output = output.transpose(1, 2)  # (batch_size, seqlen, num_heads, head_dim)
    
    return output


class FlashAttention(nn.Module):
    """
    Optimized attention module using Flash Attention when available.
    Falls back to standard attention when Flash Attention is not available.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        qkv_proj_bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.softmax_scale = softmax_scale or 1.0 / (self.head_dim ** 0.5)
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_proj_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_proj_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_proj_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=qkv_proj_bias)
        
        self._flash_supported = FLASH_AVAILABLE

    def _repeat_kv_heads(self, hidden_states: Tensor) -> Tensor:
        """Repeat k/v heads if num_kv_heads < num_heads."""
        if self.num_kv_heads != self.num_heads:
            hidden_states = hidden_states.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=2
            )
        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seqlen, _ = hidden_states.shape
        
        # Project query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(
            batch_size, seqlen, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, seqlen, self.num_kv_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, seqlen, self.num_kv_heads, self.head_dim
        )
        
        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat KV heads if needed
        key_states = self._repeat_kv_heads(key_states)
        value_states = self._repeat_kv_heads(value_states)
        
        if self._flash_supported:
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=True,
            )
        else:
            # Standard attention as fallback
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
            attn_weights = attn_weights * self.softmax_scale
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2)
        
        attn_output = attn_output.reshape(batch_size, seqlen, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value