from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class KeyValueCacheConfig:
    max_batch_size: int = 32
    max_sequence_length: int = 4096
    num_layers: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: Optional[torch.device] = None


class KeyValueCache:
    """
    Manages key-value cache for efficient autoregressive generation.
    Supports dynamic batch size and sequence length management.
    """
    def __init__(self, config: KeyValueCacheConfig):
        self.config = config
        self.device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize cache tensors
        self.key_cache = torch.zeros(
            (
                config.num_layers,
                config.max_batch_size,
                config.max_sequence_length,
                config.num_kv_heads,
                config.head_dim,
            ),
            dtype=config.dtype,
            device=self.device,
        )
        self.value_cache = torch.zeros_like(self.key_cache)
        
        # Track current state
        self.current_seq_len = 0
        self.batch_indices: Dict[int, List[int]] = {}  # Maps batch idx to sequence positions
        self._active = False

    def activate(self):
        """Activate the cache for use."""
        self._active = True
        self.current_seq_len = 0
        self.batch_indices.clear()
        return self

    def deactivate(self):
        """Deactivate the cache."""
        self._active = False
        return self

    def update(
        self,
        layer_idx: int,
        key_states: Tensor,
        value_states: Tensor,
        batch_positions: Optional[List[int]] = None,
    ) -> None:
        """
        Update the cache with new key-value states.
        
        Args:
            layer_idx: Index of the transformer layer
            key_states: New key states (batch_size, seq_len, num_kv_heads, head_dim)
            value_states: New value states (batch_size, seq_len, num_kv_heads, head_dim)
            batch_positions: Optional list of positions to update for each batch item
        """
        if not self._active:
            return

        batch_size, seq_len, num_heads, head_dim = key_states.shape
        
        if batch_positions is None:
            # Sequential update
            start_pos = self.current_seq_len
            end_pos = start_pos + seq_len
            
            if end_pos > self.config.max_sequence_length:
                raise ValueError(
                    f"Sequence length {end_pos} exceeds maximum {self.config.max_sequence_length}"
                )
            
            self.key_cache[layer_idx, :batch_size, start_pos:end_pos] = key_states
            self.value_cache[layer_idx, :batch_size, start_pos:end_pos] = value_states
            
            # Update tracking
            self.current_seq_len = end_pos
            for batch_idx in range(batch_size):
                if batch_idx not in self.batch_indices:
                    self.batch_indices[batch_idx] = []
                self.batch_indices[batch_idx].extend(range(start_pos, end_pos))
        else:
            # Indexed update
            for batch_idx, positions in enumerate(batch_positions):
                if isinstance(positions, int):
                    positions = [positions]
                for pos_idx, pos in enumerate(positions):
                    if pos >= self.config.max_sequence_length:
                        raise ValueError(
                            f"Position {pos} exceeds maximum sequence length"
                        )
                    self.key_cache[layer_idx, batch_idx, pos] = key_states[batch_idx, pos_idx]
                    self.value_cache[layer_idx, batch_idx, pos] = value_states[batch_idx, pos_idx]
                    
                    if batch_idx not in self.batch_indices:
                        self.batch_indices[batch_idx] = []
                    self.batch_indices[batch_idx].append(pos)

    def get(
        self,
        layer_idx: int,
        batch_size: Optional[int] = None,
        start_pos: Optional[int] = None,
        end_pos: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Retrieve cached key-value states.
        
        Args:
            layer_idx: Index of the transformer layer
            batch_size: Optional batch size to retrieve
            start_pos: Optional start position in sequence
            end_pos: Optional end position in sequence
            
        Returns:
            Tuple of (key_states, value_states)
        """
        if not self._active:
            raise RuntimeError("Cache is not active")

        if batch_size is None:
            batch_size = max(self.batch_indices.keys()) + 1
        if start_pos is None:
            start_pos = 0
        if end_pos is None:
            end_pos = self.current_seq_len

        key_states = self.key_cache[
            layer_idx, :batch_size, start_pos:end_pos
        ]
        value_states = self.value_cache[
            layer_idx, :batch_size, start_pos:end_pos
        ]
        
        return key_states, value_states

    def resize_cache(self, max_batch_size: Optional[int] = None, max_seq_len: Optional[int] = None):
        """Resize the cache tensors."""
        new_batch_size = max_batch_size or self.config.max_batch_size
        new_seq_len = max_seq_len or self.config.max_sequence_length
        
        if new_batch_size == self.config.max_batch_size and new_seq_len == self.config.max_sequence_length:
            return
        
        new_key_cache = torch.zeros(
            (
                self.config.num_layers,
                new_batch_size,
                new_seq_len,
                self.config.num_kv_heads,
                self.config.head_dim,
            ),
            dtype=self.config.dtype,
            device=self.device,
        )
        new_value_cache = torch.zeros_like(new_key_cache)
        
        # Copy existing cache
        copy_batch_size = min(new_batch_size, self.config.max_batch_size)
        copy_seq_len = min(new_seq_len, self.config.max_sequence_length)
        
        new_key_cache[:, :copy_batch_size, :copy_seq_len] = self.key_cache[
            :, :copy_batch_size, :copy_seq_len
        ]
        new_value_cache[:, :copy_batch_size, :copy_seq_len] = self.value_cache[
            :, :copy_batch_size, :copy_seq_len
        ]
        
        self.key_cache = new_key_cache
        self.value_cache = new_value_cache
        self.config.max_batch_size = new_batch_size
        self.config.max_sequence_length = new_seq_len

    def clear(self):
        """Clear the cache."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.current_seq_len = 0
        self.batch_indices.clear()
        return self