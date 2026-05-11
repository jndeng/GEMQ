from typing import Any, Optional

import torch
from transformers.cache_utils import (
    StaticLayer, CacheLayerMixin, Cache, PretrainedConfig, StaticSlidingWindowLayer, 
    is_torchdynamo_compiling
)


class StaticLayer(CacheLayerMixin):
    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.max_batch_size, self.num_heads, _, self.key_head_dim = key_states.shape
        # NOTE: value_states have different head_dim than key_states in DeepseekV2
        _, _, _, self.value_head_dim = value_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.key_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.value_head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class StaticCache(Cache):
    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None)

        if layer_types is None:
            if getattr(config, "sliding_window", None) is not None:
                layer_types = ["sliding_attention" for _ in range(config.num_hidden_layers)]
            elif getattr(config, "attention_chunk_size", None) is not None:
                layer_types = ["chunked_attention" for _ in range(config.num_hidden_layers)]
            else:
                layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]

        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        layers = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                layer = StaticSlidingWindowLayer(max_cache_len=max_cache_len, sliding_window=config.sliding_window)
            elif layer_type == "chunked_attention":
                layer = StaticSlidingWindowLayer(
                    max_cache_len=max_cache_len, sliding_window=config.attention_chunk_size
                )
            else:
                layer = StaticLayer(max_cache_len=max_cache_len)
            layers.append(layer)

        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
