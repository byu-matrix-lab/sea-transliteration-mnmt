"""PyTorch BART model."""

import math
import warnings
from typing import List, Callable, Optional, Union, Tuple, Any
import copy

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer # EncoderDecoderCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    auto_docstring,
    is_torch_flex_attn_available,
    is_torchdynamo_compiling,
    logging,
)
from transformers import BartConfig # from .configuration_bart import BartConfig


if is_torch_flex_attn_available():
    from transformers.integrations.flex_attention import BlockMask, make_flex_block_causal_mask

@dataclass
class BaseMultimodalModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    phonetic_last_hidden_state: torch.FloatTensor = None
    phonetic_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    phonetic_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class Seq2SeqMultimodalModelOutput(Seq2SeqModelOutput):
    """
    Base class for sequence-to-sequence multimodal model outputs.
    Inherits from `Seq2SeqModelOutput` and adds phonetic-specific attributes.

    Args:
        last_hidden_state (`torch.FloatTensor`, *optional*, returned when `config.output_hidden_states=True`):
            Sequence of hidden-states at the output of the last layer of the decoder of shape `(batch_size, sequence_length, hidden_size)`.
        past_key_values (`Tuple(torch.FloatTensor)`, *optional*, returned when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` of length `config.num_hidden_layers`, with each tuple having two tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)` for keys and values.
        decoder_hidden_states (`Tuple(torch.FloatTensor)`, *optional*, returned when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)` containing the decoder's hidden states at each layer.
        decoder_attentions (`Tuple(torch.FloatTensor)`, *optional*, returned when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)` containing the decoder's attention weights at each layer.
        cross_attentions (`Tuple(torch.FloatTensor)`, *optional*, returned when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)` containing the cross-attention weights at each layer.
        encoder_last_hidden_state (`torch.FloatTensor`, *optional*, returned when `config.output_hidden_states=True`):
            Sequence of hidden-states at the output of the last layer of the encoder of shape `(batch_size, sequence_length, hidden_size)`.
        encoder_hidden_states (`Tuple(torch.FloatTensor)`, *optional*, returned when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)` containing the encoder's hidden states at each layer.
        encoder_attentions (`Tuple(torch.FloatTensor)`, *optional*, returned when `config.output_attentions=True`): 
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)` containing the encoder's attention weights at each layer.
        encoder_phonetic_last_hidden_state (`torch.FloatTensor`, *optional*, returned when `config.output_hidden_states=True`): 
            Sequence of hidden-states at the output of the last layer of the phonetic encoder of shape `(batch_size, sequence_length, hidden_size)`.
        encoder_phonetic_hidden_states (`Tuple(torch.FloatTensor)`, *optional*, returned when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)` containing the phonetic encoder's hidden states at each layer.
        encoder_phonetic_attentions (`Tuple(torch.FloatTensor)`, *optional*, returned when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)` containing the phonetic encoder's attention weights at each layer.
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    phonetic_last_hidden_state: torch.FloatTensor = None
    phonetic_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    phonetic_attentions: Optional[Tuple[torch.FloatTensor]] = None


logger = logging.get_logger(__name__)

# class CustomGenerationMixin(GenerationMixin):
#     """
#     Custom generation mixin for BART models that supports phonetic cross-attention.
#     This class extends the `GenerationMixin` to handle phonetic cross-attention during generation.
#     """

#     def _expand_inputs_for_generation(
#         expand_size: int = 1,
#         is_encoder_decoder: bool = False,
#         input_ids: Optional[torch.LongTensor] = None,
#         **model_kwargs,
#     ) -> tuple[torch.LongTensor, dict[str, Any]]:
#         """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
#         # Do not call torch.repeat_interleave if expand_size is 1 because it clones
#         # the input tensor and thus requires more memory although no change is applied
#         if expand_size == 1:
#             return input_ids, model_kwargs

#         print("HELLO")
#         exit()

#         def _expand_dict_for_generation(dict_to_expand):
#             for key in dict_to_expand:
#                 if (
#                     key != "cache_position"
#                     and dict_to_expand[key] is not None
#                     and isinstance(dict_to_expand[key], torch.Tensor)
#                 ):
#                     dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
#             return dict_to_expand

#         if input_ids is not None:
#             input_ids = input_ids.repeat_interleave(expand_size, dim=0)
    
#         model_kwargs = _expand_dict_for_generation(model_kwargs)

#         if is_encoder_decoder:
#             if model_kwargs.get("encoder_outputs") is None:
#                 raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
#             model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

#         return input_ids, model_kwargs


class EncoderDecoderCache(DynamicCache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
        >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

        >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

        >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
        >>> self_attention_cache = DynamicCache()
        >>> cross_attention_cache = DynamicCache()
        >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        EncoderDecoderCache()
        ```

    """

    # Override @property from Cache
    is_compileable = None

    def __init__(self, self_attention_cache: Optional[Cache] = None, cross_attention_cache: Optional[Cache] = None, phonetic_cross_attention_cache: Optional[Cache] = None):
        super(DynamicCache, self).__init__(layer_classes=DynamicLayer)
        self.self_attention_cache = self_attention_cache if self_attention_cache is not None else DynamicCache()
        self.cross_attention_cache = cross_attention_cache if cross_attention_cache is not None else DynamicCache()
        self.phonetic_cross_attention_cache = phonetic_cross_attention_cache if phonetic_cross_attention_cache is not None else DynamicCache()
        self.is_compileable = getattr(self.self_attention_cache, "is_compileable", False)

        self.is_updated = {}
        self.phonetic_is_updated = {}
        for layer_idx in range(len(self.cross_attention_cache)):
            self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)
        for layer_idx in range(len(self.phonetic_cross_attention_cache)):
            self.phonetic_is_updated[layer_idx] = bool(self.phonetic_cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (
                self.self_attention_cache.layers[layer_idx].keys,
                self.self_attention_cache.layers[layer_idx].values,
                self.cross_attention_cache.layers[layer_idx].keys,
                self.cross_attention_cache.layers[layer_idx].values,
                self.phonetic_cross_attention_cache.layers[layer_idx].keys,
                self.phonetic_cross_attention_cache.layers[layer_idx].values,
            )

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.self_attention_cache.layers[layer_idx].keys,
                self.self_attention_cache.layers[layer_idx].values,
                self.cross_attention_cache.layers[layer_idx].keys,
                self.cross_attention_cache.layers[layer_idx].values,
                self.phonetic_cross_attention_cache.layers[layer_idx].keys,
                self.phonetic_cross_attention_cache.layers[layer_idx].values,
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor]]:
        """Converts the `EncoderDecoderCache` instance into its equivalent in the legacy cache format."""
        legacy_cache = ()
        if len(self.cross_attention_cache) > 0:
            for self_attn, cross_attn in zip(
                self.self_attention_cache.to_legacy_cache(), self.cross_attention_cache.to_legacy_cache()
            ):
                legacy_cache += (self_attn + cross_attn,)
        else:
            legacy_cache = self.self_attention_cache.to_legacy_cache()
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...]
    ) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position=None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not torch.Tensor`
        return self.self_attention_cache.get_seq_length(layer_idx, cache_position)

    def reset(self):
        if hasattr(self.self_attention_cache, "reset"):
            self.self_attention_cache.reset()
        if hasattr(self.cross_attention_cache, "reset"):
            self.cross_attention_cache.reset()
        if hasattr(self.phonetic_cross_attention_cache, "reset"):
            self.phonetic_cross_attention_cache.reset()
        elif not hasattr(self.self_attention_cache, "reset") and not hasattr(self.cross_attention_cache, "reset"):
            raise ValueError(
                "Neither self nor cross-attention cache have valid `.reset()` methods. `.reset()` should "
                "only be called on compatible cache classes, such as `StaticCache` or `SlidingWindowCache`. "
                f"Got {self.self_attention_cache.__str__()} for the self attention cache and "
                f"{self.cross_attention_cache.__str__()} for the cross attention cache."
            )
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False
        for layer_idx in self.phonetic_is_updated:
            self.phonetic_is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)
        self.phonetic_cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache)
            and isinstance(self.cross_attention_cache, DynamicCache)
            and isinstance(self.phonetic_cross_attention_cache, DynamicCache)
        ):
            raise ValueError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """
        Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search.
        """
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "list[EncoderDecoderCache]":
        """
        Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`
        """
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)
        phonetic_cross_attention_cache = self.phonetic_cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn, phonetic_attn in zip(self_attention_cache, cross_attention_cache, phonetic_cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn, phonetic_cross_attention_cache=phonetic_attn))
        return out

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)
        self.phonetic_cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)
        self.phonetic_cross_attention_cache.batch_select_indices(indices)

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.self_attention_cache.get_mask_sizes(cache_position, layer_idx)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0, position_ids: torch.Tensor = None):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        if position_ids is None:
            bsz, seq_len = input_ids.shape[:2]
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
            ).expand(bsz, -1)
        else:
            position_ids = position_ids.unsqueeze(0)

        return super().forward(position_ids + self.offset)


class BartScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BartConfig] = None,
        layer_idx: Optional[int] = None,
        is_phonetic: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.is_phonetic = is_phonetic

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]

        # print(hidden_states.shape)
        # exit()
        src_len = key_value_states.shape[1] if is_cross_attention else tgt_len

        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        # get query proj
        query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)

        # print(f"Past key value: {past_key_value}, is_cross_attention: {is_cross_attention}, layer_idx: {self.layer_idx}, cache_position: {cache_position}")
        

        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                if self.is_phonetic:
                    updated_cache = past_key_value.phonetic_is_updated
                else:
                    updated_cache = past_key_value.is_updated
                is_updated = updated_cache.get(self.layer_idx)
                
                if is_cross_attention:
                    if self.is_phonetic:
                        # phonetic cross-attention cache
                        # print("HERE")
                        curr_past_key_value = past_key_value.phonetic_cross_attention_cache
                    else:
                        # after the first generated id, we can subsequently re-use all key/value_states from cache
                        curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.layers[self.layer_idx].keys
            value_states = curr_past_key_value.layers[self.layer_idx].values

            # Debug
            # print("key_states shape: {}, value_states shape: {}".format(key_states.shape, value_states.shape), flush=True)
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            cur_bsz, seq_len, hidden_dim = key_states.shape
            # Debug
            # print(f"1key_states shape: {key_states.shape}, value_states shape: {value_states.shape}", flush=True)
            # print(f"1kv_input_shape: {bsz, seq_len, self.num_heads, self.head_dim}", flush=True)
            if cur_bsz != bsz:
                key_states = key_states.repeat_interleave(bsz // cur_bsz, dim=0)
                value_states = value_states.repeat_interleave(bsz // cur_bsz, dim=0)
            key_states = key_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    updated_cache[self.layer_idx] = True

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # print(f"key_states shape: {key_states.shape}, value_states shape: {value_states.shape}", flush=True)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class BartEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BartConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BartConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.phonetic_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            layer_idx=layer_idx,
            is_phonetic=True,  # This indicates that this is a phonetic cross-attention layer
        )
        self.phonetic_attn_layer_norm = nn.LayerNorm(self.embed_dim)


        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        phonetic_hidden_states: Optional[torch.Tensor] = None,
        phonetic_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
                cache in the correct position and to infer the complete sequence length.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Phonetic Cross-Attention Block
        # print("Phonetic hidden states 1: ", phonetic_hidden_states, flush=True)
        
        if phonetic_hidden_states is not None:
            residual = hidden_states
            
            hidden_states, phonetic_attn_weights = self.phonetic_attn(
                hidden_states=hidden_states,
                key_value_states=phonetic_hidden_states,
                attention_mask=phonetic_attention_mask,
                layer_head_mask=None, # TODO: Add this later maybe
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.phonetic_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)


        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


@auto_docstring
class BartPreTrainedModel(PreTrainedModel):
    config: BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"BartEncoderLayer", r"BartDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask

    def _update_causal_mask(
        self,
        attention_mask: Optional[Union[torch.Tensor, "BlockMask"]],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            # Other attention flavors support in-built causal (when `mask is None`)
            # while we need to create our specific block mask regardless
            elif attention_mask is None:
                attention_mask = make_flex_block_causal_mask(
                    torch.ones(
                        size=(input_tensor.shape[0], input_tensor.shape[1]),
                        device=attention_mask.device,
                    )
                )
            return attention_mask

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def _update_cross_attn_mask(
        self,
        encoder_hidden_states: Union[torch.Tensor, None],
        encoder_attention_mask: Union[torch.Tensor, None],
        input_shape: torch.Size,
        inputs_embeds: torch.Tensor,
    ):
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(encoder_attention_mask, torch.Tensor):
                    encoder_attention_mask = make_flex_block_causal_mask(
                        encoder_attention_mask,
                        query_length=input_shape[-1],
                        is_causal=False,
                    )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        return encoder_attention_mask


class PretrainedBartModel(BartPreTrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


class BartPretrainedModel(BartPreTrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )

class BartDualEncoder(BartPreTrainedModel):

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size

        self.encoder = BartEncoder(config, embed_tokens)

        # Make a second encoder, using the config but modifed by parameters starting with phonetic_
        if config.phonetic_encoder:
            phonetic_config = copy.deepcopy(config)
            overwrite = [prop for prop in dir(phonetic_config) if prop.startswith('phonetic_')]
            for prop in overwrite:
                setattr(phonetic_config, prop[9:], getattr(phonetic_config, prop))
                delattr(phonetic_config, prop)
            self.phonetic_encoder = BartEncoder(phonetic_config, embed_tokens)
        else:
            raise ValueError(
                "The BartDualEncoder model requires a phonetic encoder to be defined in the configuration."
            )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        phonetic_input_ids: Optional[torch.FloatTensor] = None,
        phonetic_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if return_dict:
                encoder_outputs = BaseMultimodalModelOutput()

            if input_ids is not None or inputs_embeds is not None:
                text_encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                if return_dict:
                    encoder_outputs.last_hidden_state=text_encoder_outputs.last_hidden_state
                    encoder_outputs.hidden_states=text_encoder_outputs.hidden_states
                    encoder_outputs.attentions=text_encoder_outputs.attentions
                else:
                    encoder_outputs = text_encoder_outputs

            if phonetic_input_ids is not None:
                phonetic_encoder_outputs = self.phonetic_encoder(
                    input_ids=phonetic_input_ids,
                    attention_mask=phonetic_attention_mask,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                
                if encoder_outputs is None:
                    encoder_outputs = phonetic_encoder_outputs
                elif return_dict:
                    encoder_outputs.phonetic_last_hidden_state=phonetic_encoder_outputs.last_hidden_state
                    encoder_outputs.phonetic_hidden_states=phonetic_encoder_outputs.hidden_states
                    encoder_outputs.phonetic_attentions=phonetic_encoder_outputs.attentions
                else:
                    encoder_outputs += phonetic_encoder_outputs

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseMultimodalModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseMultimodalModelOutput):
            encoder_outputs = BaseMultimodalModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                phonetic_last_hidden_state=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                phonetic_hidden_states=encoder_outputs[4] if len(encoder_outputs) > 4 else None,
                phonetic_attentions=encoder_outputs[5] if len(encoder_outputs) > 5 else None,
            )

        return encoder_outputs
    
    


class BartEncoder(BartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        attention_mask = self._update_full_mask(
            attention_mask,
            inputs_embeds,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)])

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        phonetic_hidden_states: Optional[torch.FloatTensor] = None,
        phonetic_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
                cache in the correct position and to infer the complete sequence length.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input)

        # initialize `past_key_values`
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        batch_size, seq_length = inputs_embeds.size()[:-1]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        self_attn_cache = (
            past_key_values.self_attention_cache
            if isinstance(past_key_values, EncoderDecoderCache)
            else past_key_values
        )

        attention_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            self_attn_cache,
        )
        encoder_attention_mask = self._update_cross_attn_mask(
            encoder_hidden_states,
            encoder_attention_mask,
            input_shape,
            inputs_embeds,
        )
        phonetic_attention_mask = self._update_cross_attn_mask(
            phonetic_hidden_states,
            phonetic_attention_mask,
            input_shape,
            inputs_embeds,
        )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length, position_ids=cache_position)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                phonetic_hidden_states=phonetic_hidden_states,
                phonetic_attention_mask=phonetic_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@auto_docstring
class BartModel(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = BartScaledWordEmbedding(vocab_size, config.d_model, padding_idx, embed_scale=embed_scale)

        self.encoder = BartDualEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            # Some model checkpoints like "facebook/bart-large-cnn"'s embedding weight is in decoder.embed_tokens, need check here, see issue #36247
            if self.shared.weight.device == torch.device(
                "meta"
            ) and self.decoder.embed_tokens.weight.device != torch.device("meta"):
                self._tie_or_clone_weights(self.encoder.encoder.embed_tokens, self.decoder.embed_tokens)
                self._tie_or_clone_weights(self.shared, self.decoder.embed_tokens)
            else:
                self._tie_or_clone_weights(self.encoder.encoder.embed_tokens, self.shared)
                self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        phonetic_input_ids: Optional[torch.LongTensor] = None,
        phonetic_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, Seq2SeqMultimodalModelOutput]:
        r"""
        phonetic_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of phonetic input sequence tokens in the vocabulary.
        phonetic_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices of phonetic input_ids.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
            information on the default strategy.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        """
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                phonetic_input_ids=phonetic_input_ids,
                phonetic_attention_mask=phonetic_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseMultimodalModelOutput):
            encoder_outputs = BaseMultimodalModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                phonetic_last_hidden_state=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                phonetic_hidden_states=encoder_outputs[4] if len(encoder_outputs) > 4 else None,
                phonetic_attentions=encoder_outputs[5] if len(encoder_outputs) > 5 else None,
            )

        # print("input_ids2:", input_ids2)
        # print("attention_mask2:", attention_mask2)

        # print("Encoder Outputs: ", encoder_outputs)
        # print("Encoder Outputs Phonetic Last HIdden: ", encoder_outputs.phonetic_last_hidden_state)
        # exit()

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            phonetic_hidden_states=encoder_outputs.phonetic_last_hidden_state,
            phonetic_attention_mask=phonetic_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqMultimodalModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            phonetic_last_hidden_state=encoder_outputs.phonetic_last_hidden_state,
            phonetic_hidden_states=encoder_outputs.phonetic_hidden_states,
            phonetic_attentions=encoder_outputs.phonetic_attentions,
        )


@auto_docstring(
    custom_intro="""
    The BART Model with a language modeling head. Can be used for summarization.
    """
)
class BartForConditionalGeneration(BartPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None, mean_resizing: bool = True
    ) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self.model._tie_weights()
            self._tie_or_clone_weights(self.lm_head, self.model.shared)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        phonetic_input_ids: Optional[torch.LongTensor] = None,
        phonetic_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, Seq2SeqLMOutput]:
        r"""
        phonetic_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of phonetic input sequence tokens in the vocabulary.
        phonetic_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices of phonetic input_ids.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
            information on the default strategy.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example summarization:

        ```python
        >>> from transformers import AutoTokenizer, BartForConditionalGeneration

        >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

        >>> ARTICLE_TO_SUMMARIZE = (
        ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
        ... )
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
        >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
        ```

        Mask filling example:

        ```python
        >>> from transformers import AutoTokenizer, BartForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

        >>> TXT = "My friends are <mask> but they eat too many carbs."
        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        ['not', 'good', 'healthy', 'great', 'very']
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            phonetic_input_ids=phonetic_input_ids,
            phonetic_attention_mask=phonetic_attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)


@auto_docstring(
    custom_intro="""
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """
)
class BartForSequenceClassification(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
            information on the default strategy.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@auto_docstring
class BartForQuestionAnswering(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, Seq2SeqQuestionAnsweringModelOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
            information on the default strategy.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BartDecoderWrapper(BartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = BartDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


@auto_docstring(
    custom_intro="""
    BART decoder with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """
)
class BartForCausalLM(BartPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = BartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BartForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        >>> model = BartForCausalLM.from_pretrained("facebook/bart-base", add_cross_attention=False)
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
        >>> list(logits.shape) == expected_shape
        True
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


__all__ = [
    "BartForCausalLM",
    "BartForConditionalGeneration",
    "BartForQuestionAnswering",
    "BartForSequenceClassification",
    "BartModel",
    "BartPreTrainedModel",
    "BartPretrainedModel",
    "PretrainedBartModel",
]