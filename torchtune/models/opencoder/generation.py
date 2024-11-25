from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

@dataclass
class OpenCoderGenerationConfig:
    max_length: int = 2048
    min_length: int = 0
    do_sample: bool = True
    early_stopping: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    bad_words_ids: Optional[List[List[int]]] = None
    no_repeat_ngram_size: int = 0
    num_return_sequences: int = 1
    attention_mask: Optional[Tensor] = None
    use_cache: bool = True

def prepare_inputs_for_generation(
    input_ids: Tensor,
    past_key_values: Optional[List[Tensor]] = None,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    use_cache: bool = True,
    **kwargs
) -> dict:
    """Prepare inputs for generation step."""
    # only last token for inputs_ids if past is defined in kwargs
    if past_key_values is not None:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if position_ids is not None:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }

def adjust_logits_during_generation(
    logits: Tensor,
    cur_len: int,
    max_length: int,
    min_length: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    bad_words_ids: Optional[List[List[int]]],
    input_ids: Tensor,
) -> Tensor:
    """Adjust token logits during generation."""
    # repetition penalty
    if repetition_penalty != 1.0:
        for previous_tokens in input_ids:
            # get score for previously generated tokens
            scores = logits.gather(1, previous_tokens.unsqueeze(1))
            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            logits.scatter_(1, previous_tokens.unsqueeze(1), scores)

    # no_repeat_ngram_size
    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens according to no_repeat_ngram_size
        banned_tokens = calc_banned_ngram_tokens(input_ids, no_repeat_ngram_size)
        for batch_idx, banned_tokens_slice in enumerate(banned_tokens):
            logits[batch_idx, banned_tokens_slice] = -float("inf")

    # bad_words_ids
    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)
        for batch_idx, banned_tokens_slice in enumerate(banned_tokens):
            logits[batch_idx, banned_tokens_slice] = -float("inf")

    # min_length constraint
    if min_length > 0 and cur_len < min_length:
        logits[:, 2] = -float("inf")  # 2 is the eos_token_id

    return logits

def calc_banned_ngram_tokens(
    prev_input_ids: Tensor, 
    no_repeat_ngram_size: int
) -> List[List[int]]:
    """Calculate banned tokens based on n-gram repetition constraint."""
    banned_tokens = []
    for batch_idx in range(prev_input_ids.shape[0]):
        banned_tokens_slice = []
        for ngram_size in range(no_repeat_ngram_size, 0, -1):
            ngram = prev_input_ids[batch_idx, -ngram_size:].tolist()
            for i in range(len(prev_input_ids[batch_idx]) - ngram_size + 1):
                if prev_input_ids[batch_idx, i:i + ngram_size].tolist() == ngram:
                    banned_tokens_slice.append(prev_input_ids[batch_idx, i + ngram_size - 1].item())
        banned_tokens.append(list(set(banned_tokens_slice)))
    return banned_tokens

def calc_banned_bad_words_ids(
    prev_input_ids: Tensor,
    bad_words_ids: List[List[int]]
) -> List[List[int]]:
    """Calculate banned tokens based on bad words list."""
    banned_tokens = []
    for batch_idx in range(prev_input_ids.shape[0]):
        banned_tokens_slice = []
        for word_ids in bad_words_ids:
            if len(word_ids) == 1:
                banned_tokens_slice.append(word_ids[0])
            else:
                if word_ids == prev_input_ids[batch_idx, -len(word_ids):].tolist():
                    banned_tokens_slice.append(word_ids[-1])
        banned_tokens.append(list(set(banned_tokens_slice)))
    return banned_tokens