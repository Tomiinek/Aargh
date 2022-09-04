import torch
from itertools import chain
from torchvision.transforms import Compose
from aargh.data.transforms import BaseTransform


def batchify(data, tokenizer, padding=True, enable_wrapping=True):
    """
    Convert a batch of strings or list of strings into a proper batch:
        1) If provided a list, pad it to NUM_C by [].
        2) Flatten to ([B, NUM_C, MAX_LEN] -> [B * NUM_C, MAX_LEN]).
        3) Call tokenizer (i.e.) pad it to max length and encode tokens to IDs.
        4) Reshape to original lengths (([B * NUM_C, MAX_LEN] -> [B, NUM_C, MAX_LEN]))
    Otherwise just tokenize and return token IDs and attention mask.

    Argumnets:
        data (tuple(str) or tuple(tuple(str, str)) tuple(List[str])): Batch to be prepared. 
        tokenizer: Instance of Tokenizer
        padding: If True, batch is padded to maximal length
        enable_wrapping: If True, batch might be wrapped into BOS, EOS (depends on config), otherwise must not.
    """
    is_nested = isinstance(data[0], list)
    
    if is_nested:
        s = len(data)
        x = max([len(l) for l in data])
        for l in data:
            l += [''] * (x - len(l))
        data = list(chain.from_iterable(data))

    encoding = tokenizer.encode_batch(data, padding, enable_wrapping)
    
    if is_nested:
        encoding.token_ids =      [encoding.token_ids[i:i + x] for i in range(0, len(encoding.token_ids), x)]
        encoding.attention_mask = [encoding.attention_mask[i:i + x] for i in range(0, len(encoding.attention_mask), x)]
    
    return encoding


def mask_until_last_nonempty(batch, index=-100):
    max_len = batch.size(1)
    _, ctxt_last = torch.max(torch.flip(batch != index, [1]), dim=1)
    return torch.arange(max_len, device=batch.device)[None, :] < (max_len - ctxt_last)[:, None]


def tokens_from_transforms(transforms):
    ts = transforms.transforms if isinstance(transforms, Compose) else transforms
    new_tokens = []
    for t in ts:
        if not isinstance(t, BaseTransform):
            tokens = tokens_from_transforms(t)
        else:
            tokens = t.get_new_tokens()
        new_tokens.extend(tokens)
    return new_tokens

def time_str_to_minutes(time_string):
    try:
        return int(time_string.split(':')[0]) * 60 + int(time_string.split(':')[1])
    except:
        return 0
