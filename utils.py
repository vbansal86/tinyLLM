import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformernew import MiniGPT
from torch import nn
from torch.nn.utils.rnn import pad_sequence
block_size = 1024  # must be <= model max_seq_len

# Import or define your tokenizer here
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace "gpt2" with your model name if needed






def group_texts(examples):
    # concatenate
    concatenated = []
    for ids_list in examples["input_ids"]:
        concatenated.extend(ids_list)
    # drop remainder to make exact blocks
    total_len = (len(concatenated) // block_size) * block_size
    concatenated = concatenated[:total_len]
    # split into chunks
    input_ids = [concatenated[i:i+block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": input_ids}

def collate_fn_with_padding(features):
    ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def collate_fn_fixed_length(features):
    # features: list of dicts with "input_ids"
    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    # attention_mask is optional for fixed-length causal models
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# def tokenize_function(examples):
#     print("KEYS examples:", examples.keys())
#     # join list of strings into one since TinyStories has short samples
#     # texts = examples["He"] if isinstance(examples["He"], list) else [examples["He"]]
#     texts = examples["He"] if isinstance(examples["He"], list) else [examples["She"] if isinstance(examples["She"], list) else None ]
#     return tokenizer(texts, add_special_tokens=False)

def tokenize_function(examples):
    # texts = [i[0] + " " + i[1] for i in zip(examples["He"], examples["She"]) if i[0] is not None and i[1] is not None]
    texts = examples["pn_history"]
    return tokenizer(texts, add_special_tokens=False, truncation=True, max_length=block_size)