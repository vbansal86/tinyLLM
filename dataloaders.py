from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import ByteLevelBPETokenizer

from utils import group_texts, collate_fn_with_padding, collate_fn_fixed_length, tokenize_function  

def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i+batch_size]["pn_history"]



def get_data_splits(hf_path,tokenizer_model="gpt2"):
    raw_ds = load_dataset(hf_path,split="train")  # split: 'train', 'validation' if available
    if tokenizer_model =="custom":
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            batch_iterator(raw_ds),
            vocab_size=30000,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
            )
        #tokenizer.save_model("my_tokenizer")

    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
        # ensure pad_token exists for batching
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # 1) Load text dataset (replace with your choice)
    

    block_size = 1024  # must be <= model max_seq_len
    tokenized = raw_ds.map(tokenize_function, batched=True, remove_columns=raw_ds.column_names)

    # 2) Group into fixed-length blocks for next-token prediction

    lm_ds = tokenized.map(group_texts, batched=True,remove_columns=['attention_mask'])
    train_ds = lm_ds
    eval_ds = lm_ds["validation"] if "validation" in lm_ds else None  # some datasets lack validation split
    return train_ds, eval_ds, tokenizer