import torch
import sys
sys.path.append('../src')
import random
import json
from typing import Dict, List, Sequence, Tuple, Any, Literal
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.tuning.data import SFTDataCollatorWith4DAttentionMask, get_dataset
from src.tuning.data.processors.supervised import (
    preprocess_supervised_dataset,
)
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial
import logging
from torch import tensor
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass

@dataclass
class SFTDataCollator(SFTDataCollatorWith4DAttentionMask):
    r"""
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        print( features[0].keys())
        keys = list([k for k in features[0].keys() if k in ['input_ids', 'labels', 'query_id']])
        pad_features = ['input_ids', 'labels']
        extra = {k: [example[k] for example in features] for k in keys if k not in pad_features}
        features = super().__call__([{k: example[k] for k in pad_features} for example in features])
        features.update(extra)
        return features


def causal_cross_entropy(model, input_ids: tensor, labels: tensor):
    batch_size = input_ids.size(0)
    print(batch_size)
    with torch.no_grad():
        logits = model(input_ids=input_ids, labels=None)['logits']
        shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_length-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_length-1]
        # # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        loss_per_sample = []
        for slabel, slogit in zip(shift_labels, shift_logits):
            # slogit: [seq_length-1, vocab_size], slabel: [seq_length-1]
            loss = loss_fct(slogit, slabel)  # no unsqueeze(0)
            loss_per_sample.append(loss)
        loss_per_sample = torch.stack(loss_per_sample)  # -> [batch_size]
    return loss_per_sample

def load_tokenizer(model_name):
    def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logging.info("Add eos token: {}".format(tokenizer.eos_token))
        else:
            logging.info("Replace eos token: {}".format(tokenizer.eos_token))

        if num_added_tokens > 0:
            logging.warning("New tokens have been added, make sure `resize_vocab` is True.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Add pad token: {}".format(tokenizer.pad_token))
    return tokenizer

def label_probability(
        model_name,
        data: List[Dict],
        batch_size: int,
        device: str = 'cuda'
):
    tokenizer = load_tokenizer(model_name)
    train_dataset = Dataset.from_list(data).map(
        preprocess_supervised_dataset, 
        batched=True, 
        fn_kwargs={"tokenizer": tokenizer, 'cutoff_len': 8192}, 
        num_proc=6,
        remove_columns=['messages']
    )
    print(train_dataset[0])
    data_collator = SFTDataCollator(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
    )
    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=4, shuffle=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    dtypes = set(param.dtype for param in model.parameters())
    model.eval()

    func = partial(causal_cross_entropy, model=model)
    log_probs, query_ids, sample_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(train_dataloaders):
            logprobs = func(input_ids=batch['input_ids'].to(device), labels=batch['labels'].to(device)).cpu().tolist()  # 32
            log_probs.extend(logprobs)
            print(logprobs)
            query_ids.extend(batch['query_id'])
            # sample_ids.extend(batch['sample_id'])
    # results = [{"log_probs": p, "query_id": q_id, "sample_id":s_id} for (p, q_id, s_id) in zip(log_probs, query_ids, sample_ids)]
    results = [{"log_probs": p, "query_id": q_id} for (p, q_id) in zip(log_probs, query_ids)]
    return results


import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--cutoff_len", type=int, default=8192)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    data=json.load(open(args.input_file))
    results = label_probability(
        model_name=args.model_name,
        data=data,
        batch_size=args.batch_size,
    )
    json.dump(results, open(args.output_file,'w'), indent=4)