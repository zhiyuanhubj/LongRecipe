import os
import numpy as np
import tqdm
import tiktoken
import random
import pickle
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union
from transformers import TrainingArguments, HfArgumentParser
import torch
import random

def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    model_type,
    setting_choice,
    seq_length_, 
    target_len,
    rts_file, 
    rss_s_file, 
    num_proc, 
) -> Union["Dataset", "IterableDataset"]:

    global seq_length
    seq_length = seq_length_
    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        
        global seq_length
        
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding):
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=True)
        
        if hasattr(tokenizer, "add_eos_token"):
            add_eos_token_flag = getattr(tokenizer, "add_eos_token")
            setattr(tokenizer, "add_eos_token", True)
        
        if setting_choice == 'full_str': 
            sentence = []
            for data_1, data_2 in zip(examples["prompt"], examples["response"]):
                sentence.append(data_1 + data_2)

            tokenized_examples = tokenizer(sentence, **kwargs)
        
        else:
            tokenized_examples = {"input_ids": examples["input_ids"]}

        try:
            new_tokenized_examples = {'input_ids': [data[:] for data in tokenized_examples['input_ids']]}
        except:
            new_tokenized_examples = {'input_ids': [data[:] for data in tokenized_examples]}

        concatenated_examples = {k: list(chain(*new_tokenized_examples[k])) for k in new_tokenized_examples.keys()}


        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        
        block_size = seq_length
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # make sure the saved tokenizer is the same as the original one
        if hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_eos_token", add_eos_token_flag)
        
        if setting_choice == 'full':     
            return result

        if setting_choice == 'LongRecipe':

            with open(rts_file, 'rb') as fr:
                rt1s = pickle.load(fr)
            with open(rss_s_file, 'rb') as fr:
                topk = pickle.load(fr)

            input_ids = []
            position_ids = []

            retain = 0

            for idx, ids in tqdm.tqdm(enumerate(new_tokenized_examples["input_ids"])):
                ids = flatten_list(ids)
                base_length = min(seq_length, len(ids))

                if len(topk[idx]) == 0:
                    continue
                rt1 = rt1s[idx]
                new_input_ids = ids[:base_length]
                pos_ids = torch.arange(retain + rt1).tolist()
                new_position_ids = topk[idx][rt1:base_length]
                pos_ids.extend(new_position_ids)
                
                try:
                    assert len(pos_ids) == len(new_input_ids)
                    position_ids.append(pos_ids)
                    input_ids.append(new_input_ids)
                except:
                    print(len(pos_ids))
                    print(len(new_input_ids))
                
            model_inputs = {"input_ids": input_ids, "position_ids": position_ids}
            return model_inputs


        elif setting_choice == 'R':

            input_ids = []
            position_ids = []
                
            retain = 0
            
            
            for idx, ids in tqdm.tqdm(enumerate(new_tokenized_examples["input_ids"])):
                base_length = min(seq_length, len(ids))
                new_input_ids = ids[:base_length]
                
                pos_ids = torch.arange(retain, dtype=torch.long).tolist()
                pos_ids.extend(sorted(random.sample(list(range(retain, target_len)), base_length-retain)))

                input_ids.append(new_input_ids)
                position_ids.append(pos_ids)
                try:
                    assert len(pos_ids) == len(new_input_ids)
                except:
                    print(len(pos_ids))
                    print(len(new_input_ids))

            model_inputs = {"input_ids": input_ids, "position_ids": position_ids}
            return model_inputs
               

        elif setting_choice == 'pose': 
            with open(rts_file, 'rb') as fr:
                rt1s = pickle.load(fr)
    
            
            with open(rss_s_file, 'rb') as fr:
                topk = pickle.load(fr)
                
            rts = []
            lt1s = []
            input_ids = []
            position_ids = []

            retain = 0
            
            scaled_max_position_embeddings = target_len
            
            
            for idx, ids in tqdm.tqdm(enumerate(new_tokenized_examples["input_ids"])):


                base_length = min(seq_length, len(ids))
                len_chunk = min(len(ids), base_length)
                len_input = len(ids)
                lt1 = 0
                rt1 = rt1s[idx]
                lt1 += retain; rt1 += retain
                chunked_ids = ids[:len_chunk]
                new_input_ids = ids[:retain] + chunked_ids
                input_ids.append(new_input_ids)

                pos_ids = torch.arange(len(chunked_ids), dtype=torch.long)
                len_pos_ids = len(pos_ids)
                lt = random.randint(0, scaled_max_position_embeddings-len_pos_ids)
                # lt = 0 
                rt = random.randint(lt, scaled_max_position_embeddings-len_pos_ids)
                
                new_pos_ids = torch.arange(lt, dtype=torch.long)
                new_pos_ids = torch.cat((new_pos_ids, pos_ids[lt:] + (rt + retain)))

                position_ids.append(new_pos_ids)

            model_inputs = {"input_ids": input_ids, "position_ids": position_ids}
            return model_inputs


    dataset = dataset.map(
        preprocess_pretrain_dataset,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    return dataset