
import json
import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import multiprocessing
import pickle
import random
from functools import partial
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from FSProcessor import FSProcessor
# from RSSProcessor import RSSProcessor

def process_data_single(dat, llama_tokenizer, llama_3_tokenizer):
    new_token = llama_3_tokenizer.encode(llama_tokenizer.decode(dat))
    return {'input_ids': new_token}


class DataProcessor:
    def __init__(self, dataset_name, target_model_tokenizer_file, dataset_split, model_pre_id, model_post_id, use_length, target_length, model_name, processor_type, select_ratio):
        self.dataset_name = dataset_name
        self.target_model_tokenizer_file = target_model_tokenizer_file
        self.dataset_split = dataset_split
        self.model_pre_id = model_pre_id
        self.model_post_id = model_post_id
        self.use_length = use_length
        self.target_length = target_length
        self.model_name = model_name
        self.retain = 0
        self.select_ratio = select_ratio
        self.base_length = use_length - self.retain
        self.extend_length = target_length - self.retain

        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_pre_id, use_fast_tokenizer=True)
        self.llama_3_tokenizer = AutoTokenizer.from_pretrained(model_post_id, use_fast_tokenizer=True)

        if processor_type == "FS":
            self.processor = FSProcessor(
                model_path=model_post_id,
                base_length=self.base_length,
                extend_length=self.extend_length,
                select_ratio=self.select_ratio
            )

    def load_dataset_(self):
        return load_dataset(self.dataset_name, split=self.dataset_split)

    def process_data_parallel(self, dataset):
        print('Processing dataset in parallel...')
        new_dataset = [dat['input_ids'][:] for dat in tqdm.tqdm(dataset)]

        num_cpus = multiprocessing.cpu_count()
        num_processes = min(100, num_cpus)

        length = []
        with open(self.target_model_tokenizer_file, 'w') as f:
            with multiprocessing.Pool(processes=num_processes) as pool:
                process_func = partial(process_data_single, llama_tokenizer=self.llama_tokenizer, llama_3_tokenizer=self.llama_3_tokenizer)
                for new_data in tqdm.tqdm(pool.map(process_func, new_dataset)):
                    f.write(json.dumps(new_data) + '\n')
                    length.append(len(new_data['input_ids']))
        print(np.mean(length))

    def save_tokenized_data(self, processe_file):
        processed_data = []
        with open(processe_file, 'r') as f:
            for line in f:
                processed_data.append(json.loads(line))
        return processed_data

    def run(self):
        self.processor.run_process(processed_data)


if __name__ == "__main__":
    processor = DataProcessor(
        dataset_name='yaofu/slimpajama-per-source-length-upsample',
        target_model_tokenizer_file='output/processed_llama_3_s10l_full.json',
        dataset_split='train[0:10000]',
        model_pre_id="meta-llama/Llama-2-7b-hf",
        model_post_id="meta-llama/Meta-Llama-3-8B",
        use_length=24000,
        target_length=128000,
        model_name='__',
        processor_type='FS', 
        select_ratio=0.6
    )
    processor.run()
