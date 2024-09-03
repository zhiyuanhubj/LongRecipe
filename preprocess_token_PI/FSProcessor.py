import json
import tqdm
import numpy as np
from transformers import AutoTokenizer
import pickle
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class FSProcessor:
    def __init__(self, model_path, base_length=24000, extend_length=80000, min_num=1, max_num=1, select_ratio=0.8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.base_length = base_length
        self.extend_length = extend_length
        self.min_num = min_num
        self.max_num = max_num
        self.select_ratio = select_ratio
        self.sentences = []
        self.indexes = []

    def flatten_list(self, lst):
        flattened_list = []
        for item in lst:
            if isinstance(item, list):
                flattened_list.extend(self.flatten_list(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def split_sentences(self, data, tokens_to_split):
        split_lists = []
        split_idxs = []

        current_split = []
        current_idx = []

        for idx, num in enumerate(data):
            if num in tokens_to_split:
                if current_split:
                    current_split.append(num)
                    current_idx.append(idx)
                    split_lists.append(current_split)
                    split_idxs.append(current_idx)
                    current_split = []
                    current_idx = []
                else:
                    current_split.append(num)
                    current_idx.append(idx)
                    split_lists.append(current_split)
                    split_idxs.append(current_idx)
                    current_split = []
                    current_idx = []
            else:
                current_split.append(num)
                current_idx.append(idx)
        if current_split:
            split_lists.append(current_split)
            split_idxs.append(current_idx)
        return split_lists, split_idxs

    def load_and_process_data(self, dataset):
        length_6k = [data['input_ids'][:self.base_length] for data in dataset]
        tokens_to_split = [13, 198, 30, 0, 627]

        for data in tqdm.tqdm(length_6k):
            data = self.flatten_list(data)
            split_lists, split_idxs = self.split_sentences(data,tokens_to_split)
            self.sentences.append([sent for sent in split_lists])
            self.indexes.append([sent for sent in split_idxs])

    def process_index(self, idx):
        selected = []
        tokenized = self.tokenizer.batch_decode(idx)
        for ids, tokens in zip(idx, tokenized):
            if any(char.isdigit() for char in tokens):
                selected.append(ids)
            else:
                random_number = random.randint(1, 10)
                if random_number < int(self.select_ratio * 10):
                    continue
                else:
                    selected.append(ids)
        return selected

    def process_all_indexes(self, max_workers=40):
        selected = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_index, idx) for idx in self.sentences]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                selected.append(future.result())
        return selected

    def random_chunk_list(self, input_list):
        result = []
        i = 0
        chunk_sizes = []
        while i < len(input_list):
            chunk_size = random.randint(self.min_num, self.max_num)
            result.append(input_list[i:i + chunk_size])
            i += chunk_size
            chunk_sizes.append(chunk_size)
        return result, chunk_sizes

    def process_new_indexes(self):
        new_total_indexes = []
        for index in tqdm.tqdm(self.indexes):
            result, chunk_size = self.random_chunk_list(list(range(len(index))))
            new_indexes = []
            for idx in result:
                new_index = []
                for id in idx:
                    new_index.extend(index[id])
                new_indexes.append(new_index)
            new_total_indexes.append(new_indexes)
        return new_total_indexes

    def generate_res_lists(self, new_total_indexes):
        res_lists = []
        for index in tqdm.tqdm(new_total_indexes):
            random_zeros = [0] * (self.extend_length - self.base_length + len(index))

            insert_indices = sorted(random.sample(range(len(random_zeros)), len(index)))

            random_zeros[0] = index[0]
            for random_number, idx in zip(insert_indices[1:], index[1:]):
                random_zeros[random_number] = idx

            res_list = self.flatten_list(random_zeros)
            PI_list = [id for id in range(len(res_list)) if res_list[id] != 0]
            new_PI_list = [PI_list[0] - 1] + PI_list
            if new_PI_list[0] == -1:
                updated_PI_list = [x + 1 for x in new_PI_list]
            else:
                updated_PI_list = new_PI_list
            res_lists.append(updated_PI_list)
        return res_lists

    def save_data(self, data, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(data, f)

    def run_process(self,data):
        self.load_and_process_data(data)
        selected = self.process_all_indexes()
        lengths = [len(item) for item in selected]
        print(f'sample_length: {np.mean(lengths)}')

        with open(f'output/feature_{str(int(self.select_ratio*10))}_tokens.jsonl', 'w') as f:
            for new_data in tqdm.tqdm(selected):
                f.write(json.dumps({'input_ids': new_data}) + '\n')

        new_total_indexes = self.process_new_indexes()
        res_lists = self.generate_res_lists(new_total_indexes)

        print(len(res_lists[0]))
        self.save_data(res_lists, f"output/llama3_LF_{str(int(self.select_ratio*10))}_digits_l{self.base_length}_t{self.extend_length}_min{self.min_num}_max{self.max_num}.pkl")

        right_points = [random.randint(1, (self.base_length + 1) // 2) for _ in range(len(res_lists))]
        self.save_data(right_points, f"output/llama3_LPS_{str(int(self.select_ratio*10))}_digits_l{self.base_length}_t{self.extend_length}_min{self.min_num}_max{self.max_num}.pkl")
