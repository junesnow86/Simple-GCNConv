import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class YelpDataset(Dataset):
    def __init__(self, path_to_json, tokenizer=None, max_length=200) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path_to_json, 'r') as f:
            data = json.load(f)
        self.data = [(d['text'], d['stars']) for d in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, stars = self.data[index]
        tokenized = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True)
        return torch.tensor(tokenized.input_ids), torch.tensor(stars-1)
