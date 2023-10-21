from transformers import AutoTokenizer
from helper import enconder
import pandas as pd
import numpy as np
import re

class Dataset:
    def __init__(self, path, model_name) -> None:
        self.max_len = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.dataset = self.get_data(path)
        self.context, self.target = self.create_dataset()

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values
    
    def create_dataset(self):
        context, target = [], []
        story_name = self.dataset[0][4]
        story_idx = []
        for idx in range(len(self.dataset)):
            if self.dataset[idx][4] != story_name:
                context.append(self.create_input(idx-1, story_idx))
                target.append(self.create_target(idx-1))
                story_name = self.dataset[idx][4]
                story_idx = []
            story_idx.append(idx)
        return np.array(context), np.array(target)

    def create_input(self, idx, story_idx):
        context = self.text_segmentation(story_idx)
        text = f"[SEP] {self.dataset[idx][0]} [SEP] {context}"
        return text

    def create_target(self, idx):
        text = f"{self.dataset[idx][2]}"
        return text
    
    def text_segmentation(self, story_idx):
        min_b, max_b = float('inf'), float('-inf')
        for idx in story_idx:
            for i in range(7, len(self.dataset[idx])):
                if self.dataset[idx][i] != '':
                    numbers = re.findall(r'\d+', self.dataset[idx][i])
                    min_b = min(min_b, int(numbers[-2]))
                    max_b = max(max_b, int(numbers[-1]))
        words = re.findall(r'\S+|[\s]+', self.dataset[idx][1])
        if words[min_b] == ' ':
            min_b -= 1
        return "".join(words[min_b:max_b])
    
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

if __name__ == '__main__':
    pass