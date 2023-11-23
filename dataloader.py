import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer
from helper import enconder

Relation = ['Causal Effect',
            'Temporal',
            # 'Coreference'
            ]

class Datasets:
    def __init__(self, path, model_name, relation_tag, tagging) -> None:
        self.path = path
        self.tagging = tagging
        self.max_len = 256
        self.count = 0
        if not relation_tag:
            self.type = 5 
            self.tagging_type = 'Event'
        elif relation_tag:
            self.type = 6 
            self.tagging_type = 'Relation'
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = self.get_data(path)
        if self.path == 'data/test.csv' and not self.tagging:
            self.model_tagging = self.get_data(f'save_model/{self.tagging_type}/tagging.csv')
        self.input, self.mask, self.target = [], [], []
        self.datasets = []
        self.create_dataset()

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values

    def create_dataset(self):
        for idx in tqdm(range(len(self.dataset))):
            dict = {}
            if self.dataset[idx][self.type] != '' and (self.type == 5 or self.dataset[idx][self.type].split(' - ')[0] in Relation):
                input_ids, attention_mask = self.create_input(idx)
                target_ids = self.create_target(idx)
                dict['input_ids'] = input_ids
                dict['attention_mask'] = attention_mask
                dict['labels'] =target_ids
                self.input.append(input_ids)
                self.mask.append(attention_mask)
                self.target.append(target_ids)
                self.datasets.append(dict)

    def create_input(self, idx):
        context = self.text_segmentation(idx)
        text = f"[Type] {self.dataset[idx][0]} [Context] {context} "
        if self.tagging:
            text += '[END]'
        else:
            if self.path == 'data/test.csv':
                text += f'[CLS] {self.model_tagging[self.count][2]}'
                self.count += 1
            else:  
                text += f"[{self.tagging_type}] {self.dataset[idx][self.type]}"
                for i in range(7, len(self.dataset[idx])):
                    if self.dataset[idx][i] != '':
                        left_parenthesis_index = self.dataset[idx][i].rfind('(')
                        if self.tagging_type == 'Relation' and i == 7:
                            text += " [Event1] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                        elif self.tagging_type == 'Relation' and i == 8:
                            text += " [Event2] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                        elif self.tagging_type == 'Event':
                            text += " [Arg] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                text += " [END]"
        
        encoded_sent = enconder(self.tokenizer, self.max_len, text = text)
        return encoded_sent.get('input_ids'), encoded_sent.get('attention_mask')

    def text_segmentation(self, idx):
        min_b, max_b = float('inf'), float('-inf')
        for i in range(7, len(self.dataset[idx])):
            if self.dataset[idx][i] != '':
                numbers = re.findall(r'\d+', self.dataset[idx][i])
                min_b = min(min_b, int(numbers[-2]))
                max_b = max(max_b, int(numbers[-1]))
        words = re.findall(r'\S+|[\s]+', self.dataset[idx][1])
        if words[min_b] == ' ':
            min_b -= 1
        return "".join(words[min_b:max_b])

    def create_target(self, idx):
        if self.tagging:
            text = f"[{self.tagging_type}] {self.dataset[idx][self.type]}"
            for i in range(7, len(self.dataset[idx])):
                if self.dataset[idx][i] != '':
                    left_parenthesis_index = self.dataset[idx][i].rfind('(')
                    if self.tagging_type == 'Relation' and i == 7:
                            text += " [Event1] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                    elif self.tagging_type == 'Relation' and i == 8:
                        text += " [Event2] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                    elif self.tagging_type == 'Event':
                        text += " [Arg] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
            text += " [END]"
        elif not self.tagging:
            text = f"{self.dataset[idx][2]}"

        encoded_sent = enconder(self.tokenizer, self.max_len, text = text)
        return encoded_sent.get('input_ids')

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.datasets[idx]
        _input = torch.tensor(self.input[idx])
        mask = torch.tensor(self.mask[idx])
        target = torch.tensor(self.target[idx])
        return _input, mask, target

if __name__ == '__main__':
    pass