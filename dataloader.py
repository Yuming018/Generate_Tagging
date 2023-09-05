import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer

class Dataset:
    def __init__(self, path, model_name) -> None:
        self.path = path
        self.max_len = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.dataset = self.get_data()
        self.input, self.mask, self.target = self.create_dataset()

    def get_data(self):
        data = pd.read_csv(self.path)
        data = data.fillna('')
        return data.values

    def create_dataset(self):
        _input, mask, target = [], [], []
        for idx in range(len(self.dataset)):
            if self.dataset[idx][5] != '':
                input_ids, attention_mask = self.create_input(idx)
                _input.append(input_ids)
                mask.append(attention_mask)
                target.append(self.create_target(idx))
        return np.array(_input), np.array(mask), np.array(target)

    def create_input(self, idx):
        context = self.create_context(idx)
        text = f"[SEP] {self.dataset[idx][0]} [SEP] {context} [SEP]"
        encoded_sent = self.tokenizer.encode_plus(
            text = text,  
            add_special_tokens=True,
            truncation=True,       
            max_length=self.max_len,             
            pad_to_max_length =True,         
            #return_tensors='pt',           
            return_attention_mask=True      
            )
        return encoded_sent.get('input_ids'), encoded_sent.get('attention_mask')

    def create_context(self, idx):
        min_b, max_b = float('inf'), float('-inf')
        for i in range(7, len(self.dataset[idx])):
            if self.dataset[idx][i] != '':
                numbers = re.findall(r'\d+', self.dataset[idx][i])
                min_b = min(min_b, int(numbers[-2]))
                max_b = max(max_b, int(numbers[-1]))
        words = re.findall(r'\S+|[\s]+', self.dataset[idx][1])
        # print("".join(words[min_b:max_b]))
        return "".join(words[min_b:max_b])

    def create_target(self, idx):
        text = f"[SEP] {self.dataset[idx][5]}"
        for i in range(7, len(self.dataset[idx])):
            if self.dataset[idx][i] != '':
                text += " [SEP] " + self.dataset[idx][i].split('(')[0]
        text += " [SEP]"
        encoded_sent = self.tokenizer.encode_plus(
            text = text,  
            add_special_tokens=True, 
            truncation=True,       
            max_length=self.max_len,             
            pad_to_max_length =True,   
            #return_tensors='pt',           
            return_attention_mask=True      
            )
        return encoded_sent.get('input_ids')

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        _input = torch.tensor(self.input[idx])
        mask = torch.tensor(self.mask[idx])
        target = torch.tensor(self.target[idx])
        return _input, mask, target

if __name__ == '__main__':
    pass