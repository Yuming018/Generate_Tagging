import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer

class Dataset:
    def __init__(self, path, model_name) -> None:
        self.path = path
        self.max_len = 128
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
            input_ids, attention_mask = self.create_input(idx)
            _input.append(input_ids)
            mask.append(attention_mask)
            target.append(self.create_target(idx))
        return np.array(_input), np.array(mask), np.array(target)

    def create_input(self, idx):
        encoded_sent = self.tokenizer.encode_plus(
            text = self.dataset[idx][1],  
            add_special_tokens=True,
            truncation=True,       
            max_length=self.max_len,             
            pad_to_max_length =True,         
            #return_tensors='pt',           
            return_attention_mask=True      
            )
        return encoded_sent.get('input_ids'), encoded_sent.get('attention_mask')

    def create_target(self, idx):
        text = ""
        for i in range(7, len(self.dataset[idx])):
            if self.dataset[idx][i] != '':
                text += " [SEP] " + self.dataset[idx][i]
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