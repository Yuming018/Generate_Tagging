import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer
from copy import deepcopy
from collections import defaultdict, Counter
from helper import enconder

Relation = ['Causal Effect',
            'Temporal',
            # 'Coreference'
            ]

Event = ['State',
         'Action']

Event_arg = {'State':{
                'Entity' : None,
                'Trigger_Word' : None,
                'Value' : None,
                'Agent' : None,
                'Emotion' : None,
                'Time' : None,
                'Key' : None,
                'Topic' : None,
                'Emotion_Type' : None,
                'Place' : None,
                'Trigger_word' : None,
            },
             'Action': {
                'Actor' : None,
                'Trigger_Word' : None,
                'Direct Object' : None,
                'Msg (Direct)' : None,
                'Speaker' : None,
                'Place' : None,
                'Time' : None,
                'Addressee' : None,
                'Topic (Indirect)' : None,
                'Indirect Object' : None,
                'Tool or Method' : None,
                'Trigger_word' : None,
             }}

class Datasets:
    def __init__(self, path, model_name, relation_tag, tagging, path_save_model) -> None:
        self.path = path
        self.tagging = tagging
        self.max_len = 256
        self.count = 0
        if not relation_tag:
            self.index = 5 #index
            self.tagging_type = 'Event'
        elif relation_tag:
            self.index = 6 #index
            self.tagging_type = 'Relation'
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = self.get_data(path)
        if self.path == 'data/test.csv' and not self.tagging:
            self.model_tagging = self.get_data(path_save_model + 'tagging.csv')
        self.input, self.mask, self.target = [], [], []
        self.datasets = []
        self.temp = defaultdict(Counter)
        self.create_dataset()

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values

    def create_dataset(self):
        for idx in tqdm(range(len(self.dataset))):
            dict = {}
            tag_type = self.dataset[idx][self.index].split(' - ')[0]
            if tag_type in Event or tag_type in Relation:
                input_ids, attention_mask = self.create_input(idx)
                target_ids = self.create_target(idx)
                dict['input_ids'] = input_ids
                dict['attention_mask'] = attention_mask
                dict['labels'] = target_ids
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
                text += f"[{self.tagging_type}] {self.dataset[idx][self.index]}"
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
        # temp = deepcopy(Event_arg[self.dataset[idx][self.index].split(' - ')[0]])
        if self.tagging:
            text = f"[{self.tagging_type}] {self.dataset[idx][self.index]} "
            for i in range(7, len(self.dataset[idx])):
                if self.dataset[idx][i] != '':
                    left_parenthesis_index = self.dataset[idx][i].rfind('(')
                    if self.tagging_type == 'Relation' and i == 7:
                            text += " [Event1] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                    elif self.tagging_type == 'Relation' and i == 8:
                        text += " [Event2] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                    elif self.tagging_type == 'Event':
                        # text += " [Arg] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                        # self.temp[self.dataset[idx][self.index].split(' - ')[0]][self.dataset[idx][i][:left_parenthesis_index].split(' - ')[0]] = 1
                        # temp[self.dataset[idx][i][:left_parenthesis_index].split(' - ')[0]] = " ".join(self.dataset[idx][i][:left_parenthesis_index].split(' - ')[1:])
                        temp = " ".join(self.dataset[idx][i][:left_parenthesis_index].split(' - ')[1:])
                        text += f"[{self.dataset[idx][i][:left_parenthesis_index].split(' - ')[0]}] {temp} "
            # for key, val in temp.items():
            #     text += f'[{key}] {val} '
            text += "[END]"
        elif not self.tagging:
            text = self.dataset[idx][2]
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