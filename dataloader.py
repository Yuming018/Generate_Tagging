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

Relation_definition = {
    'X intent' : "Why does X cause the event?",
    'X reaction' : "How does X feel after the event?",
    'Other reaction' : "How do others' feel after the event?",
    'X attribute' : "How would X be described?",
    'X need' : "What does X need to do before the event?",
    'Effect on X' : "What effects does the event have on X?",
    'X want' : "What would X likely want to do after the event?",
    'Other want' : "What would others likely want to do after the event?",
    'Effect on other' : "What effects does the event have on others?",
    'isBefore' : "No causal relationship exists; Event1 occurs before Event2.",
    'the same' : "No causal relationship exists; Event1 and Event2 occur simultaneously.",
    'isAfter' : "No causal relationship exists; Event1 occurs after Event2.",
}
Relation_definition_2 = {
    'If-Event-Then-Mental-State' : "Contains 'X intent', 'X reaction' and 'Other reaction'. Define three relations relating to the mental pre- and post-conditions of an event. ",
    'If-Event-Then-Event' : "Contains 'X need', 'Effect on X', 'X want', 'Other want', 'Effect on other', 'isBefore', 'the same' and 'isAfter'. Define five relations relating to events that constitute probable pre- and postconditions of a given event. Those relations describe events likely required to precede an event, as well as those likely to follow.",
    'If-Event-Then-Persona' : "Contains 'X attribute'. In addition to pre- and postconditions, Define a stative relation that describes how the subject of an event is described or perceived.",
}

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
            self.model_tagging = self.get_data("/".join(path_save_model.split('/')[:-2]) + '/tagging/tagging.csv')
        self.input, self.mask, self.target = [], [], []
        self.datasets = []
        self.temp = defaultdict(Counter)
        self.dic = Counter()
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
                self.dic[self.dataset[idx][self.index].split(' - ')[1]] += 1
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
        text = "Please utilize the provided context, question types, and type definitions to generate key information for this context, along with corresponding types ."
        # text += '[Type definitions] '
        for key, definition in Relation_definition_2.items():
            text += f'[{key}] {definition} '
        text += f"[Type] {self.dataset[idx][0]} [Context] {context} "
        if self.tagging:
            text += '[END]'
        else:
            if self.path == 'data/test.csv':
                text += f'{self.model_tagging[self.count][2]}'
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
                            temp = " ".join(self.dataset[idx][i][:left_parenthesis_index].split(' - ')[1:])
                            text += f"[{self.dataset[idx][i][:left_parenthesis_index].split(' - ')[0]}] {temp} "
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
                        temp = " ".join(self.dataset[idx][i][:left_parenthesis_index].split(' - ')[1:])
                        text += f"[{self.dataset[idx][i][:left_parenthesis_index].split(' - ')[0]}] {temp} "
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