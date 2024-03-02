import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer
from copy import deepcopy
from collections import defaultdict, Counter
from helper import enconder, text_segmentation

legal_tagging = ['Causal Effect',
            'Temporal',
            # 'Coreference'
            'State',
            'Action'
]

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

class Tagging_Datasets:
    def __init__(self, path, model_name, event_or_relation) -> None:
        self.path = path
        self.max_len = 512
        self.tagging_type = event_or_relation
        if event_or_relation == 'Event':
            self.index = 5 #index
        elif event_or_relation == 'Relation':
            self.index = 6 #index
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = self.get_data(path)
        self.datasets = []
        self.create_dataset()  

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values

    def create_dataset(self):
        story_name = "-".join(self.dataset[0][4].split('-')[:-1])
        story_list = []
        for idx in tqdm(range(len(self.dataset))):
            dict = {}
            tag_type = self.dataset[idx][self.index].split(' - ')[0]
            if self.tagging_type == 'Event':
                if tag_type in legal_tagging:
                    input_ids, attention_mask = self.create_input([idx])
                    target_ids = self.create_target([idx])
                    dict['input_ids'] = input_ids
                    dict['attention_mask'] = attention_mask
                    dict['labels'] = target_ids
                    self.datasets.append(dict)
            elif self.tagging_type == 'Relation':
                current_story_name = "-".join(self.dataset[idx][4].split('-')[:-1])
                if current_story_name != story_name and story_list and tag_type in legal_tagging:
                    input_ids, attention_mask = self.create_input(story_list)
                    target_ids = self.create_target(story_list)
                    dict['input_ids'] = input_ids
                    dict['attention_mask'] = attention_mask
                    dict['labels'] = target_ids
                    self.datasets.append(dict)

                    story_name = current_story_name
                    story_list = []
                if tag_type in legal_tagging:
                    story_list.append(idx)
        return          

    def create_input(self, story_list):
        if self.tagging_type == 'Event':
            context = text_segmentation(self.dataset[story_list[0]])
        elif self.tagging_type == 'Relation':
            context = self.dataset[story_list[0]][1]
        # for key, definition in Relation_definition_2.items():
        #     text += f'[{key}] {definition} '
        # text += f"[Type] {self.dataset[idx][0]} [Context] {context} "
        
        text = f"Please utilize the provided context to generate {self.tagging_type} 1 "
        for i in range(1, len(story_list)):
            text += f"and {self.tagging_type} {i+1} "
        text += f"key information for this context [Context] {context} [END]"
        
        encoded_sent = enconder(self.tokenizer, self.max_len, text = text)
        # print(encoded_sent.get('input_ids'))
        # print(self.tokenizer.decode(encoded_sent.get('input_ids'), skip_special_tokens=True))        
        return encoded_sent.get('input_ids'), encoded_sent.get('attention_mask')

    def create_target(self, story_list):
        """
        Tagging : 
            input : story list
            output : 所有 list 中 tagging 的集合
            
            output format : 
            [Relation 1] Causal Effect - X intent 
            [Event1] " who are these two? " asked the snow man of the yard - dog 
            [Event2] you have been here longer than i have
            [Relation 2] Causal Effect - X intent 
            [Event1] i never bite those two 
            [Event2] she has stroked my back many times 
            [END]
        """
        text = ""
        for idx, story_idx in enumerate(story_list):
            text += f"[{self.tagging_type} {idx+1}] {self.dataset[story_idx][self.index]} "
            for i in range(7, len(self.dataset[story_idx])):
                if self.dataset[story_idx][i] != '':
                    left_parenthesis_index = self.dataset[story_idx][i].rfind('(')
                    if self.tagging_type == 'Relation' and i == 7:
                            text += " [Event1] " + "".join(self.dataset[story_idx][i][:left_parenthesis_index])
                    elif self.tagging_type == 'Relation' and i == 8:
                        text += " [Event2] " + "".join(self.dataset[story_idx][i][:left_parenthesis_index])
                    elif self.tagging_type == 'Event':
                        # text += " [Arg] " + "".join(self.dataset[story_idx][i][:left_parenthesis_index])
                        # self.temp[self.dataset[story_idx][self.index].split(' - ')[0]][self.dataset[story_idx][i][:left_parenthesis_index].split(' - ')[0]] = 1
                        temp = " ".join(self.dataset[story_idx][i][:left_parenthesis_index].split(' - ')[1:])
                        text += f"[{self.dataset[story_idx][i][:left_parenthesis_index].split(' - ')[0]}] {temp} "
        text += " [END]"

        encoded_sent = enconder(self.tokenizer, self.max_len, text = text)
        # print(encoded_sent.get('input_ids'))
        # print(self.tokenizer.decode(encoded_sent.get('input_ids'), skip_special_tokens=True))
        # input()
        return encoded_sent.get('input_ids')

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]

class Question_Datasets:
    def __init__(self, path, model_name, path_save_model) -> None:
        self.path = path
        self.max_len = 1024
        self.count = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = self.get_data(path)
        if self.path == 'data/test.csv':
            self.model_tagging = self.get_data("/".join(path_save_model.split('/')[:-2]) + '/tagging/tagging.csv')
        self.datasets = []
        self.create_dataset()  

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values

    def create_dataset(self):
        story_name = "-".join(self.dataset[0][4].split('-')[:-1])
        story_list = []
        for idx in tqdm(range(len(self.dataset))):
            dict = {}
            tag_type = self.dataset[idx][self.index].split(' - ')[0]
            current_story_name = "-".join(self.dataset[idx][4].split('-')[:-1])
            if current_story_name != story_name and story_list:
                input_ids, attention_mask = self.create_input(story_list)
                target_ids = self.create_target(story_list)
                dict['input_ids'] = input_ids
                dict['attention_mask'] = attention_mask
                dict['labels'] = target_ids
                self.datasets.append(dict)

                story_name = current_story_name
                story_list = []
            if tag_type in legal_tagging:
                story_list.append(idx)
        return          

    def create_input(self, story_list):
        # context = text_segmentation(self.dataset[idx])
        context = self.dataset[story_list[0]][1]
        # for key, definition in Relation_definition_2.items():
        #     text += f'[{key}] {definition} '
        # text += f"[Type] {self.dataset[idx][0]} [Context] {context} "
        
        text = f"Please utilize the provided context and key information to generate question for this context "
        text += f'[Context] {context} '
        if self.path == 'data/test.csv':
            text += self.model_tagging[self.count][2]
            self.count += 1
        else:  
            for idx in story_list:
                tagging_type = ""
                text += f"[{tagging_type}] {self.dataset[idx][self.index]}"
                for i in range(7, len(self.dataset[idx])):
                    if self.dataset[idx][i] != '':
                        left_parenthesis_index = self.dataset[idx][i].rfind('(')
                        if tagging_type == 'Relation' and i == 7:
                            text += " [Event1] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                        elif tagging_type == 'Relation' and i == 8:
                            text += " [Event2] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
                        elif tagging_type == 'Event':
                            temp = " ".join(self.dataset[idx][i][:left_parenthesis_index].split(' - ')[1:])
                            text += f"[{self.dataset[idx][i][:left_parenthesis_index].split(' - ')[0]}] {temp} "
            text += "[END]"
        
        encoded_sent = enconder(self.tokenizer, self.max_len, text = text)
        # print(encoded_sent.get('input_ids'))
        # print(self.tokenizer.decode(encoded_sent.get('input_ids'), skip_special_tokens=True))        
        return encoded_sent.get('input_ids'), encoded_sent.get('attention_mask')

    def create_target(self, story_list):
        text = self.dataset[story_list[0]][2]
        encoded_sent = enconder(self.tokenizer, self.max_len, text = text)
        # print(encoded_sent.get('input_ids'))
        # print(self.tokenizer.decode(encoded_sent.get('input_ids'), skip_special_tokens=True))
        # input()
        return encoded_sent.get('input_ids')

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]

if __name__ == '__main__':
    pass