import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from model import create_model
from helper import enconder, text_segmentation, create_prompt
from transformers import DataCollatorWithPadding
from copy import deepcopy
from functools import lru_cache
from helper import checkdir

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

class Question_generation_Datasets:
    def __init__(self, path, model_name, tokenizer, max_len, gen_answer) -> None:
        self.path = path
        self.max_len = max_len
        self.gen_Answer = gen_answer
        self.event_idx, self.realtion_idx = 0, 0

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.count_er = Counter()
        self.dataset = self.get_data(path)
        if self.path == 'data/test.csv':
            self.Relation_tagging = self.get_data(f'save_model/Relation/{model_name}/tagging.csv')
            self.Evnet_tagging = self.get_data(f'save_model/Event/{model_name}/tagging.csv')
        self.create_dataset()  

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values

    def create_dataset(self):
        story_name = self.dataset[0][4]
        total = 0
        story_list = []
        for idx in tqdm(range(len(self.dataset))):
            current_story_name = self.dataset[idx][4]
            if current_story_name != story_name and story_list:
                count = self.create_input(story_list, story_name)
                total += count
                story_name = current_story_name
                story_list = []
            if self.dataset[idx][5].split(' - ')[0] in legal_tagging or self.dataset[idx][6].split(' - ')[0] in legal_tagging:
                story_list.append(idx)
        print(total)
        return          

    def create_input(self, story_list, story_name):

        context = self.dataset[story_list[0]][1]
        text = create_prompt(self.model_name, 'question', context, 
                            question_type=self.dataset[story_list[0]][0], 
                            gen_Answer = self.gen_Answer
                            #  question_words=self.dataset[story_list[0]][2].split(" ")[0]
                            )

        re_sum = Counter()
        tagging_count = 0
        for idx in story_list:
            tagging_type = 'Event' if self.dataset[idx][5] else 'Relation'
            re_sum[tagging_type] += 1
            if tagging_type == 'Relation':
                tagging_count += 1
            tagging = self.dataset[idx][5] if self.dataset[idx][5] else self.dataset[idx][6]
            text += f"[{tagging_type}] {tagging}"

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
        
        self.count_er[f'Event_{re_sum["Event"]}_Relation_{re_sum["Relation"]}'] += 1

        # if tagging_count > 1 :
        #     Event = defaultdict(defaultdict)
        #     text = ""
        #     for idx in story_list:
        #         tagging_type = 'Event' if self.dataset[idx][5] else 'Relation'
        #         if tagging_type == 'Event':
        #             continue
        #         for i in range(7, len(self.dataset[idx])):
        #             if self.dataset[idx][i] != '':
        #                 left_parenthesis_index = self.dataset[idx][i].rfind('(')
        #                 if tagging_type == 'Relation' and i == 7:
        #                     text += " [Event1] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
        #                     Event[idx]['Event1'] = "".join(self.dataset[idx][i][:left_parenthesis_index])
        #                 elif tagging_type == 'Relation' and i == 8:
        #                     text += " [Event2] " + "".join(self.dataset[idx][i][:left_parenthesis_index])
        #                     Event[idx]['Event2'] = "".join(self.dataset[idx][i][:left_parenthesis_index])
            
            # print(text)
            # print(Event)
            
            # graph = defaultdict(list)
            # for idx in Event:
            #     relation_dfs(idx, [], Event, graph[idx], set())
            
            # for idx in graph:
                # print(idx, graph[idx])
                # for chain in graph[idx]:
                #     for idx2 in chain:
                #         print('Event1 : ', graph[idx2]['Event1'])
                #         print('Event2 : ', graph[idx2]['Event2'])
            # input()
        
        return 1 if tagging_count > 1 else 0

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]

@lru_cache(maxsize=None)
def cached_lcs(event1, event2):
    return longest_common_subsequence(event1, event2)

def relation_dfs(idx, chains, Relation_graph, graph, visited_chains): 
    if idx in chains:
        return
    chains.append(idx)

    orinigal_event1 = Relation_graph[idx]['Event1']
    orinigal_event2 = Relation_graph[idx]['Event2']
    for next_idx in Relation_graph:
        if next_idx in chains or set(Relation_graph[next_idx].keys()) != {'Event1', 'Event2'}:
            continue
        next_event1 = Relation_graph[next_idx]['Event1']
        next_event2 = Relation_graph[next_idx]['Event2']
        if (cached_lcs(orinigal_event1, next_event1) > 0.8 or 
            cached_lcs(orinigal_event1, next_event2) > 0.8 or 
            cached_lcs(orinigal_event2, next_event1) > 0.8 or 
            cached_lcs(orinigal_event2, next_event2) > 0.8 ):
            relation_dfs(next_idx, chains, Relation_graph, graph, visited_chains)
    
    chain_tuple = tuple(chains)
    if chain_tuple not in visited_chains:
        graph.append(deepcopy(chains))
        visited_chains.add(chain_tuple)
    chains.pop()
    return 

def longest_common_subsequence(event1, event2):
    event1_len = len(event1)
    event2_len = len(event2)
    longest = 0
    lcs_table = [[0] * (event2_len + 1) for _ in range(event1_len + 1)]

    for i in range(1, event1_len + 1):
        for j in range(1, event2_len + 1):
            if event1[i - 1] == event2[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                longest = max(longest, lcs_table[i][j])
            else:
                lcs_table[i][j] = 0
    return longest / (event1_len + event2_len - longest)

if __name__ == '__main__':
    gen_answer = False,
    model_name = "flant5"
    Generation = 'question'
    path_save_model = 'save_model'
    path_save_model = checkdir(path_save_model, Generation, model_name, gen_answer)
    model, tokenizer = create_model(model_name, Generation, test_mode = False, path_save_model = path_save_model)
    train_data = Question_generation_Datasets('data/train.csv', model_name, tokenizer, max_len = 768, gen_answer = gen_answer)
    print(train_data.count_er)
    for key, val in sorted(train_data.count_er.items(), key=lambda item: item[1], reverse=True):
        print(key, val, round(val / sum(train_data.count_er.values()), 2))
