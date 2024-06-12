import sys 
sys.path.append("..") 
from helper import enconder, text_segmentation, create_prompt
from model import create_model
from coreference_resolution import corf_resolution

import re
import argparse
from tqdm import tqdm
from collections import defaultdict
from generation import create_tagging, create_question
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import torch
import itertools

question_keyword = {
    'who' : ['Speaker', 'Actor', 'Addressee', 'Agent', 'Entity'],
    'where' : ['Place'],
    'what' : ['Direct Object', 'Topic(indirect)', 'Msg(Direct)', 'key', 'value', 'Topic'],
    'when' : ['Time'],
    'how' : ['Emotion', 'Trigger_Word'],
    # 'why' : [],
}

event_type = [
    'Action - Action Verb', 'Action - Direct Speech Act', 'Action - Indirect Speech Act',
    'State - Emotion', 'State - Thought', 'State - Characteristic',
]

relation_type = [
    'Temporal - the same', 'Temporal - isAfter', 'Temporal - isBefore',
    'Causal Effect - X intent', 'Causal Effect - X need', 'Causal Effect - X attribute', 
    'Causal Effect - Effect on X', 'Causal Effect - X want', 'Causal Effect - X reaction',
    'Causal Effect - Other reaction', 'Causal Effect - Other want', 'Causal Effect - Effect on other',
]

def create_knowledge_graph(gen_answer, 
                           context_type, 
                           context, 
                           core_context, 
                           target, 
                           tokenizer, 
                           device, 
                           model_name, 
                           question_type, 
                           Event_count):
    """
    根據 Fairytale QA 的問題，判斷文章中的重點段落(core_context)為何
    並以 core_context 生成出 Knowledge graph ，命名為 core_graph
    重點段落 : 依據此段落便可以回答 Fairytale QA 的問題，主要為標記人員所標記的範圍
    """

    # event_text = create_prompt(model_name, 'Event', 'tagging', core_context)
    # event_tagging = generate_tagging(f'../save_model/Event/tagging/{model_name}/', model_name, event_text, tokenizer, device)

    relation_text = create_prompt(model_name, 'Relation', 'tagging', core_context)
    relation_tagging = generate_tagging(f'../save_model/Relation/tagging/{model_name}/', model_name, relation_text, tokenizer, device)

    core_graph = defaultdict(str)
    for tag in relation_tagging.split('[')[1:-1]:
        tag = tag.split(']')
        core_graph[tag[0]] = tag[1]

    lines = re.split(r'[,.]', context)
    new_lines = []
    remind_line = ""
    for line in lines:
        remind_line += line
        if len(remind_line.split(" ")) > 10:
            new_lines.append(remind_line)
            remind_line = ""
            
    if len(lines) > 10:
        base_len = (len(lines)//5 - 1)*5
        for i in range(len(lines)):
            if base_len == (len(lines) - i):
                range_para = i + 1
                break
        if range_para <= 3:
            range_para = 4
    else:
        range_para = 3

    Event_graph, Relation_graph = defaultdict(defaultdict), defaultdict(defaultdict)

    print('Event')
    for idx in tqdm(range(len(new_lines))):
        temp_context = new_lines[idx]
        text = create_prompt(model_name, 'Event', 'tagging', temp_context)
        tagging = generate_tagging(f'../save_model/Event/tagging/{model_name}/', model_name, text, tokenizer, device)
        Event_graph[idx]['text'] = temp_context
        # print(tagging)
        for tag in tagging.split('[')[1:-1]:
            key, entity = tag.split(']')
            Event_graph[idx][key] = entity
    
    print('\nRelation') 
    for idx in tqdm(range(len(lines) - range_para + 1)):
        temp_context = ".".join(lines[idx : idx + range_para])
        text = create_prompt(model_name, 'Relation', 'tagging', temp_context)
        tagging = generate_tagging(f'../save_model/Relation/tagging/{model_name}/', model_name, text, tokenizer, device)
        # print(temp_context)
        # print(tagging)
        Relation_graph[idx]['text'] = temp_context
        for tag in tagging.split('[')[1:-1]:
            key, entity = tag.split(']')
            Relation_graph[idx][key] = entity
    
    # Event_graph = {0: defaultdict(None, {'text': 'long  long ago there lived in kyoto a brave soldier named kintoki ', 'Event 1': ' Action - Action Verb ', 'Actor': ' kintoki ', 'Place': ' in kyoto ', 'Time': ' long long ago ', 'Trigger_Word': ' lived '}), 1: defaultdict(None, {'text': ' now he fell in love with a beautiful lady and married her ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Direct Object': ' her ', 'Trigger_Word': ' married '}), 2: defaultdict(None, {'text': ' not long after this  through the malice of some of his friends ', 'Event 1': ' Action - Action Verb ', 'Actor': ' some of his friends ', 'Trigger_Word': ' through the malice of '}), 3: defaultdict(None, {'text': ' he fell into disgrace at court and was dismissed ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Direct Object': ' dismissed ', 'Trigger_Word': ' was '}), 4: defaultdict(None, {'text': ' this misfortune so preyed upon his mind that he did not long survive his dismissal ', 'Event 1': ' Action - Action Verb ', 'Actor': ' this misfortune ', 'Direct Object': ' his mind ', 'Trigger_Word': ' did not long survive '}), 5: defaultdict(None, {'text': ' he died  leaving behind him his beautiful young wife to face the world alone ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Direct Object': ' the world alone ', 'Trigger_Word': ' face '}), 6: defaultdict(None, {'text': " fearing her husband 's enemies  she fled to the ashigara mountains as soon as her husband was dead ", 'Event 1': ' Action - Action Verb ', 'Actor': ' she ', 'Direct Object': ' the ashigara mountains ', 'Time': ' as soon as her husband was dead ', 'Trigger_Word': ' fled to '}), 7: defaultdict(None, {'text': ' there in the lonely forests where no one ever came except woodcutters ', 'Event 1': ' Action - Action Verb ', 'Actor': ' woodcutters ', 'Place': ' there in the lonely forests ', 'Trigger_Word': ' came '}), 8: defaultdict(None, {'text': ' a little boy was born to her  she called him kintaro or the golden boy ', 'Event 1': ' State - Characteristic ', 'Entity': ' she ', 'Key': ' kintaro or the golden boy ', 'Trigger_Word': ' called '}), 9: defaultdict(None, {'text': ' now the remarkable thing about this child was his great strength ', 'Event 1': ' State - Characteristic ', 'Entity': ' his great strength ', 'Trigger_Word': ' now '}), 10: defaultdict(None, {'text': ' and as he grew older he grew stronger and stronger ', 'Event 1': ' State - Characteristic ', 'Entity': ' he ', 'Trigger_Word': ' grew ', 'Value': ' stronger and stronger '}), 11: defaultdict(None, {'text': ' by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Direct Object': ' trees as quickly as the woodcutters ', 'Time': ' by the time he was eight years of age ', 'Trigger_Word': ' cut down '}), 12: defaultdict(None, {'text': ' then his mother gave him a large ax  and he used to go out in the forest and help the woodcutters ', 'Event 1': ' Action - Action Verb ', 'Actor': ' his mother ', 'Direct Object': ' he used to go out in the forest and help the woodcutters ', 'Trigger_Word': ' gave '}), 13: defaultdict(None, {'text': ' who called him " wonder - child  " and his mother the " old nurse of the mountains ', 'Event 1': ' Action - Action Verb ', 'Actor': ' who ', 'Direct Object': ' him ', 'Indirect Object': ' the old nurse of the mountains ', 'Trigger_Word': ' called '}), 14: defaultdict(None, {'text': ' " for they did not know her high rank ', 'Event 1': ' Action - Action Verb ', 'Actor': ' they ', 'Direct Object': ' her high rank ', 'Trigger_Word': ' did not know '}), 15: defaultdict(None, {'text': " another favorite pastime of kintaro 's was to smash up rocks and stones ", 'Event 1': ' Action - Action Verb ', 'Actor': " kintaro's ", 'Direct Object': ' rocks and stones ', 'Trigger_Word': ' smash up '})}
    # Relation_graph ={0: defaultdict(None, {'text': "long . long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this . through the malice of some of his friends . he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died . leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' through the malice of some of his friends ', 'Event2': ' he fell into disgrace at court and was dismissed '}), 1: defaultdict(None, {'text': " long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this . through the malice of some of his friends . he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died . leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead ", 'Relation 1': ' Temporal - isBefore ', 'Event1': ' he fell into disgrace at court and was dismissed ', 'Event2': ' this misfortune so preyed upon his mind that he did not long survive his dismissal '}), 2: defaultdict(None, {'text': " now he fell in love with a beautiful lady and married her . not long after this . through the malice of some of his friends . he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died . leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' through the malice of some of his friends ', 'Event2': ' he fell into disgrace at court and was dismissed '}), 3: defaultdict(None, {'text': " not long after this . through the malice of some of his friends . he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died . leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters . a little boy was born to her ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' through the malice of some of his friends ', 'Event2': ' he fell into disgrace at court and was dismissed '}), 4: defaultdict(None, {'text': " through the malice of some of his friends . he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died . leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters . a little boy was born to her . she called him kintaro or the golden boy ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' through the malice of some of his friends ', 'Event2': ' he fell into disgrace at court and was dismissed '}), 5: defaultdict(None, {'text': " he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died . leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters . a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength ", 'Relation 1': ' Temporal - isBefore ', 'Event1': ' he fell into disgrace at court and was dismissed ', 'Event2': ' this misfortune so prey upon his mind that he did not long survive his dismissal '}), 6: defaultdict(None, {'text': " this misfortune so preyed upon his mind that he did not long survive his dismissal . he died . leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters . a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' this misfortune so preyed upon his mind ', 'Event2': ' he did not long survive his dismissal '}), 7: defaultdict(None, {'text': " he died . leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters . a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' he died ', 'Event2': ' leaving behind him his beautiful young wife to face the world alone '}), 8: defaultdict(None, {'text': " leaving behind him his beautiful young wife to face the world alone . fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters . a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' leaving behind him his beautiful young wife to face the world alone ', 'Event2': " fearing her husband's enemies "}), 9: defaultdict(None, {'text': " fearing her husband 's enemies . she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters . a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax . and he used to go out in the forest and help the woodcutters ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': " fearing her husband's enemies ", 'Event2': ' she fled to the ashigara mountains '}), 10: defaultdict(None, {'text': ' she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters . a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax . and he used to go out in the forest and help the woodcutters . who called him " wonder - child ', 'Relation 1': ' Temporal - isBefore ', 'Event1': ' she fled to the ashigara mountains as soon as her husband was dead ', 'Event2': ' there in the lonely forests where no one ever came except woodcutters '}), 11: defaultdict(None, {'text': ' there in the lonely forests where no one ever came except woodcutters . a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax . and he used to go out in the forest and help the woodcutters . who called him " wonder - child . " and his mother the " old nurse of the mountains ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' his great strength ', 'Event2': ' he grew older '}), 12: defaultdict(None, {'text': ' a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax . and he used to go out in the forest and help the woodcutters . who called him " wonder - child . " and his mother the " old nurse of the mountains . " for they did not know her high rank ', 'Relation 1': ' Temporal - isBefore ', 'Event1': ' his mother gave him a large ax ', 'Event2': ' he used to go out in the forest and help the woodcutters '}), 13: defaultdict(None, {'text': ' she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax . and he used to go out in the forest and help the woodcutters . who called him " wonder - child . " and his mother the " old nurse of the mountains . " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' his mother gave him a large ax ', 'Event2': ' he used to go out in the forest and help the woodcutters '}), 14: defaultdict(None, {'text': ' now the remarkable thing about this child was his great strength . and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax . and he used to go out in the forest and help the woodcutters . who called him " wonder - child . " and his mother the " old nurse of the mountains . " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Relation 1': ' Temporal - isBefore ', 'Event1': ' his mother gave him a large ax ', 'Event2': ' he used to go out in the forest and help the woodcutters '})}

    question_set, score_set, text_set, question_difficulty, question_5w1h = [], [], [], [], []
    coreference = corf_resolution(context)

    # Relation 
    relation_question, relation_score, relation_text, relation_difficulty = connect_relation_graph(gen_answer, context, Relation_graph, tokenizer, device, target, model_name, question_type, Event_count)
    for question, score, text, difficulty in zip(relation_question, relation_score, relation_text, relation_difficulty):
        if question not in question_set:
            question_set.append(question)
            score_set.append(score)
            text_set.append(text)
            question_difficulty.append(difficulty)
            question_5w1h.append('why')
    
    # #Event
    # for question_type in question_keyword:
    #     event_question, event_score, event_text, event_difficulty = connect_event_graph(gen_answer, context, Event_graph, tokenizer, device, target, model_name, question_type, coreference, new_lines)
    #     for question, score, text, difficulty in zip(event_question, event_score, event_text, event_difficulty):
    #         if question not in question_set:
    #             question_set.append(question)
    #             score_set.append(score)
    #             text_set.append(text)
    #             question_difficulty.append(difficulty)
    #             question_5w1h.append(question_type)

    return question_set, score_set, text_set, question_difficulty, question_5w1h, Event_graph, Relation_graph

def generate_tagging(path_save_model, model_name, text, tokenizer, device):
    input_ids = enconder(tokenizer, max_len=512, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    tagging_ids = create_tagging(path_save_model = path_save_model, model_name = model_name, input_ids = input_ids, device = device)
    tagging = tokenizer.decode(tagging_ids, skip_special_tokens=True)
    
    # print('\nText: ', text)
    # if 'Event' in path_save_model:
    #     print('Event_tagging: ', tagging)
    # elif 'Relation' in path_save_model:
    #     print('Relation_tagging: ', tagging)
    return tagging

def generate_question(model_name, gen_answer, text, target, tokenizer, device):
    if gen_answer:
        path_save_model = f'../save_model/QA_pair/{model_name}/'
    elif not gen_answer:
        path_save_model = f'../save_model/question/{model_name}/'
    input_ids = enconder(tokenizer, max_len=768, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    question_ids = create_question(path_save_model = path_save_model, model_name = model_name, input_ids = input_ids, device = device)
    question = tokenizer.decode(question_ids, skip_special_tokens=True)
    score = eval(question, target)

    # print('Text : ', text)
    # print('Question: ', question, '\n')
    # print('Target : ', target, '\n')
    return question, score

def connect_event_graph(gen_answer, context, Event_graph, tokenizer, device, target, model_name, question_type, coreference, new_lines):
    # event_dict = defaultdict(list)
    # for idx in Event_graph:
    #     temp = []
    #     if len(Event_graph[idx]['text'].split(' ')) > 3:
    #         for key, entity in Event_graph[idx].items():
    #             if key != 'text' and 'Event' not in key:
    #                 temp.append(f'[{key}] {entity}')
    #         for key, entity in Event_graph[idx].items():
    #             if key != 'text' and 'Event' not in key and temp not in event_dict[entity]:
    #                 event_dict[entity].append(temp)
    
    event_question_set, event_score, event_text, event_difficulty = [], [], [], []
    for idx in Event_graph:
        if len(Event_graph[idx]['text'].split(' ')) > 3:
            base_text = create_prompt(model_name, None, 'question', context)
            for type in event_type:
                text = base_text + f'[Event 1] {type} '
                have_arg = False
                corf_range = 0
                for key, entity in Event_graph[idx].items():
                    if key != 'text' and 'Event' not in key:
                        flag, corf_flag = False, False
                        for argument in question_keyword[question_type]:
                            if question_type == 'who' and key in question_keyword['who'] and not corf_flag:
                                # print(" ".join(entity.split(' ')[1:-1]), coreference)
                                corf, difficulty = check_corf(" ".join(entity.split(' ')[1:-1]), coreference, new_lines, idx)
                                text += corf
                                if corf != "":
                                    corf_flag = True
                                    corf_range = difficulty
                            if argument == key:
                                text += f'[{key}] {question_type} '
                                flag = True
                                have_arg = True
                        if not flag:
                            text += f'[{key}] {entity}'
                text += '[END]'
                if have_arg :
                    event_question, e_score = generate_question(model_name, gen_answer, text, target, tokenizer, device)
                    event_question_set.append(event_question)
                    event_score.append(e_score)
                    event_text.append(text)
                    event_difficulty.append(corf_range)
            
    return event_question_set, event_score, event_text, event_difficulty

def connect_relation_graph(gen_answer, context, Relation_graph, tokenizer, device, target, model_name, question_type, Event_count):
    relation_dict = defaultdict(list)
    for idx in Relation_graph:
        if [Relation_graph[idx]['Relation 1'], Relation_graph[idx]['Event2']] not in relation_dict[Relation_graph[idx]['Event1']]:
            relation_dict[Relation_graph[idx]['Event1']].append([Relation_graph[idx]['Relation 1'], Relation_graph[idx]['Event2']])
        if [Relation_graph[idx]['Relation 1'],Relation_graph[idx]['Event1']] not in relation_dict[Relation_graph[idx]['Event2']]:
            relation_dict[Relation_graph[idx]['Event2']].append([Relation_graph[idx]['Relation 1'],Relation_graph[idx]['Event1']])

    graph = defaultdict(list)
    for text in relation_dict:
        relation_dfs(text, [], relation_dict, graph[text])
    # relation_dfs(core_graph['Event1'], [], relation_dict, graph['Event1'])
    # relation_dfs(core_graph['Event2'], [], relation_dict, graph['Event2'])

    event_flow = []
    max_len = 0
    for text in relation_dict:
        for flow in graph[text]:
            if len(flow) > max_len:
                max_len = len(flow)
                event_flow = flow
    
    lines = re.split(r'[,.]', context)
    relation_question_set, relation_score, relation_text, relation_difficulty = [], [], [], []
    event_combinations = list(itertools.combinations(event_flow, Event_count))
    
    for combination in event_combinations:
        for type in relation_type:
            min_event_idx, max_event_idx = 1000, 0
            text = create_prompt(model_name, None, 'question', context)
            text += f'[Relation1] {type} [Event 1] why '
            for i, event in enumerate(combination):
                text += f'[Event{i+2}] {event} '
                min_event_idx = min(check_event_idx(lines, event), min_event_idx)
                max_event_idx = max(check_event_idx(lines, event), max_event_idx)
            text += '[END]'

            relation_question, r_score = generate_question(model_name, gen_answer, text, target, tokenizer, device)
            relation_question_set.append(relation_question)
            relation_score.append(r_score)
            relation_text.append(text)
            relation_difficulty.append(max_event_idx - min_event_idx)
            
    # for idx in range(len(event_flow)):
    #     for idx_2 in range(idx+2, len(event_flow)):
    #         text = create_prompt(model_name, None, 'question', context)
    #         if event_flow[idx] not in event_flow[idx_2] and event_flow[idx_2] not in event_flow[idx] :
    #             for type in relation_type:
    #                 text += f'[Relation1] {type} '
    #                 text += f'[Event1] {event_flow[idx]} [Event2] why [Event3] {event_flow[idx_2]} [END]'

    #                 relation_question, r_score = generate_question(model_name, gen_answer, text, target, tokenizer, device)
    #                 relation_question_set.append(relation_question)
    #                 relation_score.append(r_score)
    #                 relation_text.append(text)
    #                 relation_difficulty.append(idx_2 - idx)

    return relation_question_set, relation_score, relation_text, relation_difficulty

def relation_dfs(text, temp, relation_dict, graph):
    if text in temp:
        if temp not in graph:
            graph.append(deepcopy(temp))
        return
    temp.append(text)
    for event in relation_dict:
        if text == event or temp[0] in event or event in temp[0]:
            for next_event in relation_dict[event]:
                relation_dfs(next_event[1], temp, relation_dict, graph)
    temp.remove(text)
    return 

def check_event_idx(lines, text):
    text = re.split(r'[,.]', text)[0]
    for idx, line in enumerate(lines):
        if text.replace(' ', '') in line.replace(' ', ''):
            return idx

def check_corf(subject, coreference, new_lines, idx):
    pronoun = ""
    flag = False
    for entity_name in coreference:
        for entity in coreference[entity_name]:
            if subject == entity:
                flag = True
                pronoun = entity_name
                break
        if flag :
            break

    corf_range = 1000
    for idx_2 in range(len(new_lines)):
        if flag and pronoun in new_lines[idx_2]:
            # print(idx, pronoun, new_lines[idx_2])
            corf_range = min(corf_range, abs(idx - idx_2))

    return (f'[Relation1] Coreference - Coref [Arg1] {subject} [Arg2] {pronoun} ', corf_range) if flag and subject != pronoun  else ("", 0)

def eval(pred, tar):
    """
    Input : 預測出來的問題，以及 Fairytale QA 的原始問題
    Output : 返回兩者的相似性分數
    """
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    query_embedding = model.encode(pred)
    passage_embedding = model.encode(tar)
    result = util.dot_score(query_embedding, passage_embedding)
    return round(result[0][0].item(), 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m', type=str, choices=['Mt0', 'T5', 'Bart', 'roberta', 'gemma', 'flant5'], default='Mt0')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = create_model(args.Model)

    # context = "' away , away ! ' barked the yard - dog . ' i 'll tell you ; they said i was a pretty little fellow once . then i used to lie in a velvet - covered chair , up at the master 's house , and sit in the mistress 's lap . they used to kiss my nose , and wipe my paws with an embroidered handkerchief , and i was called ' ami , dear ami , sweet ami . ' but after a while i grew too big for them , and they sent me away to the housekeeper 's room . so i came to live on the lower story . you can look into the room from where you stand , and see where i was master once . i was indeed master to the housekeeper . it was certainly a smaller room than those up stairs . but i was more comfortable , for i was not being continually taken hold of and pulled about by the children as i had been . i received quite as good food , or even better . i had my own cushion , and there was a stove -- it is the finest thing in the world at this season of the year . i used to go under the stove , and lie down quite beneath it . ah , i still dream of that stove . away , away ! '"
    # text = create_prompt(args.Model, None, 'question', context)
    # # text += "[Event 1] Action - Action Verb [Msg (Direct)]  what [Speaker]  they [Trigger_Word]  said [Actor]  they [Direct Object]  my nose and wipe my paws with an embroidered handkerchief [Trigger_Word]  used to kiss [END]"
    # # text += "[Event 1]  State - Characteristic [Msg (Direct)]  i was a pretty little fellow once [Speaker]  they [Trigger_Word]  said [Actor]  they [Direct Object]  what [Place]  to the housekeeper's room [Trigger_Word]  sent [END]"
    # text += "[Event 1] Action - Action Verb [Actor]   the yard dog  [Key]  what [Trigger_Word]  was called [END]"
    # target = "where did the yard-dog used to lie ?"
    # relation_question, r_score = generate_question(args.Model, text, target, tokenizer, device)

    context = "puck was careful not always to play his tricks in the same place , but visited one village after another , so that everyone trembled lest he should be the next victim . after a bit he grew tired of cowboys and shepherds , and wondered if there was no one else to give him some sport . at length he was told of a young couple who were going to the nearest town to buy all that they needed for setting up house . quite certain that they would forget something which they could not do without , puck waited patiently till they were jogging along in their cart on their return journey . he changed himself into a fly in order to overhear their conversation . for a long time it was very dull -- all about their wedding day next month , and who were to be invited . this led the bride to her wedding dress , and she gave a little scream . ' just think ! oh ! how could i be so stupid ! i have forgotten to buy the different coloured reels of cotton to match my clothes ! ' ' dear , dear ! ' exclaimed the young man . ' that is unlucky . did n't you tell me that the dressmaker was coming in to - morrow ? ' ' yes , i did , ' and then suddenly she gave another little scream , which had quite a different sound from the first . ' look ! look ! '"
    text = create_prompt(args.Model, None, 'question', context)
    text += "[Event 1]  State - Characteristic [Agent]  puck  [Agent]  he [Emotion]  how [Time]  after a bit [END]"
    target = " why did puck decide to play a trick on a couple ? "
    relation_question, r_score = generate_question(args.Model, text, target, tokenizer, device)
