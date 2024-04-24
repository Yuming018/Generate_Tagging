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

def create_knowledge_graph(context_type, context, core_context, target, tokenizer, device, model_name, question_type):
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
    
    # Event_graph = {0: defaultdict(None, {'text': '" away  away ! " barked the yard - dog ', 'Event 1': ' Action - Action Verb ', 'Actor': ' the yard dog ', 'Trigger_Word': ' barked '}), 1: defaultdict(None, {'text': ' " i \'ll tell you ; they said i was a pretty little fellow once ', 'Event 1': ' Action - Direct Speech Act ', 'Msg (Direct)': ' i was a pretty little fellow once ', 'Speaker': ' they ', 'Trigger_Word': ' said '}), 2: defaultdict(None, {'text': ' then i used to lie in a velvet - covered chair ', 'Event 1': ' Action - Action Verb ', 'Actor': ' i ', 'Direct Object': ' a velvet covered chair ', 'Trigger_Word': ' used to lie in '}), 3: defaultdict(None, {'text': " up at the master 's house  and sit in the mistress 's lap ", 'Event 1': ' Action - Action Verb ', 'Actor': " the mistress's lap ", 'Trigger_Word': ' sit in '}), 4: defaultdict(None, {'text': ' they used to kiss my nose  and wipe my paws with an embroidered handkerchief ', 'Event 1': ' Action - Action Verb ', 'Actor': ' they ', 'Direct Object': ' my nose and wipe my paws with an embroidered handkerchief ', 'Trigger_Word': ' used to kiss '}), 5: defaultdict(None, {'text': " and i was called ' ami  dear ami ", 'Event 1': ' State - Characteristic ', 'Entity': ' i ', 'Key': ' ami dear ami ', 'Trigger_Word': ' was called '}), 6: defaultdict(None, {'text': " sweet ami  ' but after a while i grew too big for them ", 'Event 1': ' Action - Action Verb ', 'Actor': ' sweet ami ', 'Direct Object': ' too big for them ', 'Trigger_Word': ' grew '}), 7: defaultdict(None, {'text': " and they sent me away to the housekeeper 's room ", 'Event 1': ' Action - Action Verb ', 'Actor': ' they ', 'Direct Object': ' me ', 'Place': " to the housekeeper's room ", 'Trigger_Word': ' sent '}), 8: defaultdict(None, {'text': ' so i came to live on the lower story ', 'Event 1': ' Action - Action Verb ', 'Actor': ' i ', 'Direct Object': ' the lower story ', 'Trigger_Word': ' came to live on '}), 9: defaultdict(None, {'text': ' you can look into the room from where you stand ', 'Event 1': ' Action - Action Verb ', 'Actor': ' you ', 'Direct Object': ' the room from where you stand ', 'Trigger_Word': ' can look into '}), 10: defaultdict(None, {'text': ' and see where i was master once  i was indeed master to the housekeeper ', 'Event 1': ' Action - Action Verb ', 'Actor': ' i ', 'Direct Object': ' to the housekeeper ', 'Trigger_Word': ' see '}), 11: defaultdict(None, {'text': ' it was certainly a smaller room than those up stairs ', 'Event 1': ' Action - Action Verb ', 'Actor': ' it ', 'Direct Object': ' a smaller room than those up stairs ', 'Trigger_Word': ' was '}), 12: defaultdict(None, {'text': ' but i was more comfortable  for i was not being continually taken hold of and pulled about by the children as i had been ', 'Event 1': ' State - Characteristic ', 'Entity': ' i ', 'Trigger_Word': ' was ', 'Value': ' more comfortable for i was not being continually taken hold of and pulled about by the children as i had been '}), 13: defaultdict(None, {'text': ' i received quite as good food  or even better ', 'Event 1': ' State - Characteristic ', 'Entity': ' i ', 'Trigger_Word': ' received ', 'Value': ' quite as good food or even better '}), 14: defaultdict(None, {'text': ' i had my own cushion  and there was a stove -- it is the finest thing in the world at this season of the year ', 'Event 1': ' Action - Action Verb ', 'Actor': ' i ', 'Direct Object': ' it is the finest thing in the world at this season of the year ', 'Trigger_Word': ' had '}), 15: defaultdict(None, {'text': ' i used to go under the stove  and lie down quite beneath it ', 'Event 1': ' Action - Action Verb ', 'Actor': ' i ', 'Direct Object': ' lie down quite beneath it ', 'Trigger_Word': ' used to '}), 16: defaultdict(None, {'text': ' ah  i still dream of that stove  away ', 'Event 1': ' Action - Action Verb ', 'Actor': ' i ', 'Direct Object': ' that stove away ', 'Trigger_Word': ' dream of '})}
    # Relation_graph = {0: defaultdict(None, {'text': '" away . away ! " barked the yard - dog . " i \'ll tell you ; they said i was a pretty little fellow once . then i used to lie in a velvet - covered chair . up at the master \'s house . and sit in the mistress \'s lap ', 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' i used to lie in a velvet - covered chair ', 'Event2': " up at the master's house "}), 1: defaultdict(None, {'text': ' away ! " barked the yard - dog . " i \'ll tell you ; they said i was a pretty little fellow once . then i used to lie in a velvet - covered chair . up at the master \'s house . and sit in the mistress \'s lap . they used to kiss my nose ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i used to lie in a velvet - covered chair ', 'Event2': " up at the master's house "}), 2: defaultdict(None, {'text': ' " i \'ll tell you ; they said i was a pretty little fellow once . then i used to lie in a velvet - covered chair . up at the master \'s house . and sit in the mistress \'s lap . they used to kiss my nose . and wipe my paws with an embroidered handkerchief ', 'Relation 1': ' Temporal - isBefore ', 'Event1': ' i used to lie in a velvet - covered chair ', 'Event2': " up at the master's house "}), 3: defaultdict(None, {'text': " then i used to lie in a velvet - covered chair . up at the master 's house . and sit in the mistress 's lap . they used to kiss my nose . and wipe my paws with an embroidered handkerchief . and i was called ' ami ", 'Relation 1': ' Temporal - isBefore ', 'Event1': ' they used to kiss my nose. and wipe my paws with an embroidered handkerchief ', 'Event2': " i was called'ami "}), 4: defaultdict(None, {'text': " up at the master 's house . and sit in the mistress 's lap . they used to kiss my nose . and wipe my paws with an embroidered handkerchief . and i was called ' ami . dear ami ", 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' they used to kiss my nose. and wipe my paws with an embroidered handkerchief ', 'Event2': " i was called'ami "}), 5: defaultdict(None, {'text': " and sit in the mistress 's lap . they used to kiss my nose . and wipe my paws with an embroidered handkerchief . and i was called ' ami . dear ami . sweet ami ", 'Relation 1': ' Temporal - isBefore ', 'Event1': ' they used to kiss my nose ', 'Event2': ' wipe my paws with an embroidered handkerchief '}), 6: defaultdict(None, {'text': " they used to kiss my nose . and wipe my paws with an embroidered handkerchief . and i was called ' ami . dear ami . sweet ami . ' but after a while i grew too big for them ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' they used to kiss my nose. and wipe my paws with an embroidered handkerchief ', 'Event2': ' i grew too big for them '}), 7: defaultdict(None, {'text': " and wipe my paws with an embroidered handkerchief . and i was called ' ami . dear ami . sweet ami . ' but after a while i grew too big for them . and they sent me away to the housekeeper 's room ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' wipe my paws with an embroidered handkerchief ', 'Event2': ' i grew too big for them '}), 8: defaultdict(None, {'text': " and i was called ' ami . dear ami . sweet ami . ' but after a while i grew too big for them . and they sent me away to the housekeeper 's room . so i came to live on the lower story ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i grew too big for them ', 'Event2': " they sent me away to the housekeeper's room "}), 9: defaultdict(None, {'text': " dear ami . sweet ami . ' but after a while i grew too big for them . and they sent me away to the housekeeper 's room . so i came to live on the lower story . you can look into the room from where you stand ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i grew too big for them ', 'Event2': " they sent me away to the housekeeper's room "}), 10: defaultdict(None, {'text': " sweet ami . ' but after a while i grew too big for them . and they sent me away to the housekeeper 's room . so i came to live on the lower story . you can look into the room from where you stand . and see where i was master once ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i grew too big for them ', 'Event2': " they sent me away to the housekeeper's room "}), 11: defaultdict(None, {'text': " ' but after a while i grew too big for them . and they sent me away to the housekeeper 's room . so i came to live on the lower story . you can look into the room from where you stand . and see where i was master once . i was indeed master to the housekeeper ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i grew too big for them ', 'Event2': " they sent me away to the housekeeper's room "}), 12: defaultdict(None, {'text': " and they sent me away to the housekeeper 's room . so i came to live on the lower story . you can look into the room from where you stand . and see where i was master once . i was indeed master to the housekeeper . it was certainly a smaller room than those up stairs ", 'Relation 1': ' Causal Effect - X intent ', 'Event1': " they sent me away to the housekeeper's room ", 'Event2': ' i came to live on the lower story '}), 13: defaultdict(None, {'text': ' so i came to live on the lower story . you can look into the room from where you stand . and see where i was master once . i was indeed master to the housekeeper . it was certainly a smaller room than those up stairs . but i was more comfortable ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i came to live on the lower story ', 'Event2': ' you can look into the room from where you stand '}), 14: defaultdict(None, {'text': ' you can look into the room from where you stand . and see where i was master once . i was indeed master to the housekeeper . it was certainly a smaller room than those up stairs . but i was more comfortable . for i was not being continually taken hold of and pulled about by the children as i had been ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i was indeed master to the housekeeper ', 'Event2': ' it was certainly a smaller room than those up stairs '}), 15: defaultdict(None, {'text': ' and see where i was master once . i was indeed master to the housekeeper . it was certainly a smaller room than those up stairs . but i was more comfortable . for i was not being continually taken hold of and pulled about by the children as i had been . i received quite as good food ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' for i was not being continually taken hold of and pulled about by the children as i had been ', 'Event2': ' i received quite as good food '}), 16: defaultdict(None, {'text': ' i was indeed master to the housekeeper . it was certainly a smaller room than those up stairs . but i was more comfortable . for i was not being continually taken hold of and pulled about by the children as i had been . i received quite as good food . or even better ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' for i was not being continually taken hold of and pulled about by the children as i had been ', 'Event2': ' i received quite as good food '}), 17: defaultdict(None, {'text': ' it was certainly a smaller room than those up stairs . but i was more comfortable . for i was not being continually taken hold of and pulled about by the children as i had been . i received quite as good food . or even better . i had my own cushion ', 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' i was not being continually taken hold of and pulled about by the children as i had been ', 'Event2': ' i received quite as good food '}), 18: defaultdict(None, {'text': ' but i was more comfortable . for i was not being continually taken hold of and pulled about by the children as i had been . i received quite as good food . or even better . i had my own cushion . and there was a stove -- it is the finest thing in the world at this season of the year ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i received quite as good food. or even better ', 'Event2': ' i had my own cushion '}), 19: defaultdict(None, {'text': ' for i was not being continually taken hold of and pulled about by the children as i had been . i received quite as good food . or even better . i had my own cushion . and there was a stove -- it is the finest thing in the world at this season of the year . i used to go under the stove ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' for i was not being continually taken hold of and pulled about by the children as i had been ', 'Event2': ' i received quite as good food '}), 20: defaultdict(None, {'text': ' i received quite as good food . or even better . i had my own cushion . and there was a stove -- it is the finest thing in the world at this season of the year . i used to go under the stove . and lie down quite beneath it ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i received quite as good food ', 'Event2': ' i had my own cushion '}), 21: defaultdict(None, {'text': ' or even better . i had my own cushion . and there was a stove -- it is the finest thing in the world at this season of the year . i used to go under the stove . and lie down quite beneath it . ah ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' there was a stove ', 'Event2': ' it is the finest thing in the world at this season of year '}), 22: defaultdict(None, {'text': ' i had my own cushion . and there was a stove -- it is the finest thing in the world at this season of the year . i used to go under the stove . and lie down quite beneath it . ah . i still dream of that stove ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' there was a stove ', 'Event2': ' it is the finest thing in the world at this season of year '}), 23: defaultdict(None, {'text': ' and there was a stove -- it is the finest thing in the world at this season of the year . i used to go under the stove . and lie down quite beneath it . ah . i still dream of that stove . away ', 'Relation 1': ' Causal Effect - X intent ', 'Event1': ' i used to go under the stove ', 'Event2': ' and lie down quite beneath it '}), 24: defaultdict(None, {'text': ' i used to go under the stove . and lie down quite beneath it . ah . i still dream of that stove . away . away ! "', 'Relation 1': ' Temporal - isBefore ', 'Event1': ' i used to go under the stove ', 'Event2': ' and lie down quite beneath it '})}

    question_set, score_set, text_set, question_difficulty, question_5w1h = [], [], [], [], []
    coreference = corf_resolution(context)

    relation_question, relation_score, relation_text, relation_difficulty = connect_relation_graph(context, Relation_graph, tokenizer, device, target, model_name, question_type)
    for question, score, text, difficulty in zip(relation_question, relation_score, relation_text, relation_difficulty):
        if question not in question_set:
            question_set.append(question)
            score_set.append(score)
            text_set.append(text)
            question_difficulty.append(difficulty)
            question_5w1h.append(question_type)
    
    for question_type in question_keyword:
        event_question, event_score, event_text, event_difficulty = connect_event_graph(context, Event_graph, tokenizer, device, target, model_name, question_type, coreference, new_lines)
        for question, score, text, difficulty in zip(event_question, event_score, event_text, event_difficulty):
            if question not in question_set:
                question_set.append(question)
                score_set.append(score)
                text_set.append(text)
                question_difficulty.append(difficulty)
                question_5w1h.append(question_type)

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

def generate_question(path_save_model, model_name, text, target, tokenizer, device):
    input_ids = enconder(tokenizer, max_len=768, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    question_ids = create_question(path_save_model = path_save_model, model_name = model_name, input_ids = input_ids, device = device)
    question = tokenizer.decode(question_ids, skip_special_tokens=True)
    score = eval(question, target)

    print('Text : ', text)
    print('Question: ', question, '\n')
    # print('Target : ', target, '\n')
    return question, score

def connect_event_graph(context, Event_graph, tokenizer, device, target, model_name, question_type, coreference, new_lines):
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
                    event_question, e_score = generate_question(f'../save_model/question/{model_name}/', model_name, text, target, tokenizer, device)
                    event_question_set.append(event_question)
                    event_score.append(e_score)
                    event_text.append(text)
                    event_difficulty.append(corf_range)
            
    return event_question_set, event_score, event_text, event_difficulty

def connect_relation_graph(context, Relation_graph, tokenizer, device, target, model_name, question_type):
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
    
    # for event in event_flow:
    #     print(event)
    # input()

    relation_question_set, relation_score, relation_text, relation_difficulty = [], [], [], []
    for idx in range(len(event_flow)):
        for idx_2 in range(idx+2, len(event_flow)):
            # print(event_flow[idx], event_flow[idx_2])
            text = create_prompt(model_name, None, 'question', context)
            if event_flow[idx] not in event_flow[idx_2] and event_flow[idx_2] not in event_flow[idx] :
                text += f'[Relation1] Temporal - the same '
                text += f'[Event1] {event_flow[idx]} [Event2] why [Event3] {event_flow[idx_2]} [END]'
                relation_question, r_score = generate_question(f'../save_model/question/{model_name}/', model_name, text, target, tokenizer, device)
                relation_question_set.append(relation_question)
                relation_score.append(r_score)
                relation_text.append(text)
                relation_difficulty.append(idx_2 - idx)

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

    # print(subject)
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
    # relation_question, r_score = generate_question(f'../save_model/question/{args.Model}/', args.Model, text, target, tokenizer, device)

    context = "puck was careful not always to play his tricks in the same place , but visited one village after another , so that everyone trembled lest he should be the next victim . after a bit he grew tired of cowboys and shepherds , and wondered if there was no one else to give him some sport . at length he was told of a young couple who were going to the nearest town to buy all that they needed for setting up house . quite certain that they would forget something which they could not do without , puck waited patiently till they were jogging along in their cart on their return journey . he changed himself into a fly in order to overhear their conversation . for a long time it was very dull -- all about their wedding day next month , and who were to be invited . this led the bride to her wedding dress , and she gave a little scream . ' just think ! oh ! how could i be so stupid ! i have forgotten to buy the different coloured reels of cotton to match my clothes ! ' ' dear , dear ! ' exclaimed the young man . ' that is unlucky . did n't you tell me that the dressmaker was coming in to - morrow ? ' ' yes , i did , ' and then suddenly she gave another little scream , which had quite a different sound from the first . ' look ! look ! '"
    text = create_prompt(args.Model, None, 'question', context)
    text += "[Event 1]  State - Characteristic [Agent]  puck  [Agent]  he [Emotion]  how [Time]  after a bit [END]"
    target = " why did puck decide to play a trick on a couple ? "
    relation_question, r_score = generate_question(f'../save_model/question/{args.Model}/', args.Model, text, target, tokenizer, device)
