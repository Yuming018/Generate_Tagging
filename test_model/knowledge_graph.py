import sys 
sys.path.append("..") 
from helper import enconder, text_segmentation

from tqdm import tqdm
from collections import defaultdict
from generation import create_tagging, create_question
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import torch

Relation_definition_2 = {
    'If-Event-Then-Mental-State' : "Contains 'X intent', 'X reaction' and 'Other reaction'. Define three relations relating to the mental pre- and post-conditions of an event. ",
    'If-Event-Then-Event' : "Contains 'X need', 'Effect on X', 'X want', 'Other want', 'Effect on other', 'isBefore', 'the same' and 'isAfter'. Define five relations relating to events that constitute probable pre- and postconditions of a given event. Those relations describe events likely required to precede an event, as well as those likely to follow.",
    'If-Event-Then-Persona' : "Contains 'X attribute'. In addition to pre- and postconditions, Define a stative relation that describes how the subject of an event is described or perceived.",
}

def create_knowledge_graph(context_type, context, core_context, target, tokenizer, model, device):
    """
    根據 Fairytale QA 的問題，判斷文章中的重點段落(core_context)為何
    並以 core_context 生成出 Knowledge graph ，命名為 core_graph
    重點段落 : 依據此段落便可以回答 Fairytale QA 的問題，主要為標記人員所標記的範圍
    """
    text = f"Please utilize the provided context to generate 1 Event key information for this context, along with corresponding types .[Context] {core_context} [END]"
    event_tagging = generate_tagging('../save_model/Event/tagging/', text, tokenizer, model, device)
    
    # text = f"Please utilize the provided context and Event key information to generate question for this context [Context] {context} "     
    Event_text = text[:-5] + event_tagging
    event_question, e_score = generate_question('../save_model/question/', Event_text, target, tokenizer, model, device)

    text = f"Please utilize the provided context to generate 1 Relation key information for this context, along with corresponding types .[Context] {core_context} [END]"    
    relation_tagging = generate_tagging('../save_model/Relation/tagging/', text, tokenizer, model, device)
    
    text = f"Please utilize the provided context and Relation key information to generate question for this context [Context] {context} "    
    Relation_text = text + relation_tagging
    relation_question, r_score = generate_question('../save_model/question/', Relation_text, target, tokenizer, model, device)

    # if e_score >= r_score:
    #     tagging = event_tagging
    # else:
    #     tagging = relation_tagging
    tagging = relation_tagging

    core_graph = defaultdict(str)
    for tag in tagging.split('[')[1:-1]:
        tag = tag.split(']')
        core_graph[tag[0]] = tag[1]

    """
    以 core_context 為基準，以句號(.)當分隔點，切成一句一句的句子，並用每句句子生成 Knowledge graph
    """
    lines = context.split(core_context)
    before, after = lines[0], lines[1]
    before_lines = before.split('.')
    after_lines = after.split('.')
    core_context = before_lines[-1] + core_context + after_lines[0]


    new_lines = before_lines[:-1] + [core_context] + after_lines[1:]
    Event_graph = defaultdict(defaultdict)
    # print('Event')
    # for idx, line in enumerate(new_lines):
    #     if line == core_context:
    #         continue
    #     if line != '':
    #         text = f"Please utilize the provided context to generate 1 Event key information for this context, along with corresponding types .[Context] {line} [END]"
    #         tagging = generate_tagging('../save_model/Event/tagging/', text, tokenizer, model, device)
            
    #         Event_graph[idx]['text'] = line
    #         for tag in tagging.split('[')[1:-1]:
    #             key, entity = tag.split(']')
    #             Event_graph[idx][key] = entity
    
    # for idx in Event_graph:
    #     print(idx)
    #     for key in Event_graph[idx]:
    #         print(key, ":", Event_graph[idx][key])

    print('\nRelation') 
    Relation_graph = defaultdict(defaultdict)
    for idx, line in enumerate(new_lines):
        if line == core_context:
            continue
        if line != '':

            if idx <= new_lines.index(core_context):
                context = ".".join(new_lines[idx : new_lines.index(core_context)+1])
            else:
                context = ".".join(new_lines[new_lines.index(core_context) : idx+1])

            text = f"Please utilize the provided context to generate 1 Relation key information for this context, along with corresponding types .[Context] {context} [END]"
            tagging = generate_tagging('../save_model/Relation/tagging/', text, tokenizer, model, device)
            
            Relation_graph[idx]['text'] = context
            for tag in tagging.split('[')[1:-1]:
                key, entity = tag.split(']')
                Relation_graph[idx][key] = entity
    
    # for idx in Relation_graph:
    #     print(idx)
    #     for key in Relation_graph[idx]:
    #         print(key, ":", Relation_graph[idx][key])

    text, match = connect_knowledge_graph(context, core_graph, Event_graph, Relation_graph, tokenizer, device, target, model)
    if match:
        event_question, e_score = generate_question('../save_model/question/', text, target, tokenizer, model, device)
        relation_question, r_score = generate_question('../save_model/question/', text, target, tokenizer, model, device)
        
    input()
    return "", 0

def generate_tagging(path_save_model, text, tokenizer, model, device):
    """
    Input : 文本
    Output : Event Knowledge graph, Relation Knowledge graph(字串的形式，還不是 dict)
    此 function 生成有關於文本的 Event Knowledge graph 以及 Relation Knowledge graph
    """
    input_ids = enconder(tokenizer, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    tagging_ids = create_tagging(path_save_model = path_save_model, model = model, input_ids = input_ids)
    tagging = tokenizer.decode(tagging_ids, skip_special_tokens=True)
    
    # print('\nText: ', text)
    # if 'Event' in path_save_model:
    #     print('Event_tagging: ', tagging)
    # elif 'Relation' in path_save_model:
    #     print('Relation_tagging: ', tagging)
    return tagging

def generate_question(path_save_model, text, target, tokenizer, model, device):
    """
    Input : Event Knowledge graph
    Output : 生成出來的問題，以及跟 Fairytale QA target 計算過相似性的分數
    """
    input_ids = enconder(tokenizer, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    question_ids = create_question(path_save_model = path_save_model, model = model, input_ids = input_ids)
    question = tokenizer.decode(question_ids, skip_special_tokens=True)
    score = eval(question, target)
    if 'Event' in path_save_model:
        print('Event_question: ', question, score)
    elif 'Relation' in path_save_model:
        print('Relation_question: ', question, score)
    return question, score

ans = []
def connect_knowledge_graph(context, core_graph, Event_graph, Relation_graph, tokenizer, device, target, model):
    """
    Input : 重點段落 Knowledge graph, 每句話的 Knowledge graph
    Output : 返回替換節點後的 Knowledge graph
    此 function 利用每一句句子生成 Knowledge graph 判斷是否跟 core_graph 有相同的節點
    如有相同的節點便替換，並生成有別於原始 Fairytale QA 的問題(更難或更簡單)
    """
    relation_dict = defaultdict(list)

    for idx in Relation_graph:
        if [Relation_graph[idx]['Relation 1'], Relation_graph[idx]['Event2']] not in relation_dict[Relation_graph[idx]['Event1']]:
            relation_dict[Relation_graph[idx]['Event1']].append([Relation_graph[idx]['Relation 1'], Relation_graph[idx]['Event2']])
        if [Relation_graph[idx]['Relation 1'],Relation_graph[idx]['Event1']] not in relation_dict[Relation_graph[idx]['Event2']]:
            relation_dict[Relation_graph[idx]['Event2']].append([Relation_graph[idx]['Relation 1'],Relation_graph[idx]['Event1']])

    dfs(core_graph['Event1'], [], relation_dict)
    # for text in ans:
    #     for event in text:
    #         print(event)
    #     print('\n')
    # dfs(core_graph['Event2'], [], relation_dict)
    
    print('Event 1', core_graph['Event1'])
    print('Event 2', core_graph['Event2'])
    
    for link in ans:
        for event in link:
            if event != core_graph['Event1'] and event != core_graph['Event2']:
                text = f"Please utilize the provided context and Relation key information to generate question for this context [Context] {context} "
                text += f'[Relation1] {core_graph["Relation1"]} '
                text += f'[Event1] {core_graph["Event2"]} '
                text += f'[Event2] {event} [END]'
                print(text)
                relation_question, r_score = generate_question('../save_model/Relation/question/', text, target, tokenizer, model, device)
                input()
    return text, not relation_dict

def dfs(text, temp, relation_dict):
    # print(text, temp)
    if text in temp:
        if temp not in ans:
            ans.append(deepcopy(temp))
        return
    temp.append(text)
    for event in relation_dict:
        if text == event or temp[0] in event or event in temp[0]:
            for next_event in relation_dict[event]:
                dfs(next_event[1], temp, relation_dict)
    temp.remove(text)
    return 

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
    pass