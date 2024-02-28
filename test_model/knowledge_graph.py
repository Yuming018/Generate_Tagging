import sys 
sys.path.append("..") 
from helper import enconder

from tqdm import tqdm
from collections import defaultdict
from generation import create_tagging, create_question
from sentence_transformers import SentenceTransformer, util
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
    text = f'[Type] {context_type} [Context] {core_context} [END]'
    
    input_ids, event_tagging_ids, event_tagging = generate_tagging('../save_model/Event/tagging/', text, tokenizer, model, device)
    Event_tagging_id = torch.cat((input_ids, event_tagging_ids), dim=0)
    event_question, e_score = generate_question('../save_model/Event/question/', Event_tagging_id, target, tokenizer, model)
    
    input_ids, relation_tagging_ids, relation_tagging = generate_tagging('../save_model/Relation/tagging/', text, tokenizer, model, device)
    Relation_tagging_id = torch.cat((input_ids, relation_tagging_ids), dim=0)
    relation_question, r_score = generate_question('../save_model/Relation/question/', Relation_tagging_id, target, tokenizer, model)
    
    if e_score >= r_score:
        tagging = event_tagging
        # question = event_question
        # score = e_score
    else:
        tagging = relation_tagging
        # question = relation_question
        # score = r_score

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
    print('Event')
    for idx, line in enumerate(new_lines):
        if line == core_context:
            continue
        if line != '':
            text = "Please utilize the provided context, question types, and type definitions to generate key information for this context, along with corresponding types ."
            for key, definition in Relation_definition_2.items():
                text += f'[{key}] {definition} ' 
            text += f'[Type] {context_type} [Context] {line} [END]'
            _, _, tagging = generate_tagging('../save_model/Event/tagging/', text, tokenizer, model, device)
            
            Event_graph[idx]['text'] = line
            for tag in tagging.split('[')[1:-1]:
                key, entity = tag.split(']')
                Event_graph[idx][key] = entity
    
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

            text = "Please utilize the provided context, question types, and type definitions to generate key information for this context, along with corresponding types ."
            for key, definition in Relation_definition_2.items():
                text += f'[{key}] {definition} ' 
            text += f'[Type] {context_type} [Context] {context} [END]'
            _, _, tagging = generate_tagging('../save_model/Relation/tagging/', text, tokenizer, model, device)
            
            Relation_graph[idx]['text'] = context
            for tag in tagging.split('[')[1:-1]:
                key, entity = tag.split(']')
                Relation_graph[idx][key] = entity
    
    # for idx in Relation_graph:
    #     print(idx)
    #     for key in Relation_graph[idx]:
    #         print(key, ":", Relation_graph[idx][key])
    # input()

    input_ids, match = connect_knowledge_graph(context, core_graph, Event_graph, Relation_graph, tokenizer, device)
    if match:
        event_question, e_score = generate_question('../save_model/Event/question/', input_ids, target, tokenizer, model)
        relation_question, r_score = generate_question('../save_model/Relation/question/', input_ids, target, tokenizer, model)
        
    # input()
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
    return input_ids, tagging_ids, tagging

def generate_question(path_save_model, input_ids, target, tokenizer, model):
    """
    Input : Event Knowledge graph
    Output : 生成出來的問題，以及跟 Fairytale QA target 計算過相似性的分數
    """
    question_ids = create_question(path_save_model = path_save_model, model = model, input_ids = input_ids)
    question = tokenizer.decode(question_ids, skip_special_tokens=True)
    score = eval(question, target)
    if 'Event' in path_save_model:
        print('Event_question: ', question, score)
    elif 'Relation' in path_save_model:
        print('Relation_question: ', question, score)
    return question, score


def connect_knowledge_graph(context, core_graph, Event_graph, Relation_graph, tokenizer, device):
    """
    Input : 重點段落 Knowledge graph, 每句話的 Knowledge graph
    Output : 返回替換節點後的 Knowledge graph
    此 function 利用每一句句子生成 Knowledge graph 判斷是否跟 core_graph 有相同的節點
    如有相同的節點便替換，並生成有別於原始 Fairytale QA 的問題(更難或更簡單)
    """
    match = False
    match_key = []
    for key, entity in core_graph.items():
        if key not in ['Event', 'Relation'] and entity in Event_graph[key]:
            match = True
            match_key.append(key)

    text = '' 
    if match:
        text += f'[Context] {context} '
        for key, entity in core_graph.items(): 
            if key not in match_key:
                text += f'[{key}]{entity}'
        for key, entity in Event_graph.items(): 
            if key not in ['Event', 'Relation'] and key not in match_key:
                text += f'[{key}]{entity}'
        text += '[END]'
    
    input_ids = enconder(tokenizer, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)
    return input_ids, match


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