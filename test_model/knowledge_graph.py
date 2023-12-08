import sys 
sys.path.append("..") 
from helper import enconder

from collections import defaultdict
from generation import generate_event_relation_tagging, generate_question
from sentence_transformers import SentenceTransformer, util
import torch

def create_knowledge_graph(context_type, context, core_context, target, tokenizer, model, device):
    """
    根據 Fairytale QA 的問題，判斷文章中的重點段落(core_context)為何
    並以 core_context 生成出 Knowledge graph ，命名為 core_graph
    重點段落 : 依據此段落便可以回答 Fairytale QA 的問題，主要為標記人員所標記的範圍
    """
    text = f'[Type] {context_type} [Context] {core_context} [END]'
    input_ids, event_tagging_ids, event_tagging, relation_tagging_ids, relation_tagging = generate_tagging(text, tokenizer, model, device)
    Event_tagging_id = torch.cat((input_ids, event_tagging_ids), dim=0)
    event_question, e_score = generate_event_question(Event_tagging_id, target, tokenizer, model)
    Relation_tagging_id = torch.cat((input_ids, relation_tagging_ids), dim=0)
    relation_question, r_score = generate_relation_question(Relation_tagging_id, target, tokenizer, model)
    
    if e_score >= r_score:
        type = 'Event'
        tagging = event_tagging
        question = event_question
        score = e_score
    else:
        type = 'Relation'
        tagging = relation_tagging
        question = relation_question
        score = r_score

    core_graph = defaultdict(str)
    for tag in tagging.split('[')[1:-1]:
        tag = tag.split(']')
        core_graph[tag[0]] = tag[1]
    # print('\n', core_context) 
    # print(core_graph, '\n')
    # input()

    """
    以 core_context 為基準，以句號(.)當分隔點，切成一句一句的句子，並用每句句子生成 Knowledge graph
    """
    lines = context.split(core_context)
    before, after = lines[0], lines[1]
    before_lines = before.split('.')
    after_lines = after.split('.')
    core_context = before_lines[-1] + core_context + after_lines[0]
    
    new_lines = before_lines[:-1] + [core_context] + after_lines[1:]
    all_graph = defaultdict(defaultdict)
    for idx, line in enumerate(new_lines):
        if line == core_context:
            continue
        otrher_paragraph_graph = defaultdict(str)
        if line != '':
            text = f'[Type] {context_type} [Context] {line} [END]'
            input_ids, event_tagging_ids, event_tagging, relation_tagging_ids, relation_tagging = generate_tagging(text, tokenizer, model, device)
            
            if type == 'Event':
                line_tagging = event_tagging
            elif type == 'Relation':
                line_tagging = relation_tagging
            for tag in line_tagging.split('[')[1:-1]:
                key, entity = tag.split(']')
                otrher_paragraph_graph[key] = entity
                all_graph[idx][key] = entity
            
            if idx <= new_lines.index(core_context):
                context = ".".join(new_lines[idx : new_lines.index(core_context)+1])
            else:
                context = ".".join(new_lines[new_lines.index(core_context) : idx])
            
            input_ids, match = connect_knowledge_graph(context, core_graph, otrher_paragraph_graph, tokenizer, device)
            if match:
                event_question, e_score = generate_event_question(input_ids, target, tokenizer, model)
                relation_question, r_score = generate_relation_question(input_ids, target, tokenizer, model)

        input()
    return question, score

def generate_tagging(text, tokenizer, model, device):
    """
    Input : 文本
    Output : Event Knowledge graph, Relation Knowledge graph(字串的形式，還不是 dict)
    此 function 生成有關於文本的 Event Knowledge graph 以及 Relation Knowledge graph
    """
    input_ids = enconder(tokenizer, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    event_tagging_ids = generate_event_relation_tagging(path_save_model = '../save_model/Event/tagging/', model = model, input_ids = input_ids)
    event_tagging = tokenizer.decode(event_tagging_ids, skip_special_tokens=True)
    relation_tagging_ids = generate_event_relation_tagging(path_save_model = '../save_model/Relation/tagging/', model = model, input_ids = input_ids)
    relation_tagging = tokenizer.decode(relation_tagging_ids, skip_special_tokens=True)
    
    print('\nText: ', text)
    print('Event_tagging: ', event_tagging)
    print('Relation_tagging: ', relation_tagging, '\n')
    return input_ids, event_tagging_ids, event_tagging, relation_tagging_ids, relation_tagging

def generate_event_question(input_ids, target, tokenizer, model):
    """
    Input : Event Knowledge graph
    Output : 生成出來的問題，以及跟 Fairytale QA target 計算過相似性的分數
    """
    event_question_ids = generate_question(path_save_model = '../save_model/Event/question/', model = model, input_ids = input_ids)
    event_question = tokenizer.decode(event_question_ids, skip_special_tokens=True)
    e_score = eval(event_question, target)
    print('Event_question: ', event_question, e_score)
    return event_question, e_score

def generate_relation_question(input_ids, target, tokenizer, model):
    """
    Input : Relation Knowledge graph
    Output : 生成出來的問題，以及跟 Fairytale QA target 計算過相似性的分數
    """
    relation_question_ids = generate_question(path_save_model = '../save_model/Relation/question/', model = model, input_ids = input_ids)
    relation_question = tokenizer.decode(relation_question_ids, skip_special_tokens=True)
    r_score = eval(relation_question, target)
    print('Relation_question: ', relation_question, r_score)
    return relation_question, r_score

def connect_knowledge_graph(context, core_graph, other_graph, tokenizer, device):
    """
    Input : 重點段落 Knowledge graph, 每句話的 Knowledge graph
    Output : 返回替換節點後的 Knowledge graph
    此 function 利用每一句句子生成 Knowledge graph 判斷是否跟 core_graph 有相同的節點
    如有相同的節點便替換，並生成有別於原始 Fairytale QA 的問題(更難或更簡單)
    """
    match = False
    match_key = []
    for key, entity in core_graph.items():
        if key not in ['Event', 'Relation'] and entity in other_graph[key]:
            match = True
            match_key.append(key)
    
    # print(core_graph)
    # print(other_graph)
    # print(match_key)

    text = '' 
    if match:
        text += f'[Context] {context} '
        for key, entity in core_graph.items(): 
            if key not in match_key:
                text += f'[{key}]{entity}'
        for key, entity in other_graph.items(): 
            if key not in ['Event', 'Relation'] and key not in match_key:
                text += f'[{key}]{entity}'
        text += '[END]'

    # print(text)
    # input()
    
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