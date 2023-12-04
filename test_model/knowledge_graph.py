import sys 
sys.path.append("..") 
from helper import enconder

from collections import defaultdict
from generation import generate_event_relation_tagging, generate_question
from sentence_transformers import SentenceTransformer, util
import torch

def create_knowledge_graph(context_type, context, focus_context, target, tokenizer, model, device):
    """
    根據 Fairytale QA 的問題，判斷文章中的重點段落(focus_context)為何
    並以 focus_context 生成出 Knowledge graph ，命名為 focus_graph
    重點段落 : 依據此段落便可以回答 Fairytale QA 的問題，主要為標記人員所標記的範圍
    """
    text = f'[Type] {context_type} [Context] {focus_context} [END]'
    input_ids, event_tagging_ids, event_tagging, relation_tagging_ids, relation_tagging = generate_tagging(text, tokenizer, model, device)
    event_question, e_score, relation_question, r_score = focus_context_generate_question(input_ids, event_tagging_ids, relation_tagging_ids, target, tokenizer, model, device)
    
    if e_score > r_score:
        type = 'Event'
        tagging = event_tagging
        question = event_question
        score = e_score
    else:
        type = 'Relation'
        tagging = relation_tagging
        question = relation_question
        score = r_score

    focus_graph = defaultdict(str)
    for tag in tagging.split('[')[1:-1]:
        tag = tag.split(']')
        focus_graph[tag[0]] = tag[1]
    # print('\n', focus_context) 
    # print(focus_tagging, '\n')
    # input()

    """
    以 focus_context 為基準，以句號(.)當分隔點，切成一句一句的句子，並用每句句子生成 Knowledge graph
    """
    lines = context.split(focus_context)
    before, after = lines[0], lines[1]
    before_lines = before.split('.')
    after_lines = after.split('.')
    focus_context = before_lines[-1] + focus_context + after_lines[0]

    for idx, line in enumerate(before_lines[:-1] + after_lines[1:]):
        graph = defaultdict(str)
        if line != '':
            text = f'[Type] {context_type} [Context] {line} [END]'
            input_ids, event_tagging_ids, event_tagging, relation_tagging_ids, relation_tagging = generate_tagging(text, tokenizer, model, device)
            if type == 'Event':
                line_tagging = event_tagging
            elif type == 'Relation':
                line_tagging = relation_tagging
            for tag in line_tagging.split('[')[1:-1]:
                tag = tag.split(']')
                graph[idx][tag[0]] = tag[1]
            event_question, e_score, relation_question, r_score = connect_knowledge_graph_generate_question(focus_graph, graph, target, tokenizer, model, device)
        input()
    return question, score

def generate_tagging(text, tokenizer, model, device):
    """
    Input : 文本
    Output : Event Knowledge graph, Relation Knowledge graph(文本的形式，還不是 dict)
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

def focus_context_generate_question(input_ids, event_tagging_ids, relation_tagging_ids, target, tokenizer, model, device):
    """
    Input : 重點段落的文本(encoding過的), Event Knowledge graph, Relation Knowledge graph
    Output : 依據 Event Knowledge graph 生成出來的問題，或者依據 Relation Knowledge graph 所生成出來的問題，以及兩者各自跟 Fairytale QA target 計算過的分數
    """
    combined_tagging_id = torch.cat((input_ids, event_tagging_ids), dim=0)
    event_question_ids = generate_question(path_save_model = '../save_model/Event/question/', model = model, input_ids = combined_tagging_id)
    event_question = tokenizer.decode(event_question_ids, skip_special_tokens=True)
    combined_tagging_id = torch.cat((input_ids, relation_tagging_ids), dim=0)
    relation_question_ids = generate_question(path_save_model = '../save_model/Relation/question/', model = model, input_ids = combined_tagging_id)
    relation_question = tokenizer.decode(relation_question_ids, skip_special_tokens=True)
    
    e_score = eval(event_question, target)
    r_score = eval(relation_question, target)
    
    # print('Event_question: ', event_question, e_score)
    # print('Relation_question: ', relation_question, r_score)
    return event_question, e_score, relation_question, r_score

def connect_knowledge_graph_generate_question(focus_graph, other_graph, target, tokenizer, model, device):
    """
    Input : 重點段落 Knowledge graph, 每句話的 Knowledge graph
    Output : 返回有別於原始 Fairytale QA 的問題(更難或更簡單)
    此 function 利用每一句句子生成 Knowledge graph 判斷是否跟 focus_graph 有相同的節點
    如有相同的節點便替換，並生成有別於原始 Fairytale QA 的問題(更難或更簡單)
    """
    text = ''
    input_ids = enconder(tokenizer, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    event_question_ids = generate_question(path_save_model = '../save_model/Event/question/', model = model, input_ids = input_ids)
    event_question = tokenizer.decode(event_question_ids, skip_special_tokens=True)

    relation_question_ids = generate_question(path_save_model = '../save_model/Relation/question/', model = model, input_ids = input_ids)
    relation_question = tokenizer.decode(relation_question_ids, skip_special_tokens=True)

    e_score = eval(event_question, target)
    r_score = eval(relation_question, target)

    print('Event_question: ', event_question, e_score)
    print('Relation_question: ', relation_question, r_score)

    return event_question, e_score, relation_question, r_score

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