import sys 
sys.path.append("..") 
from helper import enconder

from collections import defaultdict
from generation import generate_event_relation_tagging, generate_question
from sentence_transformers import SentenceTransformer, util
import torch

def create_knowledge_graph(context_type, context, focus_context, target, tokenizer, model, device):
    text = f'[Type] {context_type} [Context] {focus_context} [END]'
    input_ids, event_tagging_ids, event_tagging, relation_tagging_ids, relation_tagging = generate_tagging(text, tokenizer, model, device)
    event_question, e_score, relation_question, r_score = comibe_tagging_generate_question(input_ids, event_tagging_ids, relation_tagging_ids, target, tokenizer, model, device)
    
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

    focus_tagging = defaultdict(str)
    for tag in tagging.split('[')[1:-1]:
        tag = tag.split(']')
        focus_tagging[tag[0]] = tag[1]
    print(focus_context, '\n')
    print(focus_tagging, '\n')
    input()

    lines = context.split(focus_context)
    before, after = lines[0], lines[1]
    before_lines = before.split('.')
    after_lines = after.split('.')

    event_graph = defaultdict(defaultdict)
    relation_graph = defaultdict(defaultdict)
    for idx, line in enumerate(before_lines + after_lines):
        text = f'[Type] {context_type} [Context] {line} [END]'

        input_ids, event_tagging_ids, event_tagging, relation_tagging_ids, relation_tagging = generate_tagging(text, tokenizer, model, device)
        print('text: ', text)
        print('Event_tagging: ', event_tagging)
        print('Relation_tagging: ', relation_tagging, '\n')
        for tag in event_tagging.split('[')[1:-1]:
            tag = tag.split(']')
            event_graph[idx][tag[0]] = tag[1]
        for tag in relation_tagging.split('[')[1:-1]:
            tag = tag.split(']')
            relation_graph[idx][tag[0]] = tag[1]
        input()
    print(event_graph)
    print(relation_graph)
    input()
    
    return question, score

def generate_tagging(text, tokenizer, model, device):
    input_ids = enconder(tokenizer, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    event_tagging_ids = generate_event_relation_tagging(path_save_model = '../save_model/Event/tagging/', model = model, input_ids = input_ids)
    event_tagging = tokenizer.decode(event_tagging_ids, skip_special_tokens=True)
    relation_tagging_ids = generate_event_relation_tagging(path_save_model = '../save_model/Relation/tagging/', model = model, input_ids = input_ids)
    relation_tagging = tokenizer.decode(relation_tagging_ids, skip_special_tokens=True)
    
    # print('text: ', text)
    # print('Event_tagging: ', event_tagging)
    # print('Relation_tagging: ', relation_tagging, '\n')
    return input_ids, event_tagging_ids, event_tagging, relation_tagging_ids, relation_tagging

def comibe_tagging_generate_question(input_ids, event_tagging_ids, relation_tagging_ids, target, tokenizer, model, device):
    text_1 = "[Type] action [Context] ali baba enters the cave himself , and takes some of the treasure home [END]"
    event_tagging_ids_1 = enconder(tokenizer, text=text_1)
    event_tagging_ids_1 = torch.tensor(event_tagging_ids_1.get('input_ids')).to(device)
    event_tagging_ids_1 = generate_event_relation_tagging(path_save_model = '../save_model/Event/tagging/', model = model, input_ids = event_tagging_ids_1)
    event_tagging_1 = tokenizer.decode(event_tagging_ids_1, skip_special_tokens=True)
    text_2 = "[Type] action [Context] the thieves are gone [END]"
    event_tagging_ids_2 = enconder(tokenizer, text=text_2)
    event_tagging_ids_2 = torch.tensor(event_tagging_ids_2.get('input_ids')).to(device)
    event_tagging_ids_2 = generate_event_relation_tagging(path_save_model = '../save_model/Event/tagging/', model = model, input_ids = event_tagging_ids_2)
    event_tagging_2 = tokenizer.decode(event_tagging_ids_2, skip_special_tokens=True)
    # print(event_tagging_1)
    # print(event_tagging_2)

    text = '[Type] action [Context] one day ali baba is at work collecting and cutting firewood in the forest , and he happens to overhear a group of forty thieves visiting their treasure store . the treasure is in a cave , the mouth of which is sealed by magic . it opens on the words " open , simsim " , and seals itself on the words " close , simsim " . when the thieves are gone , ali baba enters the cave himself , and takes some of the treasure home . [Event] Action - Action Verb [Actor] the thieves [Trigger_Word] are gone [Direct Object] some of the treasure home [Trigger_Word] takes [Direct Object] a group of forty thieves visiting their treasure store [Time] one day [Trigger_Word] overhear [END]'
    question_ids_1 = enconder(tokenizer, text=text)
    question_ids_1 = torch.tensor(question_ids_1.get('input_ids')).to(device)
    # combined_tagging = torch.cat((input_ids, event_tagging_ids), dim=0)
    event_question_ids = generate_question(path_save_model = '../save_model/Event/question/', model = model, input_ids = question_ids_1)
    event_question = tokenizer.decode(event_question_ids, skip_special_tokens=True)
    combined_tagging = torch.cat((input_ids, relation_tagging_ids), dim=0)
    relation_question_ids = generate_question(path_save_model = '../save_model/Relation/question/', model = model, input_ids = question_ids_1)
    relation_question = tokenizer.decode(relation_question_ids, skip_special_tokens=True)
    
    e_score = eval(event_question, target)
    r_score = eval(relation_question, target)
    
    print(event_question, e_score)
    print(relation_question, r_score)

    # combined_tagging_ids = torch.cat((input_ids, event_tagging_ids, relation_tagging_ids), dim=0)
    # combined_event = tokenizer.decode(combined_tagging, skip_special_tokens=True)
    # combined_event_question_ids = generate_question(path_save_model = '../save_model/Event/question/', model = model, input_ids = combined_tagging_ids)
    # combined_event_question = tokenizer.decode(combined_event_question_ids, skip_special_tokens=True)
    # combined_relation_question_ids = generate_question(path_save_model = '../save_model/Relation/question/', model = model, input_ids = combined_tagging_ids)
    # combined_relation_question = tokenizer.decode(combined_relation_question_ids, skip_special_tokens=True)
    
    # ce_score = eval(combined_event_question, target)
    # cr_score = eval(combined_relation_question, target)
    # print(combined_event)
    # print(combined_event_question, ce_score)
    # print(combined_relation_question, cr_score)
    
    return event_question, e_score, relation_question, r_score

def eval(pred, tar):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    query_embedding = model.encode(pred)
    passage_embedding = model.encode(tar)
    result = util.dot_score(query_embedding, passage_embedding)
    return round(result[0][0].item(), 2)

if __name__ == '__main__':
    pass