import sys 
sys.path.append("..") 
from helper import enconder

from generation import generate_event_tagging, generate_relation_tagging
import torch

def create_knowledge_graph(context_type, context, tokenizer, model, device):
    lines = context.split('.')
    for line in lines:
        text = f'[Type] {context_type} [Context] {line} [END]'
        input_ids = enconder(tokenizer, text=text)
        input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

        event_tagging_ids = generate_event_tagging(path_save_model = '../save_model/Event/tagging/', model = model, input_ids = input_ids, relation_tag = False)
        event_tagging = tokenizer.decode(event_tagging_ids, skip_special_tokens=True)
        relation_tagging_ids = generate_relation_tagging(path_save_model = '../save_model/Relation/tagging/', model = model, input_ids = input_ids, relation_tag = True)
        relation_tagging = tokenizer.decode(relation_tagging_ids, skip_special_tokens=True)
        print('text: ', text)
        print('event_tagging: ', event_tagging)
        print('relation_tagging: ', relation_tagging)
        input()
    
if __name__ == '__main__':
    pass