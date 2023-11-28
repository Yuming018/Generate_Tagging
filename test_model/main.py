import sys 
sys.path.append("..") 
from helper import checkdir, enconder
from model import create_model

from transformers import AutoTokenizer
from generation import generate_event_tagging, generate_relation_tagging, generate_question
from dataloader import Dataset
from tqdm import tqdm
import argparse
import torch
import csv

question_type = ['action', 'outcome resolution', 'causal relationship',
                 'prediction', 'setting', 'feeling', 'character']

def save_csv(context, tagging, question, path):
    row = ['ID', 'context', 'tagging', 'prediction']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(context)):
            writer.writerow([i, type[i], context[i], tagging[i], question[i]])

def main(path_save_model = '../save_model/Event/tagging/',
        device = 'cpu',
        relation_tag = False,
):
    model_name = "bigscience/mt0-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_model(model_name).to(device)
    
    dataset = Dataset(path = '../data/test.csv', model_name = model_name)
    
    model.eval()
    context_, question_, tagging_ = [], [], []
    for context, target in tqdm(dataset):
        input_ids = enconder(tokenizer, text=context)
        input_ids = torch.tensor(input_ids.get('input_ids')).to(device)
        
        event_tagging_ids = generate_event_tagging(path_save_model = '../save_model/Event/tagging/', model = model, input_ids = input_ids, relation_tag = False)
        event_tagging = tokenizer.decode(event_tagging_ids, skip_special_tokens=True)
        relation_tagging_ids = generate_relation_tagging(path_save_model = '../save_model/Relation/tagging/', model = model, input_ids = input_ids, relation_tag = True)
        relation_tagging = tokenizer.decode(relation_tagging_ids, skip_special_tokens=True)
        print(input_ids)
        print(event_tagging_ids)
        print('context: ', context)
        print('event_tagging: ', event_tagging)
        print('relation_tagging: ', relation_tagging)
        input()
        combined_tagging = torch.cat((input_ids, event_tagging_ids), dim=0)

        question_ids = generate_question(path_save_model, relation_tag, model, combined_tagging)
        question = tokenizer.batch_decode(question_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        context_.append(context)
        tagging_.append(event_tagging)
        question_.append(question)
        
    save_csv(context_, tagging_, question_, path = '.predict.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(device=device)