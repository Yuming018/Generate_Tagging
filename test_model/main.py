import sys 
sys.path.append("..") 
from model import create_model

from transformers import AutoTokenizer
from knowledge_graph import create_knowledge_graph
from dataloader import Dataset
from tqdm import tqdm
import argparse
import torch
import csv

from transformers import logging
logging.set_verbosity_error()

question_type = ['action', 'outcome resolution', 'causal relationship',
                 'prediction', 'setting', 'feeling', 'character']

def save_csv(context, predict, reference, score, path):
    row = ['ID', 'context', 'prediction', 'reference', 'SentenceBert']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(context)):
            writer.writerow([i, context[i], predict[i], reference[i], score[i]])

def main(path_save_model = '../save_model/Event/tagging/',
        device = 'cpu',
        event_or_relation = 'Event',
):
    model_name = "bigscience/mt0-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_model(model_name).to(device)
    
    dataset = Dataset(path = '../data/test.csv', model_name = model_name)
    
    model.eval()
    context_, predict, reference, score_ = [], [], [], []
    i = 0
    for context_type, context, focus_context, target in tqdm(dataset):
        if i == 2:
            question, score = create_knowledge_graph(context_type, context, focus_context, target, tokenizer, model, device)
            
            context_.append(focus_context)
            predict.append(question)
            reference.append(target)
            score_.append(score)
        i += 1
    
    print(round(sum(score_)/len(score_), 2))
    save_csv(context_, predict, reference, score_, path = 'predict.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(device=device)