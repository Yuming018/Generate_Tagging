import sys 
sys.path.append("..") 
from model import create_model

from knowledge_graph import create_knowledge_graph
from dataloader import Dataset
from tqdm import tqdm
import argparse
import torch
import csv
from collections import defaultdict
from transformers import logging
logging.set_verbosity_error()

fairytale_qa_type = ['action', 'outcome resolution', 'causal relationship',
                 'prediction', 'setting', 'feeling', 'character']

def save_csv(record, path):
    row = ['Paragraph', 'context', 'prediction', 'reference', 'input_text',  'SentenceBert', 'Event graph', 'Relation graph']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record['context'])):
            writer.writerow([record['paragraph'][i], record['context'][i], record['predict'][i], record['reference'][i], record['input_text'][i], record['eval_score'][i], record['Event_graph'][i], record['Relation_graph'][i]])

def main(device = 'cpu',
         model_name = "Mt0",
         question_type = 'who'):
    
    model, tokenizer = create_model(model_name)
    model.to(device)
    dataset = Dataset(path = '../data/test.csv', tokenizer = tokenizer)
    
    model.eval()
    count = 0
    record = defaultdict(list)
    for context_type, paragraph, context, focus_context, target in tqdm(dataset):
        # if count == 4:
        question_set, score_set, text_set, Event_graph, Relation_graph = create_knowledge_graph(context_type, context, focus_context, target, tokenizer, device, model_name, question_type)
        for question, score, text in zip(question_set, score_set, text_set):
            record['context'].append(context)
            record['predict'].append(question)
            record['reference'].append(target)
            record['eval_score'].append(score)
            record['paragraph'].append(paragraph)
            record['Event_graph'].append(Event_graph)
            record['Relation_graph'].append(Relation_graph)
            record['input_text'].append(text)
        # save_csv(record, path = 'predict.csv')
        count += 1
        if count > 5:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m', type=str, choices=['Mt0', 'T5', 'Bart', 'roberta', 'gemma', 'flant5'], default='Mt0')
    parser.add_argument('--Question_type', '-qt', type=str, choices=['who', 'where', 'what', 'when', 'how', 'why'], default='who')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(device=device,
         model_name = args.Model,
         question_type = args.Question_type)