import sys 
sys.path.append("..") 
from model import create_model

from knowledge_graph import create_knowledge_graph
from dataloader import Dataset, Concat_dataset
from tqdm import tqdm
import argparse
import torch
import csv
import os
from helper import checkdir
from collections import defaultdict
from transformers import logging
logging.set_verbosity_error()

def save_csv(record, path):
    if not os.path.isdir('csv'):
        os.mkdir('csv')
    row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Extraction type', 'Question type', 'Question_difficulty', 'SentenceBert', 'Event graph', 'Relation graph']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record['context'])):
            writer.writerow([record['paragraph'][i], 
                             record['context'][i], 
                             record['predict'][i], 
                             record['reference'][i], 
                             record['input_text'][i], 
                             record['Extraction type'][i], 
                             record['gen_question_type'][i],
                             record['question_difficulty'][i], 
                             record['eval_score'][i], 
                             record['Event_graph'][i], 
                             record['Relation_graph'][i]])

def main(device = 'cpu',
         model_name = "Mt0",
         gen_answer = False,
         Event_count = 2):
    
    path_save_model = checkdir('../save_model', Generation = 'question', model_name = model_name, gen_answer = gen_answer)
    model, tokenizer = create_model(model_name, 'question', test_mode = True, path_save_model = path_save_model)
    model.to(device)
    # dataset = Dataset(path = '../data/test.csv')
    dataset = Concat_dataset(path = '../data/test.csv')

    model.eval()
    record = defaultdict(list)
    count = 0
    print('Dataset : ', len(dataset))
    for idx, (paragraph, context, target) in tqdm(enumerate(dataset)):
        
        print('\n', paragraph)
        question_set, score_set, text_set, question_difficulty, question_5w1h, generate_question_type, Event_graph, Relation_graph = create_knowledge_graph(gen_answer, 
                                                                                                                                    context, 
                                                                                                                                    target, 
                                                                                                                                    tokenizer, 
                                                                                                                                    device, 
                                                                                                                                    model_name, 
                                                                                                                                    Event_count)
        for question, score, text, difficulty, q_5w1h, q_type in zip(question_set, score_set, text_set, question_difficulty, question_5w1h, generate_question_type):
            record['paragraph'].append(paragraph)
            record['context'].append(context)
            record['predict'].append(question)
            record['reference'].append(target)
            record['input_text'].append(text)
            record['Extraction type'].append(q_5w1h)
            record['gen_question_type'].append(q_type)
            record['question_difficulty'].append(difficulty)
            record['eval_score'].append(score)
            record['Event_graph'].append('Event_graph')
            record['Relation_graph'].append('Relation_graph')

        if gen_answer: 
            save_csv(record, path = f'csv/1_predict_w_ans_{Event_count}.csv')
        else:
            save_csv(record, path = f'csv/1_predict_{Event_count}.csv')

        count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m', type=str, choices=['Mt0', 'T5', 'flant5'], default='flant5')
    parser.add_argument('--Answer', '-a', type=bool, default=False)
    parser.add_argument('--Event_count', '-c', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(device=device,
         model_name = args.Model,
         gen_answer = args.Answer,
         Event_count = args.Event_count)