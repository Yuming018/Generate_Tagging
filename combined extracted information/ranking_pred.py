import sys 
sys.path.append("..") 
from model import create_model
from helper import enconder, check_checkpoint, checkdir

import pandas as pd
import argparse
import csv
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

def read_data(path):
    data = pd.read_csv(path, index_col = False, encoding_errors = 'ignore')
    return data.values

def save_csv(record, path):
    row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Question type', 'label',  'Question_difficulty', 'SentenceBert', 'Event graph', 'Relation graph']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for data in record:
            writer.writerow(data)

def pred_data(record, model_name, device, use_answer):
    model, tokenizer = create_model(model_name, 'ranking')
    path_save_model = checkdir('../save_model', None, 'ranking', model_name, use_answer)
    model_path = check_checkpoint(path_save_model)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    id2label = {0: 'Can not answer', 1: 'Can answer'}
    new_record = []
    for data in tqdm(record):
        pred = data[2].split('[')[1:-1]
        pred_ques = pred[0].split(']')[1]
        pred_ans = pred[1].split(']')[1]
        if use_answer:
            text = data[1] + ' <SEP> ' + pred_ques + ' <SEP> ' + pred_ans
        elif not use_answer:
            text = pred_ques + ' <SEP> ' + data[1]
        encoded_sent = enconder(tokenizer, 512, text = text)
        input_ids = [encoded_sent.get('input_ids')]
        input_ids = torch.tensor(input_ids).to(device)
        output = model(input_ids=input_ids).logits.argmax().item()
        data = list(data)
        data.insert(6, id2label[output])
        new_record.append(data)
    return new_record

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m',
                        choices=['distil', 'deberta'],
                        type=str,
                        default='distil')
    parser.add_argument('--Answer', '-a',
                        type=bool,
                        default=False)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    record = read_data('csv/predict.csv')
    new_record = pred_data(record, args.Model, device, args.Answer)
    
    if not os.path.isdir('csv'):
        os.mkdir('csv')
    if args.Answer :
        path = f'csv/{args.Model}_w_ans_pred.csv'
    else:
        path = f'csv/{args.Model}_pred.csv'
    save_csv(new_record, path)
