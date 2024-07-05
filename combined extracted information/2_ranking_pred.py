import sys 
sys.path.append("..") 
from model import create_model
from helper import enconder, checkdir

import pandas as pd
import argparse
import csv
import torch
import os
from tqdm import tqdm

def read_data(path):
    data = pd.read_csv(path, index_col = False, encoding_errors = 'ignore')
    return data.values

def save_csv(record, path):
    row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Extraction type', 'Question type', 'ranking label',  'Question_difficulty', 'SentenceBert', 'Event graph', 'Relation graph']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for data in record:
            writer.writerow(data)

def pred_data(record, model_name, device, use_answer):

    path_save_model = checkdir('../save_model', Generation = 'ranking', model_name = model_name, gen_answer = False)
    model, tokenizer = create_model(model_name, 'ranking',  test_mode = True, path_save_model = path_save_model)
    model.to(device)

    id2label = {0: 'Can not answer', 1: 'Can answer'}
    new_record = []
    for data in tqdm(record):
        if data[6] == 'type' or data[6] == 'too many':
            continue

        pred = data[2].split('[')[1:-1]
        pred_ques = pred[0].split(']')[1]
        # if use_answer:
        #     pred_ans = pred[1].split(']')[1]
        #     text = pred_ques + ' <SEP> ' + pred_ans + ' <SEP> ' + data[1]
        # elif not use_answer:
        text = pred_ques + ' <SEP> ' + data[1]
        encoded_sent = enconder(tokenizer, 512, text = text)
        input_ids = [encoded_sent.get('input_ids')]
        input_ids = torch.tensor(input_ids).to(device)
        output = model(input_ids=input_ids).logits.argmax().item()
        data = list(data)
        data.insert(7, id2label[output])
        new_record.append(data)
    return new_record

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m',
                        choices=['distil', 'deberta'],
                        type=str,
                        default='deberta')
    parser.add_argument('--Answer', '-a',
                        type=bool,
                        default=False)
    parser.add_argument('--Event_count', '-c', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if args.Answer :
        read_path = f'csv/1_predict_w_ans_{args.Event_count}.csv'
        save_path = f'csv/2_{args.Model}_w_ans_pred_{args.Event_count}.csv'
    else:
        read_path = f'csv/1_predict_{args.Event_count}.csv'
        save_path = f'csv/2_{args.Model}_pred_{args.Event_count}.csv'

    record = read_data(read_path)
    new_record = pred_data(record, args.Model, device, args.Answer)
    save_csv(new_record, save_path)
