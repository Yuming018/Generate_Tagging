import sys 
sys.path.append("..") 
from model import create_model

import pandas as pd
from helper import checkdir, geminiapi, gptapi, enconder
import argparse
import csv
from tqdm import tqdm
import torch
import time
from sentence_transformers import SentenceTransformer, util

def load_model(model_name):
    Generation = 'answer'
    path_save_model = checkdir('../save_model', None, Generation, model_name, use_Answer = False)
    Bart_model, Bart_tokenizer = create_model(model_name, Generation, test_mode = True, path_save_model = path_save_model)
    return Bart_model, Bart_tokenizer

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')
    return data.values

def save_csv(record, path):
    row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Question type', 'label', 'GPT ans', 'Bart ans', 'T5 ans', 'Correct ans ratio', 'Question_difficulty', 'ruler_pred', 'LLM_pred', 'SentenceBert', 'Event graph', 'Relation graph']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record)):
            writer.writerow(record[i])

def main(ranking_model = 'deberta', device = 'cpu'):
    dataset = get_data(f'csv/2_{ranking_model}_w_ans_pred.csv')
    snet_T = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    Bart_model, Bart_tokenizer = load_model('bart')
    Bart_model.to(device)
    T5_model, T5_tokenizer = load_model('T5')
    T5_model.to(device)
    record = []

    for idx, data in tqdm(enumerate(dataset)):
        context = data[1]
        pred = data[2].split('[')[1:-1]
        ques = pred[0].split(']')[1]
        ans = pred[1].split(']')[1]
        correct_ans_count = 1

        gpt_text = f'Find the answer from the context and respond simply. Question : {ques}, Context : {context}'
        
        retry_count = 0
        gpt_ans = ""  
        while retry_count < 5:
            try:
                gpt_ans = gptapi(gpt_text)
                break 
            except Exception as e:
                print(f"An error occurred: {e}")
                retry_count += 1
                time.sleep(5)

        text = f'{ques} <SEP> {context}'
        input_ids = enconder(Bart_tokenizer, 512, text = text).get('input_ids')
        input_ids = torch.tensor(input_ids).to(device)
        output = Bart_model.generate(input_ids=input_ids.unsqueeze(0), max_new_tokens=100)
        bart_ans = Bart_tokenizer.decode(output[0], skip_special_tokens=True)
        
        gpt_text = f"Based on the following text, are these two answers the same? Please answer 'Yes' or 'No'. Context : {context}, Answer 1 : {gpt_ans}, Answer 2 : {bart_ans}"
        gpt_response = ""
        retry_count = 0
        while retry_count < 5:
            try:
                gpt_response = gptapi(gpt_text)
                if 'Yes' in gpt_response or 'yes' in gpt_response or 'No' in gpt_response or 'no' in gpt_response:
                    break 
            except Exception as e:
                print(f"An error occurred: {e}")
                retry_count += 1
                time.sleep(5)
        if 'Yes' in gpt_response or 'yes' in gpt_response:
            correct_ans_count += 1

        # query_embedding = snet_T.encode(gpt_ans)
        # passage_embedding = snet_T.encode(bart_ans)
        # result = util.dot_score(query_embedding, passage_embedding)
        # if result[0][0].item() > 0.8:
        #     correct_ans_count += 1

        input_ids = enconder(T5_tokenizer, 512, text = text).get('input_ids')
        input_ids = torch.tensor(input_ids).to(device)
        output = T5_model.generate(input_ids=input_ids.unsqueeze(0), max_new_tokens=100)
        T5_ans = T5_tokenizer.decode(output[0], skip_special_tokens=True)
        
        gpt_text = f"Based on the following text, are these two answers the same? Please answer 'Yes' or 'No'. Context : {context}, Answer 1 : {gpt_ans}, Answer 2 : {T5_ans}"
        gpt_response = ""
        retry_count = 0
        while retry_count < 5:
            try:
                gpt_response = gptapi(gpt_text)
                if 'Yes' in gpt_response or 'yes' in gpt_response or 'No' in gpt_response or 'no' in gpt_response:
                    break 
            except Exception as e:
                print(f"An error occurred: {e}")
                retry_count += 1
                time.sleep(5)
        if 'Yes' in gpt_response or 'yes' in gpt_response:
            correct_ans_count += 1

        # query_embedding = snet_T.encode(gpt_ans)
        # passage_embedding = snet_T.encode(T5_ans)
        # result = util.dot_score(query_embedding, passage_embedding)
        # if result[0][0].item() > 0.8:
        #     correct_ans_count += 1

        data = list(data)
        data.insert(7, gpt_ans)
        data.insert(8, bart_ans)
        data.insert(9, T5_ans)
        data.insert(10, correct_ans_count)
        record.append(data)
        # if idx > 10:
        #     break
    
    save_csv(record, 'csv/4_correct_ratio.csv')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ranking_model', '-r', type=str, choices=['distil', 'deberta'], default='deberta')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(ranking_model = args.Ranking_model, device = device)