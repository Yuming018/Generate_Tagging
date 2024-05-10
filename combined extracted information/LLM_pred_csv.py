import sys 
sys.path.append("..") 
from helper import gptapi, geminiapi

import csv
import argparse
import pandas as pd
import logging
from collections import defaultdict
from tqdm import tqdm
import random

file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)

def get_data(path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values

def save_csv(record, path):
    row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Question type', 'label',  'Question_difficulty', 'ruler_pred', 'LLM_pred', 'SentenceBert', 'Event graph', 'Relation graph']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record)):
            writer.writerow(record[i])

def judge_pred(response):
    level = ""
    for line in response.split('\n'):
        if 'simple' in line or 'Simple' in line:
            level = 'Simple'
        elif 'medium' in line or 'Medium' in line:
            level = 'Medium'
        elif 'hard' in line or 'Hard' in line:
            level = 'Difficult'
    return level
    
def main(model_name = "openai", ranking_model = 'deberta'):
    record = []
    dataset = get_data(f'csv/{ranking_model}_pred.csv')

    para = dataset[0][0]
    for data in tqdm(dataset):
        if data[0] != para:
            break
        context = data[1]
        ques = data[2] 

        prompt = f"Help me classify the questions based on the following passage and questions into categories of easy, \
            medium, or difficult. The difficulty classification is based on the distance between paragraphs. \
            For example, if the question and answer appear in the same sentence, classify it as an simple question. \
            If the question and answer are separated by approximately 2-3 sentences, classify it as a medium question. \
            If the question and answer are separated by approximately 4-5 sentences, classify it as a hard question. \
            Just tell me simple, medium, or hard categories.\
            The passage is as follows: '{context}', and the questions are: {ques}"
        # prompt = f"幫我依據以下的文章以及問題，把問題進行難度分類，分類成簡單的問題、中等的問題或困難的問題。\
        #     難度分級依照跨段落距離當參考基準，舉例來說，當問題與答案出現在同一個句子中，便把他歸類為簡單問題，\
        #     當問題與答案相隔的句子大約2-3句話的時候，便把他歸類為中等問題，當問題與答案相隔的句子約為4-5句話的時候，\
        #     便把他歸類為困難的問題。文章如下:'{context}', 問題如下 : {ques}"
        if data[6] == 'Can answer':
            try:
                if model_name == "openai":
                    response = gptapi(prompt, version=3.5, temperature =0)
                elif model_name == "gemini":
                    response = geminiapi(prompt)
                level = judge_pred(response)

                count = 0
                while level == "":
                    if model_name == "openai":
                        response = gptapi(prompt, temperature =0)
                    elif model_name == "gemini":
                        response = geminiapi(prompt)
                    if count == 10:
                        break
                    count += 1
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                print(f"An error occurred: {e}")
            
            data = list(data)
            data.insert(8, level)
            if int(data[7]) < 2:
                data.insert(8, 'Simple')
            elif int(data[7]) < 5:
                data.insert(8, 'Medium')
            else:
                data.insert(8, 'Difficult')
        elif data[6] == 'Can not answer':
            data = list(data)
            data.insert(8, None)
            data.insert(8, None)
        record.append(data)

    save_csv(record, f"csv/test.csv")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m', type=str, choices=['openai', 'gemini'], default='openai')
    parser.add_argument('--Ranking_model', '-r', type=str, choices=['distil', 'deberta'], default='deberta')
    args = parser.parse_args()

    main(model_name = args.Model, ranking_model = args.Ranking_model)