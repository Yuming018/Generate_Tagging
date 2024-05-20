import sys 
sys.path.append("..") 
from helper import gptapi, geminiapi

import csv
import argparse
import pandas as pd
import logging
from collections import defaultdict
from tqdm import tqdm

file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)

def get_data(path):
    data = pd.read_csv(path)
    # data = data.drop(columns=['Event graph', 'Relation graph', 'Input_text', 'SentenceBert'])
    data = data.fillna('')
    return data.values

def save_csv(record, path):
    row = ['Paragraph', 'Content', 'Difficulty level','Prediction']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record['Paragraph'])):
            writer.writerow([record['Paragraph'][i], record['Content'][i], record['Difficulty level'][i], record['Prediction'][i]])

def judge_response(response):
    judge = defaultdict(bool)
    for line in response.split('\n'):
        if 'Simple' in line :
            judge['Simple'] = True
        if 'Medium' in line:
            judge['Medium'] = True
        if 'Difficult' in line:
            judge['Difficult'] = True
    if len(response.split('\n')) >= 12 and len(response.split('\n')) <= 15:
        judge['len'] = True
    return judge['Simple'] and judge['Medium'] and judge['Difficult'] and judge['len']

def main(model_name = "openai"):
    record = defaultdict(list)
    dataset = get_data('../data/test.csv')
    exampler = 3
    temp = 0
    paragraph = ""
    for data in tqdm(dataset):
        if paragraph == data[4]:
            continue
        paragraph = data[4]
        context = data[1]
        
        prompt = f"Based on the following article, generate the first {exampler} simple questions, the next {exampler} medium questions, and the last {exampler} difficult questions. Just generate question, don't generate answer.\
            Difficulty levels are categorized based on the distance across paragraphs. \
            For example, if the question and answer appear in the same sentence, it is classified as a simple question.\
            If the question and answer are separated by approximately 2-3 sentences, it is categorized as a Medium question.\
            If the question and answer are separated by approximately 4-5 sentences, it is classified as a difficult question.\
            But the generated questions must have their answers appear in the Context.Context as following: : {context}"
        # prompt = f"幫我依據以下的文章，分別生成出前面三個簡單的問題，中間三個中等的問題，最後三個困難的問題。\
        #     難度分級依照跨段落距離當參考基準，舉例來說，當問題與答案出現在同一個句子中，便把他歸類為簡單問題，\
        #     當問題與答案相隔的句子大約2-3句話的時候，便把他歸類為中等問題，當問題與答案相隔的鋸子約為4-5句話的時候，\
        #     便把他歸類為困難的問題，但生成的出來問題，答案一定要出現在文章中。文章如下:'{context}'"
        try:
            if model_name == "openai":
                response = gptapi(prompt)
            elif model_name == "gemini":
                response = geminiapi(prompt)

            count = 0
            while not judge_response(response):
                if model_name == "openai":
                    response = gptapi(prompt)
                elif model_name == "gemini":
                    response = geminiapi(prompt)
                if count == 10:
                    break
                count += 1
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print(f"An error occurred: {e}")
        
        level = ''
        for line in response.split('\n'):
            if 'Simple' in line:
                level = 'Simple'
            if 'Medium' in line:
                level = 'Medium'
            if 'Difficult' in line:
                level = 'Difficult'
            if 'question' not in line and 'Questions' not in line and line != '' and level != '':
                record['Difficulty level'].append(level)
                record['Paragraph'].append(data[4])
                record['Content'].append(context)
                if '(' in line:
                    record['Prediction'].append(line.split('('))
                else:
                    record['Prediction'].append(line)
        temp += 1
        if temp == 5:
            break

    save_csv(record, f"csv/0_{model_name}_generate.csv")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m', type=str, choices=['openai', 'gemini'], default='openai')
    args = parser.parse_args()

    main(model_name = args.Model)