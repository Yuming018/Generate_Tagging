import sys 
sys.path.append("..") 
from helper import gptapi, geminiapi

import csv
import argparse
import pandas as pd
import logging
from collections import defaultdict
from tqdm import tqdm

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')
    return data.values

def save_csv(record, path):
    row = ['Paragraph', 'Content', 'Difficulty level','Prediction']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record['Paragraph'])):
            writer.writerow([record['Paragraph'][i], record['Content'][i], record['Difficulty level'][i], record['Prediction'][i]])

def judge_response(response, exampler):
    judge = defaultdict(bool)
    for line in response.split('\n'):
        if 'Very Easy' in line:
            judge['very easy'] = True
        if 'Simple' in line :
            judge['Simple'] = True
        if 'Medium' in line:
            judge['Medium'] = True
        if 'Difficult' in line:
            judge['Difficult'] = True
    if len(response.split('\n')) >= (exampler+1)*4 and len(response.split('\n')) <= (exampler+2)*4:
        judge['len'] = True
    return judge['very easy'] and judge['Simple'] and judge['Medium'] and judge['Difficult'] and judge['len']

def judge_response_2(response, exampler, level):
    judge = defaultdict(bool)
    # for line in response.split('\n'):
    #     if level in line:
    #         judge[level] = True
    if len(response.split('\n')) >= (exampler+1) and len(response.split('\n')) <= (exampler+2):
        judge['len'] = True
    return judge[level] and judge['len']

def main(model_name = "openai", exampler = 5):
    record = defaultdict(list)
    dataset = get_data('../data/test.csv')
    temp = 0
    curr_para = ""

    for data in tqdm(dataset):
        paragraph = data[4]
        context = data[1]
        para = "-".join(paragraph.split('-')[:-1])
        if curr_para == para:
            continue
        curr_para = para

        # for level in ['Very Easy', 'Simple', 'Medium', 'Difficult']:
            # prompt = f"Based on the following context, generate {exampler} {level} questions questions.A total of {exampler} questions.\
            #     The answers to the generated questions must be found in the article.  Just generate question, don't generate answer.\
            #     Context as following: : {context}"
        prompt = f"Based on the following context, generate {exampler} Very Easy questions, {exampler} Simple questions, {exampler} Medium questions, and {exampler} Difficult questions.A total of {4*exampler} questions.\
            The answers to the generated questions must be found in the article.  Just generate question, don't generate answer.\
            Context as following: : {context}"
            # prompt = f"幫我依據以下的文章，分別生成出五個非常簡單的問題，五個簡單的問題，中間五個中等的問題，最後五個困難的問題，總共 20 個問題。\
            #     難度分級依照跨段落距離當參考基準，舉例來說，當問題與答案出現在同一個句子中，便把他歸類為簡單問題，\
            #     當問題與答案相隔的句子大約2-3句話的時候，便把他歸類為中等問題，當問題與答案相隔的鋸子約為4-5句話的時候，\
            #     便把他歸類為困難的問題，但生成的出來問題，答案一定要出現在文章中。文章如下:'{context}'"
        try:
            if model_name == "openai":
                response = gptapi(prompt, temperature=0)
            elif model_name == "gemini":
                response = geminiapi(prompt)

            count = 0
            # while not judge_response_2(response, exampler, level):
            while not judge_response(response, exampler):
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
            if 'Very Easy' in line:
                level = 'Very Easy'
            if 'Simple' in line:
                level = 'Simple'
            if 'Medium' in line:
                level = 'Medium'
            if 'Difficult' in line:
                level = 'Difficult'
            if level not in line and 'question' not in line and 'Questions' not in line and line != '' and level != '':
                record['Difficulty level'].append(level)
                record['Paragraph'].append(data[4])
                record['Content'].append(context)
                if '(' in line:
                    record['Prediction'].append(line.split('('))
                else:
                    record['Prediction'].append(line)
        
        save_csv(record, f"csv/0_{model_name}_generate_{exampler}.csv")
        temp += 1
        if temp == 6:
            break

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m', type=str, choices=['openai', 'gemini'], default='openai')
    parser.add_argument('--exampler', '-e', type=int, default=5)
    args = parser.parse_args()

    main(model_name = args.Model, exampler = args.exampler)