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
        # data = data.drop(columns=['Event graph', 'Relation graph', 'Input_text', 'SentenceBert'])
        data = data.fillna('')
        return data.values

def process_our_dataset(dataset):
    record = defaultdict(lambda: defaultdict(set))
    all_record = defaultdict(lambda: defaultdict(list))
    our_record = defaultdict(lambda: defaultdict(list))

    for data in dataset:
        if data[6] == 'Can answer':
            ques = data[2].split('[')[1:-1]
            ques = ques[0].split(']')[1]
            if int(data[7]) < 2:
                record[data[0]]['Simple'].add(ques)
            elif int(data[7]) < 5:
                record[data[0]]['Medium'].add(ques)
            else:
                record[data[0]]['Difficult'].add(ques)
        all_record[data[0]]['Story'].append(data[1])

    for story in record:
        for level in record[story]:
            random_questions = random.sample(list(record[story][level]), 3)
            all_record[story][level] = random_questions[:]
            our_record[story][level] = random_questions[:]
    return all_record, our_record

def process_llm_dataset(dataset, all_record):
    llm_record = defaultdict(lambda: defaultdict(list))
    for data in dataset:
        ques = data[3].split('.')[1]
        level = ''
        if data[2] == 'Simple':
            level = 'Simple'
        elif data[2] == 'Medium':
            level = 'Medium'
        elif data[2] == 'Difficult':
            level = 'Difficult'
        all_record[data[0]][level].append(ques)
        llm_record[data[0]][level].append(ques)
    return all_record, llm_record

def decide_level(store_record, response, our_record, llm_record, para, all_record):
    
    new_record = defaultdict(list)
    level = ''
    for line in response.split('\n'):
        if 'Simple' in line:
            level = 'Simple'
        if 'Medium' in line:
            level = 'Medium'
        if 'Difficult' in line:
            level = 'Difficult'
        if 'question' not in line and 'Questions' not in line and line != '' and level != '':
            new_record[level].append(line.split('(')[0].lower())
    
    # print('Our_question')
    for level in ['Simple', 'Medium', 'Difficult']:
        # print(level)
        for ques in our_record[level]:
            ques = ques.lower()
            ques = " ".join(ques.split(" ")[1:-1])
            for LLM_level in new_record:
                for LLM_quest in new_record[LLM_level]:
                    if ques in LLM_quest:
                        # print(ques, LLM_level)
                        store_record['paragraph'].append(para)
                        store_record['context'].append(all_record[para]['Story'][0])
                        store_record['who_generate'].append('Our')
                        store_record['question'].append(ques)
                        store_record['origin_level'].append(level)
                        store_record['pred_level'].append(LLM_level)

    # print('LLM_question')
    for level in ['Simple', 'Medium', 'Difficult']:
        # print(level)
        for ques in llm_record[level]:
            ques = ques.lower()
            for LLM_level in new_record:
                for LLM_quest in new_record[LLM_level]:
                    if ques in LLM_quest:
                        # print(ques, LLM_level)
                        store_record['paragraph'].append(para)
                        store_record['context'].append(all_record[para]['Story'][0])
                        store_record['who_generate'].append('LLM')
                        store_record['question'].append(ques)
                        store_record['origin_level'].append(level)
                        store_record['pred_level'].append(LLM_level)
    return store_record

def save_csv(record, path):
    row = ['Paragraph', 'Content', 'who generate','Question', 'Origin level', 'Pred level']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record['paragraph'])):
            writer.writerow([record['paragraph'][i], record['context'][i], record['who_generate'][i], record['question'][i], record['origin_level'][i], record['pred_level'][i]])

def main(model_name = "openai", ranking_model = 'deberta'):
    store_record = defaultdict(list)
    our_dataset = get_data(f'csv/{ranking_model}_pred.csv')
    llm_dataset = get_data(f'csv/{model_name}.csv')

    all_record, our_record = process_our_dataset(our_dataset)
    all_record, llm_record = process_llm_dataset(llm_dataset, all_record)
    
    for para in tqdm(all_record):
        context = all_record[para]['Story'][0]
        all_ques = "" 
        count = 1
        all_quest_set = []

        for level in all_record[para]:
            if level != 'Story':
                all_quest_set += all_record[para][level]

        random.shuffle(all_quest_set)
        for ques in all_quest_set:
            all_ques += f'{count} :{ques}, '
            count += 1

        prompt = f"Help me classify the questions based on the following context and questions into levels of difficulty, categorizing them as Simple, Medium, and Difficult questions. \
            Difficulty levels are determined by the distance across paragraphs as a reference. \
            For example, when a question and its answer appear in the same sentence, it is classified as an easy question. \
            When the question and its answer are separated by approximately 1-2 sentences, it is classified as a medium question. \
            When the question and its answer are separated by approximately 3-4 sentences, it is classified as a hard question. \
            And the categorized questions should be exactly the same as the original questions.\
            The context is as follows: '{context}'. The questions are as follows: {all_ques}"
        # prompt = f"幫我依據以下的文章以及問題，把問題進行難度分類，分別分類成簡單的問題，中等的問題以及困難的問題。\
        #     難度分級依照跨段落距離當參考基準，舉例來說，當問題與答案出現在同一個句子中，便把他歸類為簡單問題，\
        #     當問題與答案相隔的句子大約1~2句話的時候，便把他歸類為中等問題，當問題與答案相隔的句子約為3~4句話的時候，\
        #     便把他歸類為困難的問題，並且分類出來的問題要跟原始問題一模一樣。文章如下:'{context}', 問題如下 : {all_ques}"
        # try:

        if model_name == "openai":
            response = gptapi(prompt, version=3.5)
        elif model_name == "gemini":
            response = geminiapi(prompt)
        
        # print(response)
        store_record = decide_level(store_record, response, our_record[para], llm_record[para], para, all_record)
        
            # count = 0
            # while True:
            #     if model_name == "openai":
            #         response = gptapi(prompt)
            #     elif model_name == "gemini":
            #         response = geminiapi(prompt)
            #     if count == 10:
            #         break
            #     count += 1
        # except Exception as e:
        #     logging.error(f"An error occurred: {e}")
        #     print(f"An error occurred: {e}")

    save_csv(store_record, f"csv/final.csv")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m', type=str, choices=['openai', 'gemini'], default='openai')
    parser.add_argument('--Ranking_model', '-r', type=str, choices=['distil', 'deberta'], default='deberta')
    args = parser.parse_args()

    main(model_name = args.Model, ranking_model = args.Ranking_model)