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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

def load_model(model_name):
    Generation = 'answer'
    path_save_model = checkdir('../save_model', Generation, model_name, gen_answer = False)
    Bart_model, Bart_tokenizer = create_model(model_name, Generation, test_mode = True, path_save_model = path_save_model)
    return Bart_model, Bart_tokenizer

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')
    return data.values

def save_csv(record, path, our_or_llm):
    if our_or_llm == 'our':
        row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Extraction type', 'Question type', 'Golden Answer', 'T5 small ans', 'T5 base ans', 'T5 large ans', 'Correct ans ratio', 'Question_difficulty', 'SentenceBert', 'Event graph', 'Relation graph']
    elif our_or_llm == 'llm':
        row = ['Paragraph', 'Context', 'Difficulty level', 'Prediction', 'Golden Answer', 'T5 small ans', 'T5 base ans', 'T5 large ans', 'Correct ans ratio']
    elif our_or_llm == 'fairytale':
        row = ['Paragraph', 'Context', 'Prediction', 'Golden Answer', 'T5 small ans', 'T5 base ans', 'T5 large ans', 'Correct ans ratio', 'Question type']
    elif our_or_llm == 'fairytale_w_llm':
        row = ['Paragraph', 'Context', 'Question', 'Fairytale Answer', 'Golden Answer', 'T5 small ans', 'T5 base ans', 'T5 large ans', 'Correct ans ratio', 'Question type']
    
    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record)):
            writer.writerow(record[i])

def is_answer_correct(context, golden_ans, text, model, tokenizer):
    
    input_ids = enconder(tokenizer, 512, text = text).get('input_ids')
    input_ids = torch.tensor(input_ids).to(device)
    output = model.generate(input_ids=input_ids.unsqueeze(0), max_new_tokens=100)
    ans = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # sntence_bert
    snet_T = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1').to(device)
    query_embedding = snet_T.encode(golden_ans, convert_to_tensor=True).to(device)
    passage_embedding = snet_T.encode(ans, convert_to_tensor=True).to(device)
    result = util.pytorch_cos_sim(query_embedding, passage_embedding)

    if result[0][0] >= 0.5:
        return (1, ans, result[0][0])
    return (0, ans, result[0][0])

def main(device = 'cpu', Event_count = 2, our_or_llm ='our', exampler = 5, answer = False):
    if our_or_llm == 'our':
        if answer:
            dataset = get_data(f'csv/3_question_ans_w_golden_ans_{Event_count}.csv')
        elif not answer:
            dataset = get_data(f'csv/3_question_w_golden_ans_{Event_count}.csv')
    elif our_or_llm == 'llm':
        dataset = get_data(f'csv/3_llm_question_w_golden_ans_{exampler}.csv')
    elif our_or_llm == 'fairytale':
        dataset = get_data('../data/test.csv')
    elif our_or_llm == 'fairytale_w_llm':
        dataset = get_data('csv/3_fairytale_question_w_golden.csv')

    T5_model, T5_tokenizer = load_model('T5')
    T5_model.to(device)
    T5_small_model, T5_small_tokenizer = load_model('T5_small')
    T5_small_model.to(device)
    T5_base_model, T5_base_tokenizer = load_model('T5_base')
    T5_base_model.to(device)
    record = []
    curr_para = ""

    for idx, data in tqdm(enumerate(dataset)):

        if our_or_llm == 'our':
            context = data[1]
            pred = data[2].split('[')[1:-1]
            ques = pred[0].split(']')[1]
            golden_ans = data[7]
            insert_index = 8
            if 'False' in golden_ans:
                continue
            if answer:
                save_path = f'csv/4_w_ans_correct_ratio_{Event_count}.csv'
            elif not answer:
                save_path = f'csv/4_wo_ans_correct_ratio_{Event_count}.csv'
        elif our_or_llm == 'llm':
            context = data[1]
            ques = data[3].split('.')[-1]
            golden_ans = data[4]
            insert_index = 5
            save_path = f'csv/4_llm_correct_ratio_{exampler}.csv'
        elif our_or_llm == 'fairytale':
            paragraph = data[4]
            if curr_para == paragraph:
                continue
            curr_para = paragraph

            context = data[1]
            ques = data[2]
            golden_ans = data[3]
            data = [data[4]] + list(data)[1:4] + [data[0]]
            insert_index = 4
            save_path = f'csv/4_fairytale_correct_ratio.csv'

        elif our_or_llm == 'fairytale_w_llm':
            context = data[1]
            ques = data[2]
            golden_ans = data[3]
            insert_index = 5
            save_path = f'csv/4_fairytale_w_llm_correct_ratio.csv'
        
        correct_ans_count = 0
        text = f'{ques} <SEP> {context}'
        count, T5_small_ans, T5_small_score = is_answer_correct(context, golden_ans, text, T5_small_model, T5_small_tokenizer)
        correct_ans_count += count
        count, T5_base_ans, T5_base_score = is_answer_correct(context, golden_ans, text, T5_base_model, T5_base_tokenizer)
        correct_ans_count += count
        count, T5_ans, T5_score = is_answer_correct(context, golden_ans, text, T5_model, T5_tokenizer)
        correct_ans_count += count

        data = list(data)
        data.insert(insert_index, T5_small_ans + f' ({round(float(T5_small_score), 2)})')
        data.insert(insert_index+1, T5_base_ans + f' ({round(float(T5_base_score), 2)})')
        data.insert(insert_index+2, T5_ans + f' ({round(float(T5_score), 2)})')
        data.insert(insert_index+3, correct_ans_count)
        record.append(data)
        
        save_csv(record, save_path, our_or_llm)
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--our_or_llm', '-m', type=str, choices=['our', 'llm', 'fairytale', 'fairytale_w_llm'], default='our')
    parser.add_argument('--Answer', '-a', type=bool, default=False)
    parser.add_argument('--Event_count', '-c', type=int, default=2)
    parser.add_argument('--exampler', '-e', type=int, default=5)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(device = device, 
         Event_count = args.Event_count, 
         our_or_llm = args.our_or_llm, 
         exampler = args.exampler,
         answer = args.Answer)
		

    # ans_list = ['his beautiful young wife to face the world alone.', 
    #             'a little boy was born to her.',
    #             'his beautiful young wife to face the world alone.']
    # golden_ans = " to face the world alone "    
    # for ans in ans_list:
    #     print(ans)
    #     snet_T = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1').to(device)
    #     query_embedding = snet_T.encode(golden_ans, convert_to_tensor=True).to(device)
    #     passage_embedding = snet_T.encode(ans, convert_to_tensor=True).to(device)
    #     result = util.pytorch_cos_sim(query_embedding, passage_embedding)
    #     print(result)