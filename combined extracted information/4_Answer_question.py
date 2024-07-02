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

sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

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
    snet_T = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    input_ids = enconder(tokenizer, 512, text = text).get('input_ids')
    input_ids = torch.tensor(input_ids).to(device)
    output = model.generate(input_ids=input_ids.unsqueeze(0), max_new_tokens=100)
    ans = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # LLM response
    # gpt_text = f"Based on the following text, are these two answers the same? Please answer 'Yes' or 'No'. Context : {context}, Answer 1 : {golden_ans}, Answer 2 : {ans}"
    # gpt_response = ""
    # retry_count = 0
    # while retry_count < 5:
    #     try:
    #         gpt_response = gptapi(gpt_text,
    #                             version=4,
    #                             temperature=0)
    #         if 'Yes' in gpt_response or 'yes' in gpt_response or 'No' in gpt_response or 'no' in gpt_response:
    #             break 
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         retry_count += 1
    #         time.sleep(10)
    
    # sntence_t
    # print(ans, golden_ans)
    # query_embedding = snet_T.encode(golden_ans)
    # passage_embedding = snet_T.encode(ans)
    # result = util.dot_score(query_embedding, passage_embedding)
    # print(result)

    # Cosine similarity
    embeddings1 = sentence_model.encode([ans])
    embeddings2 = sentence_model.encode([golden_ans])
    result = cosine_similarity(embeddings1, embeddings2)
    if result[0][0] > 0.52:
        return (1, ans)
    return (0, ans)

    return (1, ans) if 'Yes' in gpt_response or 'yes' in gpt_response else (0, ans)

def main(device = 'cpu', Event_count = 2, our_or_llm ='our', exampler = 5, answer = False):
    if our_or_llm == 'our':
        if answer:
            dataset = get_data(f'csv/3_question_ans_w_golden_ans_{Event_count}.csv')
        elif not answer:
            dataset = get_data(f'csv/3_question_w_golden_ans_{Event_count}.csv')
    elif our_or_llm == 'llm':
        dataset = get_data(f'csv/3_llm_question_w_golden_ans_{exampler}.csv')
    elif our_or_llm == 'fairytale':
        dataset = get_data('../data/train.csv')
    elif our_or_llm == 'fairytale_w_llm':
        dataset = get_data('csv/3_fairytale_question_w_golden.csv')

    # Bart_model, Bart_tokenizer = load_model('bart')
    # Bart_model.to(device)
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
            golden_ans = data[6]
            insert_index = 8
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
            if idx < 7000:
                continue
            paragraph = data[4]
            if curr_para == paragraph:
                continue
            curr_para = paragraph

            context = data[1]
            ques = data[2]
            golden_ans = data[3]
            data = [data[4]] + list(data)[1:4] + [data[0]]
            insert_index = 4
            # save_path = f'csv/4_fairytale_correct_ratio.csv'
            save_path = f'csv/4_aaa.csv'
            if idx == 7070:
                break
        elif our_or_llm == 'fairytale_w_llm':
            context = data[1]
            ques = data[2]
            golden_ans = data[3]
            insert_index = 5
            save_path = f'csv/4_fairytale_w_llm_correct_ratio.csv'
        
        correct_ans_count = 0
        text = f'{ques} <SEP> {context}'
        # count, bart_ans = is_answer_correct(context, gpt_ans, text, Bart_model, Bart_tokenizer)
        # correct_ans_count += count
        count, T5_small_ans = is_answer_correct(context, golden_ans, text, T5_small_model, T5_small_tokenizer)
        correct_ans_count += count
        count, T5_base_ans = is_answer_correct(context, golden_ans, text, T5_base_model, T5_base_tokenizer)
        correct_ans_count += count
        count, T5_ans = is_answer_correct(context, golden_ans, text, T5_model, T5_tokenizer)
        correct_ans_count += count

        data = list(data)
        data.insert(insert_index, T5_small_ans)
        data.insert(insert_index+1, T5_base_ans)
        data.insert(insert_index+2, T5_ans)
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
    main(device = device, 
         Event_count = args.Event_count, 
         our_or_llm = args.our_or_llm, 
         exampler = args.exampler,
         answer = args.Answer)