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
    path_save_model = checkdir('../save_model', None, Generation, model_name, gen_answer = False)
    Bart_model, Bart_tokenizer = create_model(model_name, Generation, test_mode = True, path_save_model = path_save_model)
    return Bart_model, Bart_tokenizer

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')
    return data.values

def save_csv(record, path):
    row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Question type', 'Golden Answer', 'label', 'T5 small ans', 'T5 base ans', 'T5 ans', 'Correct ans ratio', 'Question_difficulty', 'SentenceBert', 'Event graph', 'Relation graph']

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
    # print(result)
    if result[0][0] > 0.55:
        return (1, ans)
    return (0, ans)

    return (1, ans) if 'Yes' in gpt_response or 'yes' in gpt_response else (0, ans)

def main(ranking_model = 'deberta', device = 'cpu', Event_count = 2):
    dataset = get_data(f'csv/3_question_w_golden_ans_{Event_count}.csv')
    # Bart_model, Bart_tokenizer = load_model('bart')
    # Bart_model.to(device)
    T5_model, T5_tokenizer = load_model('T5')
    T5_model.to(device)
    T5_small_model, T5_small_tokenizer = load_model('T5_small')
    T5_small_model.to(device)
    T5_base_model, T5_base_tokenizer = load_model('T5_base')
    T5_base_model.to(device)
    record = []

    for idx, data in tqdm(enumerate(dataset)):
        # if idx <= 200:
        #     continue
        # if idx > 200:
        #     break

        context = data[1]
        pred = data[2].split('[')[1:-1]
        ques = pred[0].split(']')[1]
        golden_ans = data[6]
        correct_ans_count = 0

        if data[7] == 'Can answer':
            # gpt_text = f'Find the answer from the context and respond simply. Question : {ques}, Context : {context}'
            
            # retry_count = 0
            # gpt_ans = ""  
            # while retry_count < 5:
            #     try:
            #         gpt_ans = geminiapi(gpt_text, temperature=0)
            #         break 
            #     except Exception as e:
            #         print(f"An error occurred: {e}")
            #         retry_count += 1
            #         time.sleep(10)
            
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
            # data.insert(7, gpt_ans)
            data.insert(8, T5_small_ans)
            data.insert(9, T5_base_ans)
            data.insert(10, T5_ans)
            data.insert(11, correct_ans_count)
            record.append(data)
            
            save_csv(record, f'csv/4_correct_ratio_{Event_count}.csv')
        # if idx > 5:
        #     break
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ranking_model', '-r', type=str, choices=['distil', 'deberta'], default='deberta')
    parser.add_argument('--Event_count', '-c', type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(ranking_model = args.Ranking_model, device = device, Event_count = args.Event_count)