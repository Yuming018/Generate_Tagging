import sys 
sys.path.append("..") 

import pandas as pd
import argparse
import csv
from tqdm import tqdm
import time
from helper import geminiapi, gptapi
import evaluate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

metric = evaluate.load("bleu")
model = SentenceTransformer('all-mpnet-base-v2')
snet_T = SentenceTransformer('all-mpnet-base-v2')
sleep_time = 5

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')
    return data.values

def save_csv(record, path, our_or_llm):
    if our_or_llm == 'our':
        row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Extraction type', 'Question type', 'Golden Answer', 'Question_difficulty', 'SentenceBert', 'Event graph', 'Relation graph']
    elif our_or_llm == 'llm':
        row = ['Paragraph', 'Context', 'Difficulty level', 'Prediction', 'Golden Answer']
    elif our_or_llm == 'fairytale':
        row = ['Paragraph', 'Context', 'Question', 'Fairytale Answer', 'Golden Answer']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record)):
            writer.writerow(record[i])

def main(ranking_model = 'deberta', answer = False, Event_count = 2, our_or_llm ='our', exampler = 5):
    if our_or_llm == 'our':
        if answer:
            dataset = get_data(f'csv/2_{ranking_model}_w_ans_pred_{Event_count}.csv')
        elif not answer:
            dataset = get_data(f'csv/2_{ranking_model}_pred_{Event_count}.csv')
    elif our_or_llm == 'llm':
        dataset = get_data(f'csv/0_openai_generate_{exampler}.csv')
    elif our_or_llm == 'fairytale':
        dataset = get_data('../data/test.csv')

    record = []
    
    for idx, data in tqdm(enumerate(dataset)):
        # if idx < 2903:
        #     continue
        # if idx == 2904:
        #     break
        
        if our_or_llm == 'our':
            if data[7] == 'Can not answer':
                continue

            context = data[1]
            pred = data[2].split('[')[1:-1]
            ques = pred[0].split(']')[1]
            insert_index = 7
            if answer:
                save_path = f'csv/3_question_ans_w_golden_ans_{Event_count}.csv'
                try:
                    ans = pred[1].split(']')[1]
                except:
                    print(pred)
                    continue
            elif not answer:
                save_path = f'csv/3_question_w_golden_ans_{Event_count}.csv'
                temp = data[4].split('[')[1:-1]
                ans = temp[-1].split(']')[1]
            data = list(data)[:7] + list(data)[8:] # remove label
        elif our_or_llm == 'llm':
            context = data[1]
            ques = data[3].split('.')[-1]
            insert_index = 4
            save_path = f'csv/3_llm_question_w_golden_ans_{exampler}.csv'
        elif our_or_llm == 'fairytale':
            context = data[1]
            ques = data[2]
            data = [data[4]] + list(data)[1:4]
            insert_index = 4
            save_path = f'csv/3_fairytale_question_w_golden.csv'
            if idx == 30:
                break
        
        data = list(data)
        ans_flag, error_flag = check_ans(ans, ques, context)
        if error_flag:
            data.insert(insert_index, "False")
        else:
            if ans_flag:
                data.insert(insert_index, ans)
            elif not ans_flag:
                result, gpt_ans = create_ans(ques, context)
                if result[0][0] > 0.8:
                    data.insert(insert_index, gpt_ans)
                else:
                    data.insert(insert_index, "False, " + gpt_ans)
        record.append(data)
        save_csv(record, save_path, our_or_llm)
    return

def check_ans(ans, ques, context):
    retry_count = 0
    error_flag = False
    while retry_count < 3:
        correct_count = 0
        # gpt_text = f'Without considering completeness and details, is this answer correct in responding to this question? Please answer "Yes" or "No".Answer : {gemini_ans} Question : {ques}, Context : {context}'
        # gpt_text = f'Is this answer correct in responding to this question? Please answer "Yes" or "No".Answer : {ans}, Question : {ques}, Context : {context}'
        gpt_text = f'Context : {context}.\nQuestion : {ques}.\nAnswer : {ans}\nIs this answer correct in responding to this question? Please answer "Yes" or "No".'

        retry_count_gpt = 0
        while retry_count_gpt < 5:
            try:
                gpt_ans = gptapi(gpt_text, temperature=0)
                if 'Yes' in gpt_ans or 'yes' in gpt_ans:
                    correct_count += 1
                    break
                elif 'No' in gpt_ans or 'no' in gpt_ans:
                    break
            except Exception as e:
                print(f"Chatgpt : An error occurred: {e}")
                retry_count_gpt += 1
                time.sleep(sleep_time)
        gemini_text = f'Context : {context}.\nQuestion : {ques}.\nAnswer : {ans}.\nIs this answer correct in responding to this question? Please answer "Yes" or "No".'
        retry_count_gemini = 0
        while retry_count_gemini < 5:
            try:
                gemini_ans = geminiapi(gemini_text, temperature=0)
                if 'Yes' in gemini_ans or 'yes' in gemini_ans:
                    correct_count += 1
                    break
                elif 'No' in gemini_ans or 'no' in gemini_ans:
                    break
            except Exception as e:
                print(f"Gemini : An error occurred: {e}")
                if "response.prompt_feedback" in str(e):
                    error_flag = True
                    break
                retry_count_gemini += 1
                time.sleep(sleep_time)
        retry_count += 1
        if error_flag:
            break

        if correct_count == 2:
            break
    return (True, error_flag) if correct_count == 2 else (False, error_flag)

def create_ans(ques, context):
    gemini_ans = ""
    gpt_ans = "" 
    retry_count = 0
    error_flag = False
    while retry_count < 7:
        gpt_text = f'Extract the answer from the text and reply with just the answer. Question : {ques}, Context : {context}'
        # Example: Context : let me try , " cried biernuga , the bony fish , but he had no better luck , and no more had kumbal , the bream , nor any of the rest . " it is no use , " exclaimed thuggai , at last . " the wood is too wet . we must just sit and wait till the sun comes out again and dries it . " then a very little fish indeed , not more than four inches long and the youngest of the tribe , bowed himself before thuggai , saying , " ask my father , guddhu the cod , to light the fire . he is skilled in magic more than most fishes . " so thuggai asked him , and guddhu stripped some pieces of bark off a tree , and placed them on top of the smouldering ashes . then he knelt by the side of the fire and blew at it for a long while , till slowly the feeble red glow became a little stronger and the edges of the bark showed signs of curling up . when the rest of the tribe saw this they pressed close , keeping their backs towards the piercing wind , but guddhu told them they must go to the other side , as he wanted the wind to fan his fire . by and by the spark grew into a flame , and a merry crackling was heard .\
        #     Question : why couldn"t the fish tribe light the fire again ?, Answer : the wood was too wet .\
        #     Question : who was skilled in magic more than most fishes ?, Answer : guddhu the cod .'
        retry_count_gpt = 0
        while retry_count_gpt < 5:
            try:
                gpt_ans = gptapi(gpt_text, temperature=0)
                # if 'Yes' in gpt_ans or 'yes' in gpt_ans or 'No' in gpt_ans or 'no' in gpt_ans:
                #     break
                if '\n' in gemini_ans:
                    raise ValueError("Response contains newline")
                break
            except Exception as e:
                print(f"Chatgpt : An error occurred: {e}")
                retry_count_gpt += 1
                time.sleep(sleep_time)

        gemini_text = f'Extract the answer from the text and reply with just the answer. Respond in {len(gpt_ans.split())} words. Question : {ques}, Context : {context}'
        retry_count_gemini = 0
        while retry_count_gemini < 5:
            try:
                gemini_ans = geminiapi(gemini_text, temperature=0)
                if '\n' in gemini_ans:
                    raise ValueError("Response contains newline")
                break 
            except Exception as e:
                if e == "The `response.parts` quick accessor only works for a single candidate, but none were returned. Check the `response.prompt_feedback` to see if the prompt was blocked.":
                    error_flag = True
                    break
                print(f"Gemini : An error occurred: {e}")
                retry_count_gemini += 1
                time.sleep(sleep_time)
        retry_count += 1
        
        if error_flag:
            break

        #sntence_t
        query_embedding = snet_T.encode(gpt_ans)
        passage_embedding = snet_T.encode(gemini_ans)
        result = util.pytorch_cos_sim(query_embedding, passage_embedding)

        if result[0][0] < 0.8:
            continue
        else :
            break
    return result, gpt_ans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ranking_model', '-r', type=str, choices=['distil', 'deberta'], default='deberta')
    parser.add_argument('--Answer', '-a', type=bool, default=False)
    parser.add_argument('--our_or_llm', '-m', type=str, choices=['our', 'llm', 'fairytale'], default='our')
    parser.add_argument('--Event_count', '-c', type=int, default=2)
    parser.add_argument('--exampler', '-e', type=int, default=5)
    args = parser.parse_args()
    main(ranking_model = args.Ranking_model,
         answer = args.Answer,
         Event_count = args.Event_count,
         our_or_llm = args.our_or_llm, 
         exampler = args.exampler)