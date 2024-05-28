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
model = SentenceTransformer('bert-base-nli-mean-tokens')

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')
    return data.values

def save_csv(record, path):
    row = ['Paragraph', 'Context', 'Prediction', 'Reference', 'Input_text', 'Question type', 'Golden Answer', 'label', 'Correct ans ratio', 'Question_difficulty', 'SentenceBert', 'Event graph', 'Relation graph']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(record)):
            writer.writerow(record[i])

def main(ranking_model = 'deberta', Event_count = 2):
    dataset = get_data(f'csv/2_{ranking_model}_w_ans_pred_{Event_count}.csv')
    snet_T = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    record = []
    
    for idx, data in tqdm(enumerate(dataset)):
        # if idx > 5:
        #     break

        context = data[1]
        pred = data[2].split('[')[1:-1]
        ques = pred[0].split(']')[1]

        golden_ans = ""
        gemini_ans = ""
        gpt_ans = ""  
        
        retry_count = 0
        while retry_count < 10:
            # gpt_text = f'Without considering completeness and details, is this answer correct in responding to this question? Please answer "Yes" or "No".Answer : {gemini_ans} Question : {ques}, Context : {context}'
            # gpt_text = f'Is this answer correct in responding to this question? Please answer "Yes" or "No".Answer : {gemini_ans} Question : {ques}, Context : {context}'
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
                    print(f"An error occurred: {e}")
                    retry_count_gpt += 1
                    time.sleep(10)

            gemini_text = f'Extract the answer from the text and reply with just the answer. Respond in {len(gpt_ans.split())} words. Question : {ques}, Context : {context}'
            retry_count_gemini = 0
            while retry_count_gemini < 5:
                try:
                    gemini_ans = geminiapi(gemini_text, temperature=0)
                    if '\n' in gemini_ans:
                        raise ValueError("Response contains newline")
                    break 
                except Exception as e:
                    print(f"An error occurred: {e}")
                    retry_count_gemini += 1
                    time.sleep(10)

            retry_count += 1
            # bleu
            # result = metric.compute(predictions=[gemini_ans], references=[gpt_ans])

            #sntence_t
            # query_embedding = snet_T.encode(gpt_ans)
            # passage_embedding = snet_T.encode(gemini_ans)
            # result = util.dot_score(query_embedding, passage_embedding)

            #Cosine similarity
            embeddings1 = model.encode([gpt_ans])
            embeddings2 = model.encode([gemini_ans])
            result = cosine_similarity(embeddings1, embeddings2)
            if result[0][0] < 0.8:
                continue
            else :
                break
       
        data = list(data)
        if result[0][0] > 0.8:
            data.insert(6, gpt_ans)
        else:
            data.insert(6, "False, " + gpt_ans)
        
        record.append(data)
        save_csv(record, f'csv/3_question_w_golden_ans_{Event_count}.csv')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ranking_model', '-r', type=str, choices=['distil', 'deberta'], default='deberta')
    parser.add_argument('--Event_count', '-c', type=int, default=2)
    args = parser.parse_args()
    main(ranking_model = args.Ranking_model, Event_count = args.Event_count)