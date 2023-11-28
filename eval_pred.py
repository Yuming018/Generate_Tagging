import pandas as pd
import numpy as np
import csv
import evaluate
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

metric = evaluate.load("bleu")

def read_data(path):
    data = pd.read_csv(path, index_col = False, encoding_errors = 'ignore')
    return data.values

def bleu_eval(dataset):
    record = []
    bleu_score = [0, 0, 0, 0]
    for data in tqdm(dataset):
        pred, tar =  data[2], data[3]
        result = metric.compute(predictions=[pred], references=[tar])
        bleu_1 = round(result['precisions'][0], 2)
        bleu_2 = round(result['precisions'][1], 2)
        bleu_3 = round(result['precisions'][2], 2)
        bleu_4 = round(result['precisions'][3], 2)
        data = np.concatenate((data, [bleu_1, bleu_2, bleu_3, bleu_4]))
        bleu_score[0] += bleu_1
        bleu_score[1] += bleu_2
        bleu_score[2] += bleu_3
        bleu_score[3] += bleu_4
        record.append(data)
    return record, bleu_score

def SentenceTransformer_eval(dataset):
    record = []
    score = 0
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    for data in tqdm(dataset):
        pred, tar =  data[2], data[3]
        query_embedding = model.encode(pred)
        passage_embedding = model.encode(tar)
        result = util.dot_score(query_embedding, passage_embedding)
        data = np.concatenate((data, [round(result[0][0].item(), 2)]))
        score += round(result[0][0].item(), 2)
        record.append(data)
    return record, score

def save_csv(dataset, path):
    row_1 = ['ID', "Story", "Question_predict", "Question_target", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "Sentence_Bert_Score"]
    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row_1)
        for data in dataset:
            writer.writerow(data) 

if __name__ == '__main__':
    path = 'save_model/Event/tagging/'
    data = read_data(path + 'tagging.csv')

    record, bleu_score = bleu_eval(data)
    record, s_score = SentenceTransformer_eval(record)
    num = len(record)
    print("BLEU-1 : ", round(bleu_score[0]/num, 2))
    print("BLEU-2 : ", round(bleu_score[1]/num, 2))
    print("BLEU-3 : ", round(bleu_score[2]/num, 2))
    print("BLEU-4 : ", round(bleu_score[3]/num, 2))
    print("SentenceTransformer : ", round(s_score/num, 2))
    save_csv(record, path + 'score.csv')