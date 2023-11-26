import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def read_data(path):
    data = pd.read_csv(path, index_col = False, encoding_errors = 'ignore')
    return data.values

def SentenceTransformer_eval(dataset):
    record = []
    score = 0
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    for data in tqdm(dataset):
        pred, tar =  data[-2], data[-1]
        query_embedding = model.encode(pred)
        passage_embedding = model.encode(tar)
        result = util.dot_score(query_embedding, passage_embedding)
        data = np.concatenate((data, [round(result[0][0].item(), 2)]))
        score += round(result[0][0].item(), 2)
        record.append(data)
    return record, score

def save_csv(dataset, path):
    row_1 = ['ID', "Story", "Question_predict", "Question_target", "Sentence_Bert_Score"]
    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row_1)
        for data in dataset:
            writer.writerow(data) 


if __name__ == '__main__':
    path = 'save_model/Event/tagging.csv'
    data = read_data(path)

    record, s_score = SentenceTransformer_eval(data)
    print("SentenceTransformer : ", round(s_score/len(record), 2))
    save_csv(record, 'score.csv')