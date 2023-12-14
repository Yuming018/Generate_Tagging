import pandas as pd
import numpy as np
import csv
import argparse
import evaluate
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer, util

metric = evaluate.load("bleu")

def read_data(path):
    data = pd.read_csv(path, index_col = False, encoding_errors = 'ignore')
    return data.values

def save_csv(dataset, path):
    row_1 = ['ID', "Story", "Question_predict", "Question_target", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "Sentence_Bert_Score"]
    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row_1)
        for data in dataset:
            writer.writerow(data) 

class eval_Realtion:
    def __init__(self, path) -> None:
        self.pred_dict, self.tar_dict = defaultdict(defaultdict), defaultdict(defaultdict)
        self.dataset = read_data(path)
        self.label = [['X attribute'], ['X intent', 'X reaction', 'Other reaction'], ['isBefore', 'the same', 'isAfter', 'X need', 'Effect on X',  'X want', 'Other want', 'Effect on other']]
        self.process_data()

    def process_data(self):
        for idx, data in tqdm(enumerate(self.dataset)):
            pred, tar =  data[2], data[3]
            pred = pred.split('[')[1:-1]
            tar = tar.split('[')[1:-1]
            self.pred_dict[idx]['data'] = data
            for p, t in zip(pred, tar):
                self.pred_dict[idx][p.split(']')[0]] = p.split(']')[1]
                self.tar_dict[idx][t.split(']')[0]] = t.split(']')[1]
    
    def process_label(self):
        pred, true = [], []
        for key in tqdm(self.pred_dict):
            for idx in range(len(self.label)):
                pred_type = self.pred_dict[key]['Relation'].split(' - ')[1]
                tar_type = self.tar_dict[key]['Relation'].split(' - ')[1]
                if pred_type[:-1] in self.label[idx]:
                    pred.append(idx+1)
                if tar_type[:-1] in self.label[idx]:
                    true.append(idx+1)
        print(len(true), len(pred))
        precision = precision_score(true, pred, average='micro')
        recall = recall_score(true, pred, average='micro')
        f1_s = f1_score(true, pred, average='micro')
        print(f'Micro-average Precision: {precision}')
        print(f'Micro-average Recall: {recall}')
        print(f'Micro-average F1_score: {f1_s}')
        print(classification_report(true, pred))
        print(confusion_matrix(true, pred))

    def bleu_eval(self):
        record = []
        bleu_score = [0, 0, 0, 0]
        for key in tqdm(self.pred_dict):
            pred = self.pred_dict[key]
            tar = self.tar_dict[key]
            keys_to_check = ['Relation', 'Event1', 'Event2']
            if not all(key in pred for key in keys_to_check) or not all(key in tar for key in keys_to_check):
                continue
            pred_label, true_label = 0, 0
            for idx in range(len(self.label)):
                pred_type = self.pred_dict[key]['Relation'].split(' - ')[1]
                tar_type = self.tar_dict[key]['Relation'].split(' - ')[1]
                if pred_type[:-1] in self.label[idx]:
                    pred_label = idx+1
                if tar_type[:-1] in self.label[idx]:
                    true_label = idx+1
            if pred_label == true_label:
                result, result2 = self.cal_result(pred, tar)
                bleu_1 = (round(result['precisions'][0], 2) + round(result2['precisions'][0], 2)) / 2
                bleu_2 = (round(result['precisions'][1], 2) + round(result2['precisions'][1], 2)) / 2
                bleu_3 = (round(result['precisions'][2], 2) + round(result2['precisions'][2], 2)) / 2
                bleu_4 = (round(result['precisions'][3], 2) + round(result2['precisions'][3], 2)) / 2
                data = np.concatenate((pred['data'], [bleu_1, bleu_2, bleu_3, bleu_4]))
                bleu_score[0] += bleu_1
                bleu_score[1] += bleu_2
                bleu_score[2] += bleu_3
                bleu_score[3] += bleu_4
                record.append(data)
            # else:
            #     data = np.concatenate((pred['data'], [0, 0, 0, 0]))
            
        return record, bleu_score
    
    def cal_result(self, pred_dict, tar_dict):
        result = metric.compute(predictions=[pred_dict['Event1']], references=[tar_dict['Event1']])
        result2 = metric.compute(predictions=[pred_dict['Event2']], references=[tar_dict['Event2']])
        result3 = metric.compute(predictions=[pred_dict['Event1']], references=[tar_dict['Event2']])
        result4 = metric.compute(predictions=[pred_dict['Event2']], references=[tar_dict['Event1']])
        if result['bleu'] + result2['bleu'] >= result3['bleu'] + result4['bleu']:
            return result, result2
        else:
            return result3, result4

    def SentenceTransformer_eval(self, dataset):
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
    
class eval_Event:
    def __init__(self) -> None:
        pass
    
    def bleu_eval(self, dataset):
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

    def SentenceTransformer_eval(self, dataset):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Type', '-t', type=str, default='Relation')
    parser.add_argument('--Generation', '-g', type=str, default='tagging')
    args = parser.parse_args()
    
    print('Type : ', args.Type)
    print('Generation : ', args.Generation, '\n')

    path = f'save_model/{args.Type}/{args.Generation}/'
    # data = read_data(path + f'{args.Generation}.csv')

    if args.Type == 'Event':
        eval = eval_Event(path + f'{args.Generation}.csv')
    elif args.Type == 'Relation':
        eval = eval_Realtion(path + f'{args.Generation}.csv')
        eval.process_label()

    record, bleu_score = eval.bleu_eval()
    record, s_score = eval.SentenceTransformer_eval(record)
    num = len(record)
    print("BLEU-1 : ", round(bleu_score[0]/num, 2))
    print("BLEU-2 : ", round(bleu_score[1]/num, 2))
    print("BLEU-3 : ", round(bleu_score[2]/num, 2))
    print("BLEU-4 : ", round(bleu_score[3]/num, 2))
    print("SentenceTransformer : ", round(s_score/num, 2))
    save_csv(record, path + 'score.csv')