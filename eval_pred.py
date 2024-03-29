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

    row_1 = ["Story", "Content", "Question_predict", "Question_target", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "Sentence_Bert_Score", "Overlap score"]
    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row_1)
        for data in dataset:
            writer.writerow(data) 

class eval_Realtion:
    def __init__(self, path) -> None:
        self.pred_dict, self.tar_dict = defaultdict(defaultdict), defaultdict(defaultdict)
        self.dataset = read_data(path)
        self.label = [['Other'], ['X attribute'], ['X intent', 'X reaction', 'Other reaction'], ['isBefore', 'the same', 'isAfter', 'X need', 'Effect on X',  'X want', 'Other want', 'Effect on other']]
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
                pred_type = self.pred_dict[key]['Relation 1'].split(' - ')[1]
                tar_type = self.tar_dict[key]['Relation 1'].split(' - ')[1]
                if pred_type[:-1] in self.label[idx]:
                    pred.append(idx+1)
                if tar_type[:-1] in self.label[idx]:
                    true.append(idx+1)
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
        keys_to_check = ['Relation 1', 'Event1', 'Event2']
        for idx in tqdm(self.pred_dict):
            pred = self.pred_dict[idx]
            tar = self.tar_dict[idx]
            if not all(key in pred for key in keys_to_check) or not all(key in tar for key in keys_to_check):
                continue
            if not pred['Event1'].replace(' ', "") or not pred['Event2'].replace(' ', ""):
                continue
            pred_label, true_label = 0, 0
            for label_idx in range(len(self.label)):
                pred_type = self.pred_dict[idx]['Relation 1'].split(' - ')[1]
                tar_type = self.tar_dict[idx]['Relation 1'].split(' - ')[1]
                if pred_type[:-1] in self.label[label_idx]:
                    pred_label = label_idx+1
                if tar_type[:-1] in self.label[label_idx]:
                    true_label = label_idx+1
            # if pred_label == true_label:
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

    def Sentence_Overlap_eval(self, record):
        # print(len(record))
        keys_to_check = ['Relation 1', 'Event1', 'Event2']
        sent_score, overlap_score, count = 0, 0, 0
        for idx in tqdm(self.pred_dict):
            pred = self.pred_dict[idx]
            tar = self.tar_dict[idx]
            if not all(key in pred for key in keys_to_check) or not all(key in tar for key in keys_to_check):
                continue
            if not pred['Event1'].replace(' ', "") or not pred['Event2'].replace(' ', ""):
                continue
            """
            計算 senteceTransformer score
            """
            result = self.cal_sentence(pred, tar)
            sent_score += result
            record[count] = np.append(record[count], result)
            
            """
            計算 overlap ration score
            """
            result = self.cal_overlap(pred, tar)
            overlap_score += result
            record[count] = np.append(record[count], result)

            count += 1
            
        return record, sent_score, overlap_score

    def cal_sentence(self, pred_dict, tar_dict):
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        query_embedding_1 = model.encode(pred_dict['Event1'])
        passage_embedding_1 = model.encode(tar_dict['Event1'])
        query_embedding_2 = model.encode(pred_dict['Event2'])
        passage_embedding_2 = model.encode(tar_dict['Event2'])

        result = util.dot_score(query_embedding_1, passage_embedding_1)
        result2 = util.dot_score(query_embedding_1, passage_embedding_2)
        result3 = util.dot_score(query_embedding_2, passage_embedding_1)
        result4 = util.dot_score(query_embedding_2, passage_embedding_2)

        return max(round(result[0][0].item(), 2) + round(result4[0][0].item(), 2), round(result2[0][0].item(), 2) + round(result3[0][0].item(), 2)) /2

    def cal_overlap(self, pred_dict, tar_dict):
        lcs_length = self.longest_common_subsequence(pred_dict['Event1'], tar_dict['Event1'])
        lcs_length_2 = self.longest_common_subsequence(pred_dict['Event1'], tar_dict['Event2'])
        lcs_length_3 = self.longest_common_subsequence(pred_dict['Event2'], tar_dict['Event1'])
        lcs_length_4 = self.longest_common_subsequence(pred_dict['Event2'], tar_dict['Event2'])

        overlap_ratio   = lcs_length   / (len(pred_dict['Event1']) + len(tar_dict['Event1']) - lcs_length)
        overlap_ratio_2 = lcs_length_2 / (len(pred_dict['Event1']) + len(tar_dict['Event2']) - lcs_length_2)
        overlap_ratio_3 = lcs_length_3 / (len(pred_dict['Event2']) + len(tar_dict['Event1']) - lcs_length_3)
        overlap_ratio_4 = lcs_length_4 / (len(pred_dict['Event2']) + len(tar_dict['Event2']) - lcs_length_4)

        return max(round(overlap_ratio, 2) + round(overlap_ratio_4, 2), round(overlap_ratio_2, 2) + round(overlap_ratio_3, 2)) /2

    def longest_common_subsequence(self, pred, tar):
        m = len(pred)
        n = len(tar)
        # Create a table to store lengths of LCS
        lcs_table = [[0] * (n + 1) for _ in range(m + 1)]

        # Building the LCS table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == tar[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

        # Length of LCS is the value at the bottom-right cell of the table
        return lcs_table[m][n]
    
class eval_Event:
    def __init__(self, path) -> None:
        self.pred_dict, self.tar_dict = defaultdict(defaultdict), defaultdict(defaultdict)
        self.all_type_dict = dict()
        self.repeat_label = { 'Trigger_Word' : 'Trigger_word',
                       'Emotion' : 'Emotion_Type'}
        self.dataset = read_data(path)
        self.process_data()

    def process_data(self):
        type_idx = 0
        for idx, data in tqdm(enumerate(self.dataset)):
            
            pred, tar =  data[2], data[3]
            pred = pred.split('[')[1:-1]
            tar = tar.split('[')[1:-1]
            self.pred_dict[idx]['data'] = data
            for p in pred:
                label = p.split(']')[0]
                if label != 'Event' and label not in self.repeat_label and label not in self.all_type_dict:
                    self.all_type_dict[label] = type_idx
                    type_idx += 1
                self.pred_dict[idx][label] = p.split(']')[1]
            for t in tar:
                label = t.split(']')[0]
                if label != 'Event' and label not in self.repeat_label and label not in self.all_type_dict:
                    self.all_type_dict[label] = type_idx
                    type_idx += 1
                self.tar_dict[idx][label] = t.split(']')[1]
        self.all_type_dict['other'] = type_idx
    
    def NER_eval(self):
        # get results classification
        gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
        keys_to_check = ['data', 'Event']
        for idx in tqdm(self.pred_dict):
            predict = self.pred_dict[idx]
            tar = self.tar_dict[idx]

            for key, _ in tar.items():
                if key not in keys_to_check:
                    pred_arg_n += 1
            
            for key, _ in predict.items():
                if key not in keys_to_check:
                    gold_arg_n += 1
            # print(pred_in_gold_n, gold_in_pred_n)

            for key, entity in tar.items():
                if key not in keys_to_check:
                    bleu_score, max_type = 0, self.all_type_dict['other']
                    entity_len = len(entity.split())
                    
                    for p_key, p_entity in predict.items():
                        if p_key not in keys_to_check:
                            result = metric.compute(predictions=[entity], references=[p_entity])
                            if result['precisions'][min(4, entity_len)-1] > bleu_score:
                                bleu_score = result['precisions'][min(4, entity_len)-1]
                                max_type = p_key
                    
                    if bleu_score == 1:
                        if max_type in self.repeat_label:
                            max_type = self.repeat_label[max_type]
                        if key in self.repeat_label:
                            key = self.repeat_label[key]
                        if max_type == key:
                            gold_in_pred_n += 1
            
            for key, entity in predict.items():
                if key not in keys_to_check:
                    bleu_score, max_type = 0, self.all_type_dict['other']
                    entity_len = len(entity.split())
                    
                    for t_key, t_entity in tar.items():
                        if t_key not in keys_to_check:
                            result = metric.compute(predictions=[entity], references=[t_entity])
                            if result['precisions'][min(4, entity_len)-1] > bleu_score:
                                bleu_score = result['precisions'][min(4, entity_len)-1]
                                max_type = t_key
                    
                    if bleu_score == 1:
                        if max_type in self.repeat_label:
                            max_type = self.repeat_label[max_type]
                        if key in self.repeat_label:
                            key = self.repeat_label[key]
                        if max_type == key:
                            pred_in_gold_n += 1

        if pred_arg_n != 0:
            prec_c = 100.0 * pred_in_gold_n / pred_arg_n
        else:
            prec_c = 0
        if gold_arg_n != 0:
            recall_c = 100.0 * gold_in_pred_n / gold_arg_n
        else:
            recall_c = 0
        if prec_c or recall_c:
            f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
        else:
            f1_c = 0
        
        
        # get results identification
        gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
        for idx in tqdm(self.pred_dict):
            predict = self.pred_dict[idx]
            tar = self.tar_dict[idx]

            for key, _ in tar.items():
                if key not in keys_to_check:
                    pred_arg_n += 1
            
            for key, _ in predict.items():
                if key not in keys_to_check:
                    gold_arg_n += 1

            for key, entity in tar.items():
                if key not in keys_to_check:
                    bleu_score = 0
                    entity_len = len(entity.split())
                    
                    for p_key, p_entity in predict.items():
                        if p_key not in keys_to_check:
                            result = metric.compute(predictions=[entity], references=[p_entity])
                            if result['precisions'][min(4, entity_len)-1] > bleu_score:
                                bleu_score = result['precisions'][min(4, entity_len)-1]
                    
                    if bleu_score == 1:
                        gold_in_pred_n += 1
            
            for key, entity in predict.items():
                if key not in keys_to_check:
                    bleu_score = 0
                    entity_len = len(entity.split())
                    
                    for t_key, t_entity in tar.items():
                        if t_key not in keys_to_check:
                            result = metric.compute(predictions=[entity], references=[t_entity])
                            if result['precisions'][min(4, entity_len)-1] > bleu_score:
                                bleu_score = result['precisions'][min(4, entity_len)-1]
                    
                    if bleu_score == 1:
                        pred_in_gold_n += 1

        if pred_arg_n != 0:
            prec_i = 100.0 * pred_in_gold_n / pred_arg_n
        else:
            prec_i = 0
        if gold_arg_n != 0:
            recall_i = 100.0 * gold_in_pred_n / gold_arg_n
        else:
            recall_i = 0
        if prec_i or recall_i:
            f1_i = 2 * prec_i * recall_i / (prec_i + recall_i)
        else:
            f1_i = 0
        
        return f1_i, f1_c
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Type', '-t',
                        choices=["Relation", "Event"],
                        type=str,
                        default='Event')
    parser.add_argument('--Generation', '-g',
                        choices=["tagging", "question"],
                        type=str,
                        default='tagging')
    parser.add_argument('--Model', '-m',
                        choices=['Mt0', 'T5', 'Bart', 'roberta', 'gemma', 'flant5'],
                        type=str,
                        default='Mt0')
    args = parser.parse_args()
    
    print('Type : ', args.Type)
    print('Generation : ', args.Generation)
    print('Model : ', args.Model, '\n')

    path = f'save_model/{args.Type}/{args.Generation}/{args.Model}/'
    if args.Type == 'Event':
        eval = eval_Event(path + f'{args.Generation}.csv')
        f1_i, f1_c = eval.NER_eval()

        print('f1_c : ', f1_c)
        print('f1_i : ', f1_i)   
    elif args.Type == 'Relation':
        eval = eval_Realtion(path + f'{args.Generation}.csv')
        # eval.process_label()
        record, bleu_score = eval.bleu_eval()
        record, s_score, o_score = eval.Sentence_Overlap_eval(record)
        
        num = len(record)
        print("BLEU-1 : ", round(bleu_score[0]/num, 2))
        print("BLEU-2 : ", round(bleu_score[1]/num, 2))
        print("BLEU-3 : ", round(bleu_score[2]/num, 2))
        print("BLEU-4 : ", round(bleu_score[3]/num, 2))
        print("SentenceTransformer : ", round(s_score/len(record), 2))
        print("Overlap Ratio : ", round(o_score/len(record), 2))
        save_csv(record, path + 'score.csv')