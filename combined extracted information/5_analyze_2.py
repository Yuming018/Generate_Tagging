import pandas as pd
from collections import Counter
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import evaluate
from tqdm import tqdm

sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
snet_T = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
metric = evaluate.load("bleu")

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')  
    return data.values

def analyze(path, Event_count):
    dataset = get_data(path)
    total = len(dataset)
    not_answer = 0
    eval_metric = Counter()
    quest_set = set()

    for data in tqdm(dataset):
        pred = data[2].split('[')[1:-1]
        ques = pred[0].split(']')[1]
        ans = pred[1].split(']')[1]
        golden_ans = data[6]
        quest_set.add(ques)

    #     embeddings1 = sentence_model.encode([ans])
    #     embeddings2 = sentence_model.encode([golden_ans])
    #     result = cosine_similarity(embeddings1, embeddings2)
    #     eval_metric['cosine'] += result[0][0]

    #     query_embedding = snet_T.encode(ans)
    #     passage_embedding = snet_T.encode(golden_ans)
    #     result = util.dot_score(query_embedding, passage_embedding)
    #     eval_metric['sentence'] += result[0][0].item()

    #     result = metric.compute(predictions=[ans], references=[golden_ans])
    #     eval_metric['bleu_1'] += result['precisions'][0]
    #     eval_metric['bleu_2'] += result['precisions'][1]
    #     eval_metric['bleu_3'] += result['precisions'][2]
    #     eval_metric['bleu_4'] += result['precisions'][3]

    #     if data[7] == 'Can not answer':
    #         not_answer += 1
    
    # print(f'{Event_count} Event')
    # print('Total : ', total)
    # print('Not answer : ', not_answer)
    # print('can not answer ratio: ', round((not_answer/total), 2), '\n')
    # print('Evaluate metric')
    # for e_metric in eval_metric:
    #     print(e_metric, " : ", round(eval_metric[e_metric]/total, 2), '\n')

    return quest_set

def analyze_llm(path):
    dataset = get_data(path)
    record = Counter()
    for data in dataset:
        record[data[2]] += 1
        record[data[2] + '_' + str(data[-1])] += 1
    
    print(record)
    print('Very Easy : ', record['Very Easy_3']/record['Very Easy'])
    print('Simple : ', record['Simple_2']/record['Simple'])
    print('Medium : ', record['Medium_1']/record['Medium'])
    print('Difficult : ', record['Difficult_0']/record['Difficult'])

def analyze_dataset(path):
    dataset = get_data(path)
    record = Counter()
    for data in dataset:
        record[data[2]] += 1
        record[data[2] + '_' + str(data[-1])] += 1
    
    print(record)
    print('Very Easy : ', record['Very Easy_3']/record['Very Easy'])
    print('Simple : ', record['Simple_2']/record['Simple'])
    print('Medium : ', record['Medium_1']/record['Medium'])
    print('Difficult : ', record['Difficult_0']/record['Difficult'])

def cal_ques_intersection(set_1, set_2):
    intersection = set_1 & set_2

    print(len(set_1))
    print(len(set_2))
    print('intersection : ',len(intersection), '\n')
    return

def cal_new_quest_set(set_2, set_3, set_4):
    union_set = set_2 | set_3
    new_elements_in_set3 = set_4 - union_set

    print('New question : ',len(new_elements_in_set3))
    # for ques in new_elements_in_set3:
    #     print(ques)
    return

if __name__ == '__main__':

    quest_set_4 = analyze_llm(f'csv/4_llm_correct_ratio_5_together.csv')

    # quest_set_2 = analyze(f'csv/3_question_w_golden_ans_2.csv', Event_count = 2)
    # quest_set_3 = analyze(f'csv/3_question_w_golden_ans_3.csv', Event_count = 3)
    # quest_set_4 = analyze(f'csv/3_question_w_golden_ans_4.csv', Event_count = 4)

    # cal_ques_intersection(quest_set_2, quest_set_3)
    # cal_ques_intersection(quest_set_3, quest_set_4)
    # cal_new_quest_set(quest_set_2, quest_set_3, quest_set_4)

    