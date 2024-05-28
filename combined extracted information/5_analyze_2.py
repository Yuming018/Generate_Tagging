import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import argparse

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')  # 填充空值
    return data.values

def analyze(path, Event_count):
    dataset = get_data(path)
    total = len(dataset)
    not_answer = 0
    for data in dataset:
        if data[7] == 'Can not answer':
            not_answer += 1
    
    print(f'{Event_count} Event')
    print('Total : ', total)
    print('Not answer : ', not_answer)
    print('can not answer ratio: ', round((not_answer/total), 2), '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, choices=['deberta', 'distil'], default='deberta')
    args = parser.parse_args()

    print('Model : ', args.model)
    analyze(f'csv/3_question_w_golden_ans_2.csv', Event_count = 2)
    analyze(f'csv/3_question_w_golden_ans_3.csv', Event_count = 3)
    analyze(f'csv/3_question_w_golden_ans_4.csv', Event_count = 4)