import pandas as pd
from collections import defaultdict, Counter
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')  # 填充空值
    return data.values

def plt_show(data):
    plt.hist(data, bins=30, edgecolor='k', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def cal_score(score_list, score_type_list, dataset, our_or_llm, answer):
    if our_or_llm == 'our':
        begin_idx, end_idx = 8, 11
    elif our_or_llm == 'fairytale':
        begin_idx, end_idx = 4, 7

    for data in dataset:
        if our_or_llm == 'fairytale':
            if data[-1] in ['character', 'action', 'setting', 'feeling']:
                event_or_relation = 'Event'
            else:
                event_or_relation = 'Relation'
            question_type = data[-1] 
        else:
            event_or_relation = data[5]
            question_type = data[6]

        golden_ans = data[7]
        if answer:
            temp = data[2].split('[')[1:-1]
            ans = temp[1].split(']')[1]
        else:
            temp = data[4].split('[')[1:-1]
            ans = temp[-1].split(']')[1]
        if golden_ans == ans:
            answer_source = 'default'
        else :
            answer_source = 'LLM'

        for ans in data[begin_idx:end_idx]:
            score = ans.split('(')[-1]
            score = score.split(')')[0]
            score_list[event_or_relation].append(float(score))
            score_type_list[question_type].append(float(score))
            score_list[event_or_relation + '_' + answer_source].append(float(score))
    
    return score_list, score_type_list

def cal_correct_ans(dataset, our_or_llm, score_table, answer, correct_ratio):
    record = Counter()
    question_range = defaultdict(Counter)
    for data in dataset:
        if our_or_llm == 'our':
            question_type_idx = 5
            begin_idx, end_idx = 8, 11
        elif our_or_llm == 'fairytale':
            question_type_idx = -1
            begin_idx, end_idx = 4, 7
        
        question_type = data[question_type_idx]

        if our_or_llm == 'fairytale':
            threshold = 0.5
        else:
            golden_ans = data[7]
            if answer:
                temp = data[2].split('[')[1:-1]
                ans = temp[1].split(']')[1]
            else:
                temp = data[4].split('[')[1:-1]
                ans = temp[-1].split(']')[1]
            if golden_ans == ans:
                answer_source = 'default'
            else :
                answer_source = 'LLM'
            # threshold = score_table[question_type + '_' + answer_source + '_median']
            threshold = 0.5

        correce_ratio = 0
        for idx, ans in enumerate(data[begin_idx:end_idx]):
            score = ans.split('(')[-1]
            score = score.split(')')[0]
            if float(score) > threshold:
                correce_ratio += 1 
                correct_ratio[idx] +=1
        
        correct_ratio['total'] +=1
        
        if correce_ratio == 3:
            q_defficult = 'very Easy'
        elif correce_ratio == 2:
            q_defficult = 'Easy'
        elif correce_ratio == 1 :
            q_defficult = 'Medium'
        elif correce_ratio == 0:
            q_defficult = 'Hard'
        
        record[q_defficult] += 1

        if our_or_llm == 'our':
            if data[-4] < 5:
                q_range = 'Near'
            elif data[-4] < 12:
                q_range = 'Moderate'
            else:
                q_range = 'Far'
            question_range[q_range][q_defficult] += 1
    
    return record, question_range, correct_ratio

def main(answer = False):
    if answer:
        dataset_2 = []
        dataset_2 = get_data(f'csv/4_w_ans_correct_ratio_2.csv')
        dataset_3 = get_data(f'csv/4_w_ans_correct_ratio_3.csv')
        dataset_4 = get_data(f'csv/4_w_ans_correct_ratio_4.csv')
    elif not answer:
        dataset_2 = get_data(f'csv/4_wo_ans_correct_ratio_2_MiniLM_50.csv')
        dataset_3 = get_data(f'csv/4_wo_ans_correct_ratio_3_MiniLM_50.csv')
        dataset_4 = get_data(f'csv/4_wo_ans_correct_ratio_4_MiniLM_50.csv')
    fairytale_dataset = get_data(f'csv/4_fairytale_correct_ratio.csv')
    
    score_list, score_type_list = defaultdict(list), defaultdict(list)
    score_list, score_type_list = cal_score(score_list, score_type_list, dataset_2, our_or_llm = 'our', answer=answer)
    score_list, score_type_list = cal_score(score_list, score_type_list, dataset_3, our_or_llm = 'our', answer=answer)
    score_list, score_type_list = cal_score(score_list, score_type_list, dataset_4, our_or_llm = 'our', answer=answer)
    # score_list, score_type_list = cal_score(score_list, score_type_list, fairytale_dataset, our_or_llm = 'fairytale', answer=answer)

    score_table = dict()
    for type in score_type_list:
        score_table[type] = round(np.median(score_type_list[type]), 2)
    
    for type in score_list:
        average = np.mean(score_list[type])
        median = np.median(score_list[type])
        std_dev = np.std(score_list[type])
        print(f'{type} average score: ', average)
        print(f'{type} median score: ', median)
        print(f'{type} std: ', std_dev, '\n')
        score_table[type + '_average'] = round(average, 2)
        score_table[type + '_median'] = round(median, 2)
        score_table[type + '_std_dev'] = round(std_dev, 2)

    correct_ratio = Counter()
    stat_type_2, stat_range_2, correct_ratio =  cal_correct_ans(dataset_2, our_or_llm = 'our', score_table = score_table, answer=answer, correct_ratio= correct_ratio)
    stat_type_3, stat_range_3, correct_ratio =  cal_correct_ans(dataset_3, our_or_llm = 'our', score_table = score_table, answer=answer, correct_ratio= correct_ratio)
    stat_type_4, stat_range_4, correct_ratio =  cal_correct_ans(dataset_4, our_or_llm = 'our', score_table = score_table, answer=answer, correct_ratio= correct_ratio)
    print('\n', correct_ratio)
    print('T5 small : ', round(correct_ratio[0]/correct_ratio['total'], 2))
    print('T5 base : ', round(correct_ratio[1]/correct_ratio['total'], 2))
    print('T5 large : ', round(correct_ratio[2]/correct_ratio['total'], 2))
    correct_ratio = Counter()
    fairytale_stat, _        , correct_ratio = cal_correct_ans(fairytale_dataset, our_or_llm = 'fairytale', score_table = score_table, answer=answer, correct_ratio= correct_ratio)
    print('\n', correct_ratio)
    print('T5 small : ', round(correct_ratio[0]/correct_ratio['total'], 2))
    print('T5 base : ', round(correct_ratio[1]/correct_ratio['total'], 2))
    print('T5 large : ', round(correct_ratio[2]/correct_ratio['total'], 2))
    """
    統計 Event 數量多寡影響問題難易度
    """
    
    print('FairytaleQA :', fairytale_stat)
    print('2 Events :', stat_type_2)
    print('3 Events :', stat_type_3)
    print('4 Events :', stat_type_4)

    

    """
    統計 Range 距離遠近影響問題難易度
    """
    record = defaultdict(Counter)
    for type in stat_range_2:
        for level in stat_range_2[type]:
            record[type][level] += stat_range_2[type][level]
    
    for type in stat_range_3:
        for level in stat_range_3[type]:
            record[type][level] += stat_range_3[type][level]
    
    for type in stat_range_4:
        for level in stat_range_4[type]:
            record[type][level] += stat_range_4[type][level]

    print('\n', record)

    # plt_show(score_list['Event'])
    # plt_show(score_list['Relation'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Answer', '-a', type=bool, default=False)
    args = parser.parse_args()
    main(answer=args.Answer)