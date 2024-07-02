import pandas as pd
import argparse
from collections import Counter

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')  
    return data.values

def count_not_answer(model, answer, Event_count):
    paragraph = set()
    if answer :
        path = f'csv/2_{model}_w_ans_pred_{Event_count}.csv'
    else:
        path = f'csv/2_{model}_pred_{Event_count}.csv'
    print(path)
    record = get_data(path)
    stat = Counter()
    for data in record:
        paragraph.add(data[0])
        if data[7] == 'Can answer':
            stat['can answer'] += 1
        elif data[7] == 'Can not answer':
            stat['can not answer'] += 1
    
    total = stat['can answer'] + stat['can not answer']
    print('Paragraph : ', len(paragraph))
    print('Can answer : ', round(stat['can answer']/total, 2), stat['can answer'])
    print('Can not answer : ', round(stat['can not answer']/total, 2), stat['can not answer'])
    return

def count_golden_ans(answer, Event_count):
    if answer :
        path = f'csv/3_question_ans_w_golden_ans_{Event_count}.csv'
    else:
        path = f'csv/3_question_w_golden_ans_{Event_count}.csv'
    print(path)

    record = get_data(path)
    stat = Counter()
    for data in record:
        if data[8] == 'Can not answer':
            continue
        golden_ans = data[6]
        temp = data[4].split('[')[1:-1]
        ref_ans = temp[-1].split(']')[1]
        
        if golden_ans == ref_ans:
            stat['our'] += 1
        elif 'False' in golden_ans:
            stat['not'] += 1
        else:
            stat['llm'] += 1
    
    total = stat['our'] + stat['llm'] + stat['not']
    print(total)
    print('Our answer : ', round(stat['our']/total, 2), stat['our'])
    print('llm answer : ', round(stat['llm']/total, 2), stat['llm'])
    print('Not answer : ', round(stat['not']/total, 2), stat['not'])
    return

def count_all_paragraph():
    path = '../data/test.csv'
    record = get_data(path)

    paragraph = set()
    for data in record:
        para = "-".join(data[4].split('-')[:-1])
        paragraph.add(para)
    print('all paragraph : ', len(paragraph))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', '-m',
                        choices=['distil', 'deberta'],
                        type=str,
                        default='deberta')
    parser.add_argument('--Answer', '-a',
                        type=bool,
                        default=False)
    parser.add_argument('--Event_count', '-c', 
                        type=int, 
                        default=2)
    args = parser.parse_args()

    # count_not_answer(args.Model, args.Answer, args.Event_count)

    count_golden_ans(args.Answer, args.Event_count)

    # count_all_paragraph()