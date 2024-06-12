import pandas as pd
from collections import defaultdict, Counter

ques_idx = 2
event_idx = 5
relation_idx = 6

legal_tagging = ['Causal Effect',
            'Temporal',
            'Coreference'
            'State',
            'Action'
]

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')  # 填充空值
    return data.values

def count_attribute_tagging(path, question, attribute_tagging, header):
    data = get_data(path)
    dic = set()
    
    for i in range(len(data)):
        if data[i][event_idx]:
            label = data[i][event_idx].split(' - ')[0]
        elif data[i][relation_idx]:
            label = data[i][relation_idx].split(' - ')[0]
        attribute = data[i][0]
        attribute_tagging[attribute][label] += 1
        
        if data[i][ques_idx] not in dic:
            question[attribute] += 1
        dic.add(data[i][ques_idx])

    # for attribute in attribute_tagging:
    #     print('\n',attribute)
    #     plt_show(attribute_tagging[attribute], header)
    
    return question, attribute_tagging, header

def plt_show(count, header):
    count = sorted(count.items(), key = lambda x: x[1])
    for label, data in count:
        header[label] = data

    for label, data in header.items():
        print(label, data)
    
if __name__ == '__main__':
    question = Counter()
    header = Counter()
    attribute_tagging = defaultdict(Counter)
    question, attribute_tagging, header = count_attribute_tagging('data/train.csv', question, attribute_tagging, header)
    question, attribute_tagging, header = count_attribute_tagging('data/valid.csv', question, attribute_tagging, header)
    question, attribute_tagging, header = count_attribute_tagging('data/test.csv', question, attribute_tagging, header)

    print(question)
    total = sum(question.values())
    print(total)
    input()

    for attribute in attribute_tagging:
        print('\n',attribute)
        plt_show(attribute_tagging[attribute], header)