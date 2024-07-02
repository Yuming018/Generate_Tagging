import pandas as pd
from collections import defaultdict, Counter

ques_idx = 2
event_idx = 5
relation_idx = 6

legal_tagging = ['Causal Effect',
            'Temporal',
            # 'Coreference',
            'State',
            'Action',
]

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')  # 填充空值
    return data.values

def count_attribute_tagging(path, question, attribute_tagging):
    data = get_data(path)
    dic = set()
    ques_set = set()
    
    for i in range(len(data)):
        if data[i][event_idx]:
            label = data[i][event_idx].split(' - ')[0]
            # attribute_tagging[attribute]['Event'] += 1
        elif data[i][relation_idx]:
            label = data[i][relation_idx].split(' - ')[0]
            # attribute_tagging[attribute]['Relation'] += 1
        attribute = data[i][0]
        ques = data[i][2]
        attribute_tagging[attribute][label] += 1
        if ques not in ques_set:
            attribute_tagging[attribute]['all'] += 1
            ques_set.add(ques)
        
        if data[i][ques_idx] not in dic:
            question[attribute] += 1
        dic.add(data[i][ques_idx])
    
    return question, attribute_tagging

def show_data(count):
    header = Counter()

    count = sorted(count.items(), key = lambda x: x[1])
    for label, data in count:
        header[label] = data

    
    total = header['Causal Effect'] + header['State'] + header['Action'] + header['Temporal']
    print('All:', total)
    for label, data in header.items():
        for tag in legal_tagging:
            if tag in label:
                print(label, round(data/total, 2), data)
        # if label in legal_tagging:
        #     print(label, round(data/header['all'], 2))
        #     # print(label, data)

def statistic_answer(path, ans_tagging, temp):
    data = get_data(path)
    para_set = set()
    match_arg, match_type = "", ""
    match_ratio = 0
    current_ques = data[0][2]
    
    for story_idx in range(len(data)):
        ques = data[story_idx][2]

        if data[story_idx][2] not in para_set:
            temp[data[story_idx][0]] += 1
            para_set.add(data[story_idx][2])

        if current_ques != ques:
            # ans_tagging[data[story_idx-1][0]][match_arg] += 1
            # ans_tagging[data[story_idx-1][0]]['all'] += 1
            ans_tagging[match_type][match_arg] += 1
            ans_tagging[match_type]['all'] += 1
            match_arg, match_type = "", ""
            match_ratio = 0
            current_ques = ques

        if data[story_idx][event_idx]:
            label = data[story_idx][event_idx].split(' - ')[0]
        elif data[story_idx][relation_idx]:
            label = data[story_idx][relation_idx].split(' - ')[0]
        
        if label not in legal_tagging:
            continue
        
        arg = []
        for idx in range(7, len(data[story_idx])):
            if data[story_idx][idx] != '':
                left_parenthesis_index = data[story_idx][idx].rfind('(')
                if label in ['State', 'Action']:
                    argument = data[story_idx][idx][:left_parenthesis_index].split(' - ')
                    arg.append([argument[0], " - ".join(argument[1:])])
                else:
                    if idx == 9:
                        break
                    arg.append([f'Arg{idx-6}', data[story_idx][idx][:left_parenthesis_index]])
    
        ans = data[story_idx][3]
        for argument in arg:
            ratio = longest_common_subsequence(ans, argument[1])
            if match_ratio < ratio:
                match_ratio = ratio
                match_arg = argument[0]
                match_type = label
        #     print(ans, argument[1])
        # print('\n', match_ratio, match_arg)
        # input()
        
    return ans_tagging, temp

def longest_common_subsequence(text, text2):
    text1_len = len(text)
    text2_len = len(text2)
    # Create a table to store lengths of LCS
    lcs_table = [[0] * (text2_len + 1) for _ in range(text1_len + 1)]

    # Building the LCS table
    for i in range(1, text1_len + 1):
        for j in range(1, text2_len + 1):
            if text[i - 1] == text2[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    
    return lcs_table[text1_len][text2_len] / (text1_len + text2_len - lcs_table[text1_len][text2_len])


if __name__ == '__main__':
    question = Counter()
    attribute_tagging = defaultdict(Counter)
    question, attribute_tagging = count_attribute_tagging('data/train.csv', question, attribute_tagging)
    question, attribute_tagging = count_attribute_tagging('data/valid.csv', question, attribute_tagging)
    question, attribute_tagging = count_attribute_tagging('data/test.csv', question, attribute_tagging)

    total = sum(question.values())
    # print(question)
    # print("Question total : ", total)
    # input()

    for attribute in attribute_tagging:
        print('\n',attribute)
        show_data(attribute_tagging[attribute])

    # ans_tagging = defaultdict(Counter)
    # temp = Counter()
    # ans_tagging, temp = statistic_answer('data/train.csv', ans_tagging, temp)
    # ans_tagging, temp = statistic_answer('data/valid.csv', ans_tagging, temp)
    # ans_tagging, temp = statistic_answer('data/test.csv', ans_tagging, temp)

    # for attribute in ans_tagging:
    #     total = ans_tagging[attribute]['all']
    #     print(attribute, total)
    #     for label in ans_tagging[attribute]:
    #         print(label, ans_tagging[attribute][label], round(ans_tagging[attribute][label]/total, 3))
    #     print('\n')
    # print(temp)
    # print(sum(temp.values()))