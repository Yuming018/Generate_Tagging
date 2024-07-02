import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import argparse

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')  # 填充空值
    return data.values

def display_question_defficulty_histogram(data, categories, plt_name, Event_count, x_label):  
    bar_width = 0.15
    colors = ['blue', 'orange', 'green', 'red']

    # 計算每個問題類別的索引位置
    group_labels = list(data.keys())
    index = range(len(group_labels))

    max_value = 0
    for category in categories:
        category_values = [int(data[group][category]*100) for group in group_labels]
        max_value = max(max_value, max(category_values))
    y_max = (max_value // 10 + 1) * 10

    for i, category in enumerate(categories):
        category_values = [int(data[group][category]*100) for group in group_labels]
        plt.bar(
            [j + i * bar_width for j in index],
            category_values,
            width=bar_width,
            color=colors[i],
            label=category,
            edgecolor='black'
        )
        # 添加標籤
        for j, value in enumerate(category_values):
            plt.text(j + i * bar_width, value + 0.01, f'{int(value)}', ha='center', va='bottom')

    # 設置 x 軸標籤，使其與問題類別對齊
    plt.xticks([r + bar_width * 1.5 for r in range(len(group_labels))], group_labels)

    # 設置 xy 軸範圍
    plt.xlim(-0.5, len(group_labels) - 0.5 + bar_width * 3)
    plt.ylim(0, y_max)

    plt.title(f'Analyze {Event_count}')
    plt.xlabel(x_label)
    plt.ylabel('ratio of questions')
    plt.legend()
    plt.savefig(f'pictures/{plt_name}.png')
    # plt.show()
    plt.clf()

def display_inner_ratio_histogram(data, categories, plt_name, Event_count):  
    bar_width = 0.15
    difficulty_levels = ['very Easy', 'Easy', 'Medium', 'Hard']
    sub_colors = ['blue', 'orange', 'green', 'red']

    index = range(len(difficulty_levels))
    max_value = 0

    for range_label in data:
        for difficulty in difficulty_levels:
            total = sum(data[range_label][category][difficulty] for category in categories) * 100
            max_value = max(max_value, total)
            
    y_max = (max_value // 10 + 1) * 10

    for i, range_label in enumerate(data):
        for j, difficulty in enumerate(difficulty_levels):
            bottom = 0
            for k, category in enumerate(categories):
                value = data[range_label][category][difficulty] * 100
                plt.bar(
                    i + j * bar_width,
                    value,
                    width=bar_width,
                    color=sub_colors[k % len(sub_colors)],
                    edgecolor='black',
                    bottom=bottom
                )
                bottom += value
                if value > 0:
                    plt.text(i + j * bar_width, bottom - value / 2, f'{int(value)}', ha='center', va='center', color='white', fontsize=8)
            
            total = sum(data[range_label][category][difficulty] * 100 for category in categories)
            plt.text(i + j * bar_width, total + 1, f'{int(total)}', ha='center', va='bottom', color='black')

    plt.xticks([r + bar_width * 1.5 for r in range(len(data))], data.keys())
    plt.xlim(-0.5, len(data) - 0.5 + bar_width * len(difficulty_levels))
    plt.ylim(0, y_max)

    plt.title(f'Analyze {Event_count}')
    plt.xlabel('Argument distance')
    plt.ylabel('ratio of questions')
    
    plt.legend(categories, loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
    plt.tight_layout()
    plt.savefig(f'pictures/{plt_name}.png')
    plt.clf()

def process_data(data, categories):
    record = defaultdict(Counter)
    for key in data:
        total = sum(data[key].values())
        for difficulty in categories:
            record[key][difficulty] = round(data[key][difficulty] / total, 2)
    return record

def statistics_data(dataset, Event_count, categories, our_or_llm):
    question_type = defaultdict(Counter)
    question_range = defaultdict(Counter)
    question_type_range = defaultdict(lambda: defaultdict(Counter))

    count = 0
    for data in dataset:
        if our_or_llm == 'our':
            ques = data[2].split('[')[1]
            research_idx = -5
            if 'False' in data[6]:
                continue
        elif our_or_llm == 'llm':
            ques = data[3].split('.')[-1]
            research_idx = -1
        elif our_or_llm == 'fairytale':
            ques = data[2]
            research_idx = -2
        count += 1
        
        if our_or_llm == 'llm' or our_or_llm == 'fairytale' or our_or_llm == 'our':#or data[5] == 'why':
            if 'who' in ques or 'Who' in ques:
                q_type = 'who'
            elif 'when' in ques or 'When' in ques:
                q_type = 'when'
            elif 'what' in ques or 'What' in ques:
                q_type = 'what'
            elif 'where' in ques or 'Where' in ques:
                q_type = 'where'
            elif 'why' in ques or 'Why' in ques:
                q_type = 'why'
            elif 'how' in ques or 'How' in ques:
                q_type = 'how'

            if data[research_idx] == 3:
                q_defficult = 'very Easy'
            elif data[research_idx] == 2:
                q_defficult = 'Easy'
            elif data[research_idx] == 1 :
                q_defficult = 'Medium'
            elif data[research_idx] == 0:
                q_defficult = 'Hard'
            
            question_type[q_type][q_defficult] += 1

            # Fairytale
            if our_or_llm == 'fairytale':
                q_range = data[-1]
                question_range[q_range][q_defficult] += 1

            # LLM no event distance
            if our_or_llm == 'our':
                if data[-4] < 2:
                    q_range = 'Near'
                elif data[-4] < 4:
                    q_range = 'Moderate'
                else:
                    q_range = 'Far'
                question_range[q_range][q_defficult] += 1
                question_type_range[q_range][q_type][q_defficult] += 1
        
    # print(question_type)
    # for key in question_type:
    #     print(key, sum(question_type[key].values()))

    # print('\n', question_range)
    # for key in question_range:
    #     print(key, sum(question_range[key].values()))

    processed_question_type =  process_data(question_type, categories)
    display_question_defficulty_histogram(processed_question_type, categories, f'Question type {Event_count}', f"{Event_count} Event",  x_label="Question type")

    if our_or_llm == 'our':
        processed_question_range =  process_data(question_range, categories)
        display_question_defficulty_histogram(processed_question_range, categories, f'Question range {Event_count}', f"{Event_count} Event",  x_label="Argument distance")

    return question_type, question_range, question_type_range

def count_range(record, question_type, range):
    for q_type in question_type:
        for level in question_type[q_type]:
            record[range][q_type][level] += question_type[q_type][level]
    
    return record

def cal_ratio(record):
    for range_label in record:
        total = 0
        for q_type in record[range_label]:
            total += sum(record[range_label][q_type].values())

        for q_type in record[range_label]:
            for level in record[range_label][q_type]:
                record[range_label][q_type][level] = round(record[range_label][q_type][level] / total, 2)
    return record

def main(method, exampler = 5, answer = False):
    if answer:
        dataset_2 = get_data(f'csv/4_w_ans_correct_ratio_2_{method}.csv')
        dataset_3 = get_data(f'csv/4_w_ans_correct_ratio_3_{method}.csv')
        dataset_4 = get_data(f'csv/4_w_ans_correct_ratio_4_{method}.csv')
    elif not answer:
        dataset_2 = get_data(f'csv/4_wo_ans_correct_ratio_2_{method}.csv')
        dataset_3 = get_data(f'csv/4_wo_ans_correct_ratio_3_{method}.csv')
        dataset_4 = get_data(f'csv/4_wo_ans_correct_ratio_4_{method}.csv')

    llm_dataset_respective = get_data(f'csv/4_llm_correct_ratio_5_respective.csv')
    llm_dataset_together = get_data(f'csv/4_llm_correct_ratio_5_together.csv')
    llm_3_dataset_together = get_data(f'csv/4_llm_correct_ratio_3.csv')

    fairytale_dataset = get_data(f'csv/4_fairytale_correct_ratio.csv')
    # fairytale_w_llm_dataset = get_data(f'csv/4_fairytale_w_llm_correct_ratio.csv')

    categories = ['very Easy', 'Easy', 'Medium', 'Hard']
    stat_type_2, stat_range_2, stat_type_range_2 = statistics_data(dataset_2, Event_count = 2, categories = categories, our_or_llm = 'our')
    stat_type_3, stat_range_3, stat_type_range_3 = statistics_data(dataset_3, Event_count = 3, categories = categories, our_or_llm = 'our')
    stat_type_4, stat_range_4, stat_type_range_4 = statistics_data(dataset_4, Event_count = 4, categories = categories, our_or_llm = 'our')

    llm_stat_respective, _, _ = statistics_data(llm_dataset_respective, Event_count = "respective", categories = categories, our_or_llm = 'llm')
    llm_stat_together, _, _ = statistics_data(llm_dataset_together, Event_count = "together", categories = categories, our_or_llm = 'llm')
    llm_3_stat, _, _ = statistics_data(llm_3_dataset_together, Event_count = "together", categories = categories, our_or_llm = 'llm')

    fairytale_stat, fairytale_range, _ = statistics_data(fairytale_dataset, Event_count = "fairytale", categories = categories, our_or_llm = 'fairytale')
    # fairytale_w_llm_stat, _, _ = statistics_data(fairytale_w_llm_dataset, Event_count = "fairytale_w_llm", categories = categories, our_or_llm = 'fairytale')

    """
    統計 Event 數量多寡影響問題難易度
    """
    total_record = defaultdict(Counter)

    for type in fairytale_stat:
        for level in fairytale_stat[type]:
            total_record['Fairytale'][level] += fairytale_stat[type][level]

    for type in stat_type_2:
        for level in stat_type_2[type]:
            total_record[2][level] += stat_type_2[type][level]
    
    for type in stat_type_3:
        for level in stat_type_3[type]:
            total_record[3][level] += stat_type_3[type][level]
    
    for type in stat_type_4:
        for level in stat_type_4[type]:
            total_record[4][level] += stat_type_4[type][level]
    
    print(total_record)
    input()

    # processed_total = process_data(total_record, categories)
    # display_question_defficulty_histogram(processed_total, categories, 'Question type total', "", x_label='number of events')

    # Comparsion Fairytale

    # for type in fairytale_w_llm_stat:
    #     for level in fairytale_w_llm_stat[type]:
    #         total_record['Fairytale_w_llm'][level] += fairytale_w_llm_stat[type][level]

    processed_total = process_data(total_record, categories)
    display_question_defficulty_histogram(processed_total, categories, 'Comparsion Fairytale', "", x_label='number of events')

    # Comparsion LLM
    for type in llm_3_stat:
        for level in llm_3_stat[type]:
            total_record['Chatgpt 3'][level] += llm_3_stat[type][level]
    
    # for type in llm_stat_together:
    #     for level in llm_stat_together[type]:
    #         total_record['Chatgpt 5'][level] += llm_stat_together[type][level]
    
    processed_total = process_data(total_record, categories)
    display_question_defficulty_histogram(processed_total, categories, 'Comparsion LLM & Fairytale', "", x_label='number of events')

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
    
    processed_record = process_data(record, categories)
    display_question_defficulty_histogram(processed_record, categories, 'Question range total', "total range", x_label="Argument distance")

    """
    計算各個類別 Range ，不同的問題類型影響問題難易度
    """
    record = defaultdict(lambda: defaultdict(Counter))
    for range in stat_type_range_2:
        record = count_range(record, stat_type_range_2[range], range)
    
    for range in stat_type_range_3:
        record = count_range(record, stat_type_range_3[range], range)

    for range in stat_type_range_4:
        record = count_range(record, stat_type_range_4[range], range)
    
    # Near, Medorate, Far histogram
    for range in record:
        processed_record = process_data(record[range], categories)
        display_question_defficulty_histogram(processed_record, categories, range, f"{range} range", x_label="Question type")
    
    record = cal_ratio(record)
    categories = ['who', 'what', 'when', 'why']
    display_inner_ratio_histogram(record, categories, f'Question_range_total', "total range")
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Answer', '-a', type=bool, default=False)
    parser.add_argument('--method', '-m', type=str, choices=['gemini', 'gpt', 'cosine60', 'cosine50', 'cosine55', 'cosine53', 'cosine52'], default='cosine53')
    parser.add_argument('--exampler', '-e', type=int, default=5)
    args = parser.parse_args()
    
    main(method=args.method, exampler = args.exampler, answer=args.Answer)