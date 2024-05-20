import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')
    return data.values

def display_question_defficulty_histogram(data):
    who_keys, who_values = list(data['who'].keys()), list(data['who'].values())
    where_keys, where_values = list(data['where'].keys()), list(data['where'].values())
    why_keys, why_values = list(data['why'].keys()), list(data['why'].values())
    when_keys, when_values = list(data['when'].keys()), list(data['when'].values())
    what_keys, what_values = list(data['what'].keys()), list(data['what'].values())
    how_keys, how_values = list(data['how'].keys()), list(data['how'].values())
    
    bar_width = 0.12
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    categories = ['Easy', 'Medium', 'Hard']
    index_who = [categories.index(key) for key in who_keys]
    index_where = [categories.index(key) for key in where_keys]
    index_why = [categories.index(key) for key in why_keys]
    index_when = [categories.index(key) for key in when_keys]
    index_what = [categories.index(key) for key in what_keys]
    index_how = [categories.index(key) for key in how_keys]

    offsets = [-bar_width * 2.5, -bar_width * 1.5, -bar_width * 0.5, bar_width * 0.5, bar_width * 1.5, bar_width * 2.5]
    plt.bar([i + offsets[0] for i in index_who], who_values, width=bar_width, color=colors[0], label='Who', edgecolor='black')
    plt.bar([i + offsets[1] for i in index_where], where_values, width=bar_width, color=colors[1], label='Where', edgecolor='black')
    plt.bar([i + offsets[2] for i in index_why], why_values, width=bar_width, color=colors[2], label='Why', edgecolor='black')
    plt.bar([i + offsets[3] for i in index_when], when_values, width=bar_width, color=colors[3], label='When', edgecolor='black')
    plt.bar([i + offsets[4] for i in index_what], what_values, width=bar_width, color=colors[4], label='What', edgecolor='black')
    plt.bar([i + offsets[5] for i in index_how], how_values, width=bar_width, color=colors[5], label='How', edgecolor='black')

    # 添加標籤
    for i, value in enumerate(who_values):
        plt.text(index_who[i] + offsets[0], value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=colors[0])

    for i, value in enumerate(where_values):
        plt.text(index_where[i] + offsets[1], value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=colors[1])

    for i, value in enumerate(why_values):
        plt.text(index_why[i] + offsets[2], value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=colors[2])

    for i, value in enumerate(when_values):
        plt.text(index_when[i] + offsets[3], value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=colors[3])

    for i, value in enumerate(what_values):
        plt.text(index_what[i] + offsets[4], value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=colors[4])

    for i, value in enumerate(how_values):
        plt.text(index_how[i] + offsets[5], value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=colors[5])

    plt.xticks(range(len(categories)), categories)

    # 設置 xy 軸範圍
    plt.xlim(-0.5, len(categories) - 0.5)
    plt.ylim(0, 1.1)  
    
    plt.title('Analyze')
    plt.xlabel('Question Difficulty')
    plt.ylabel('Question count')
    plt.legend()
    plt.savefig('csv/Question type.png')
    # plt.show()
    plt.clf()

def display_question_range_histogram(data):
    Near_keys, Near_values = list(data['Near'].keys()), list(data['Near'].values())
    Moderate_keys, Moderate_values = list(data['Moderate'].keys()), list(data['Moderate'].values())
    Far_keys, Far_values = list(data['Far'].keys()), list(data['Far'].values())
    
    bar_width = 0.25
    color_train = 'blue'
    color_valid = 'orange'
    color_test = 'green'

    categories = ['Easy', 'Medium', 'Hard']
    index_train = [categories.index(key) for key in Near_keys]
    index_valid = [categories.index(key) for key in Moderate_keys]
    index_test = [categories.index(key) for key in Far_keys]

    plt.bar([i - bar_width for i in index_train], Near_values, width=bar_width, color=color_train, label='Near(0<=x<3)', edgecolor='black')
    plt.bar(index_valid, Moderate_values, width=bar_width, color=color_valid, label='Moderate(3<=x<6)', edgecolor='black')
    plt.bar([i + bar_width for i in index_test], Far_values, width=bar_width, color=color_test, label='Far(6<=x)', edgecolor='black')

    for i, value in enumerate(Near_values):
        plt.text(index_train[i] - bar_width, value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=color_train)

    for i, value in enumerate(Moderate_values):
        plt.text(index_valid[i], value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=color_valid)

    for i, value in enumerate(Far_values):
        plt.text(index_test[i] + bar_width, value + 0.01, f'{value:.2f}', ha='center', va='bottom', color=color_test)

    plt.xticks(range(len(categories)), categories)

    # 設置 xy 軸範圍
    plt.xlim(-1, len(categories))
    plt.ylim(0, 1.1)  
    
    plt.title('Analyze')
    plt.xlabel('Question Difficulty')
    plt.ylabel('Question count')
    plt.legend()
    plt.savefig('csv/Question range.png')
    # plt.show()
    plt.clf()

def process_data(data):
    record = defaultdict(Counter)
    keys = ['Easy', 'Medium', 'Hard']
    for key in data:
        total = sum(data[key].values())
        for difficulty in keys:
            record[key][difficulty] = round(data[key][difficulty] / total, 2)
    return record

def main():
    dataset = get_data('csv/4_correct_ratio.csv')
    question_set = set()
    question_type = defaultdict(Counter)
    question_range = defaultdict(Counter)

    for data in dataset:
        ques = data[2].split('[')[1]
        if 'who' in ques:
            q_type = 'who'
        elif 'when' in ques:
            q_type = 'when'
        elif 'what' in ques:
            q_type = 'what'
        elif 'where' in ques:
            q_type = 'where'
        elif 'why' in ques:
            q_type = 'why'
        elif 'how' in ques:
            q_type = 'how'

        if data[11] < 3:
            q_range = 'Near'
        elif data[11] < 6:
            q_range = 'Moderate'
        else:
            q_range = 'Far'
        
        if data[10] == 3:
            q_defficult = 'Easy'
        elif data[10] == 2 :
            q_defficult = 'Medium'
        elif data[10] == 1:
            q_defficult = 'Hard'
        
        if ques not in question_set:
            question_type[q_type][q_defficult] += 1
            question_range[q_range][q_defficult] += 1
        question_set.add(ques)
    
    processed_question_type =  process_data(question_type)
    display_question_defficulty_histogram(processed_question_type)
    
    processed_question_range =  process_data(question_range)
    display_question_range_histogram(processed_question_range)
    return 

if __name__ == '__main__':
    main()