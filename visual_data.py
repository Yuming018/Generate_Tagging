import pandas as pd
import argparse
import statistics
from collections import defaultdict, Counter

legal_tagging = ['Causal Effect',
            'Temporal',
            # 'Coreference'
            'State',
            'Action'
]

Relation_label = {'X attribute': 'Persona', 
         'X intent' : 'MentalState', 
         'X reaction' : 'MentalState', 
         'Other reaction' : 'MentalState',
         'isBefore' : 'isBefore', 
         'the same' : 'the same', 
         'isAfter' : 'isAfter', 
         'X need' : 'Event', 
         'Effect on X' : 'Event',  
         'X want'  : 'Event', 
         'Other want'  : 'Event', 
         'Effect on other'  : 'Event'}

class Tagging_Datasets:
    def __init__(self, path, event_or_relation) -> None:
        self.tagging_type = event_or_relation
        if event_or_relation == 'Event':
            self.index = 5 #index
        elif event_or_relation == 'Relation':
            self.index = 6 #index
        self.vis_data, self.count_data = defaultdict(Counter), Counter()
        self.dataset = self.get_data(path)
        self.statistics_data()

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values

    def statistics_data(self):
        story_name = "-".join(self.dataset[0][4].split('-')[:-1])
        story_list = []
        for idx in range(len(self.dataset)):
            tag_type = self.dataset[idx][self.index].split(' - ')[0]
            if self.tagging_type == 'Event':
                if tag_type in legal_tagging:
                    self.count_data[self.dataset[idx][4]] += 1
                    self.vis_data['all']['all'] += 1
                    self.vis_data[tag_type]['all'] += 1
                    self.vis_data[tag_type][self.dataset[idx][self.index].split(' - ')[1]] += 1
            
            elif self.tagging_type == 'Relation':
                current_story_name = "-".join(self.dataset[idx][4].split('-')[:-1])
                if current_story_name != story_name and story_list and tag_type in legal_tagging:
                    story_name = current_story_name
                    story_list = []
                if tag_type in legal_tagging:
                    self.count_data[self.dataset[idx][4]] += 1
                    if self.dataset[idx][self.index].split(' - ')[1] in Relation_label:
                        self.vis_data['all']['all'] += 1
                        self.vis_data[tag_type]['all'] += 1
                        self.vis_data[tag_type][Relation_label[self.dataset[idx][self.index].split(' - ')[1]]] += 1
                    story_list.append(idx)

class Question_Datasets:
    def __init__(self, path) -> None:
        self.dataset = self.get_data(path)
        if path == 'data/test.csv':
            self.Relation_tagging = self.get_data('save_model/Relation/tagging/tagging.csv')
            self.Evnet_tagging = self.get_data('save_model/Event/tagging/tagging.csv')
        self.vis_data, self.count_data = defaultdict(Counter), Counter()
        self.statistics_data()

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values

    def statistics_data(self):
        story_name = "-".join(self.dataset[0][4].split('-')[:-1])
        story_list = []
        for idx in range(len(self.dataset)):
            current_story_name = "-".join(self.dataset[idx][4].split('-')[:-1])
            if current_story_name != story_name and story_list:
                story_name = current_story_name
                story_list = []
            if self.dataset[idx][5].split(' - ')[0] in legal_tagging or self.dataset[idx][6].split(' - ')[0] in legal_tagging:
                self.count_data[self.dataset[idx][4]] += 1
                self.vis_data['all']['all'] += 1
                story_list.append(idx)
        return          

def visualize(title, vis_data, count_data):
    print(title)
    for key in vis_data:
        print(key)
        for type in vis_data[key]:
            print(type, vis_data[key][type])
        print()

    total = 0
    for story in count_data:
        total += count_data[story]
    print('average :', total /len(count_data))
    print('S.D :', statistics.pstdev(count_data.values()))
    print('min :', count_data.most_common()[-1])
    print('max :', count_data.most_common()[0], '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_or_relation', '-t', type=str, choices=['Event', 'Relation'], default='Event')
    parser.add_argument('--Generation', '-g', type=str, choices=['tagging', 'question'], default='tagging')
    args = parser.parse_args()

    if args.Generation == 'tagging' :
        train_data = Tagging_Datasets('data/train.csv', event_or_relation = args.event_or_relation)
        valid_data = Tagging_Datasets('data/valid.csv', event_or_relation = args.event_or_relation)
        test_data = Tagging_Datasets('data/test.csv', event_or_relation = args.event_or_relation)
    elif args.Generation == 'question':
        train_data = Question_Datasets('data/train.csv')
        valid_data = Question_Datasets('data/valid.csv')
        test_data = Question_Datasets('data/test.csv')


    visualize('Train', train_data.vis_data, train_data.count_data)
    visualize('Valid', valid_data.vis_data, valid_data.count_data)
    visualize('Test', test_data.vis_data, test_data.count_data)

    if args.Generation == 'tagging' :
        print('Tagging : ', args.event_or_relation)
    print('Generation : ', args.Generation)
