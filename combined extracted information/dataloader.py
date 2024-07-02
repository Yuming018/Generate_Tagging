from helper import enconder
import pandas as pd
import re

def get_data(path):
    data = pd.read_csv(path)
    data = data.fillna('')
    return data.values

class Dataset:
    def __init__(self, path) -> None:
        self.dataset = get_data(path)
        self.paragraph, self.context, self.target = [], [], []
        self.create_dataset()
    
    def create_dataset(self):
        story_name = ""
        for idx in range(len(self.dataset)):
            
            current_name = "-".join(self.dataset[idx][4].split('-')[:-1])
            if current_name == story_name:
                continue
            self.context.append(self.dataset[idx][1])
            self.target.append(self.dataset[idx][2])
            self.paragraph.append(self.dataset[idx][4])
            story_name = current_name

        return
   
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.paragraph[idx], self.context[idx], self.target[idx]

class Concat_dataset:
    def __init__(self, path) -> None:
        self.dataset = get_data(path)
        self.paragraph, self.context, self.target = [], [], []
        self.create_dataset()
    
    def create_dataset(self):
        story_name = ""
        story_count, para_count = [], []

        for idx in range(len(self.dataset)):
            current_name = "-".join(self.dataset[idx][4].split('-')[:-1])
            if current_name == story_name:
                continue
                           
            if idx != 0 and (len(story_count) == 3 or current_name[:-2] != story_name[:-2] or int(current_name[-2:]) != int(story_name[-2:])+1):
                # print(current_name[:-2], story_name[:-2])
                # print(int(current_name[-2:]), int(story_name[-2:]))
                # print(current_name, para_count)
                # input()
                context = ""
                for story_idx in story_count:
                    context += f'{self.dataset[story_idx][1]} '
                self.context.append(context)
                self.target.append(self.dataset[story_count[0]][2])
                self.paragraph.append(story_name[:-2] + f'{para_count[0]} ~ {para_count[-1]}')
                story_count = [idx] 
                para_count = [int(current_name[-2:])]     
            else:
                story_count.append(idx)
                para_count.append(int(current_name[-2:]))         
            
            story_name = current_name
        return

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.paragraph[idx], self.context[idx], self.target[idx]


if __name__ == '__main__':
    pass