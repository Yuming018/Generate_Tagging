from helper import enconder
import pandas as pd
import re

class Dataset:
    def __init__(self, path, tokenizer) -> None:
        self.max_len = 512
        self.tokenizer = tokenizer
        self.dataset = self.get_data(path)
        self.context_type, self.paragraph, self.context, self.focus_context, self.target = [], [], [] ,[], []
        self.create_dataset()

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values
    
    def create_dataset(self):
        story_name = self.dataset[0][4]
        story_idx = []
        for idx in range(len(self.dataset)):
            if self.dataset[idx][4] != story_name:
                context_type, context, focus_context = self.create_input(idx-1, story_idx)
                self.context_type.append(context_type)
                self.paragraph.append(self.dataset[idx-1][4])
                self.context.append(context)
                self.focus_context.append(focus_context)
                self.target.append(self.create_target(idx-1))
                story_name = self.dataset[idx][4]
                story_idx = []
            story_idx.append(idx)
        return

    def create_input(self, idx, story_idx):
        focus_context = self.text_segmentation(story_idx)
        context_type, context = self.dataset[idx][0], self.dataset[idx][1]
        return context_type, context, focus_context

    def create_target(self, idx):
        text = self.dataset[idx][2]
        return text
    
    def text_segmentation(self, story_idx):
        min_b, max_b = float('inf'), float('-inf')
        for idx in story_idx:
            for i in range(7, len(self.dataset[idx])):
                if self.dataset[idx][i] != '':
                    numbers = re.findall(r'\d+', self.dataset[idx][i])
                    min_b = min(min_b, int(numbers[-2]))
                    max_b = max(max_b, int(numbers[-1]))
        words = re.findall(r'\S+|[\s]+', self.dataset[idx][1])
        if words[min_b] == ' ':
            min_b -= 1
        return "".join(words[min_b:max_b])
    
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.context_type[idx], self.paragraph[idx], self.context[idx], self.focus_context[idx], self.target[idx]

if __name__ == '__main__':
    pass