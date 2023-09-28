from transformers import AutoTokenizer, BartForConditionalGeneration
from helper import checkdir, enconder
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import torch
import os
import csv
import re

question_type = ['action', 'outcome resolution', 'causal relationship',
                 'prediction', 'setting', 'feeling', 'character']

def main(path_save_model = 'save_model/',
        device = 'cpu',
        relation_tag = False,
):
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    text_seg = read_data(path = 'data/context/test')
    
    context_, question_, tagging_, type = [], [], [], []
    for text in tqdm(text_seg):
        for tagging_type in question_type:
            story = f'[SEP] {tagging_type} [SEP] {text} [SEP]'
            input_ids = enconder(tokenizer, text=story)
            input_ids = torch.tensor([input_ids.get('input_ids')]).to(device)
            tagging_ids = generate_tagging(path_save_model, relation_tag, model, input_ids)
            combined_tensor = torch.cat((input_ids, tagging_ids), dim=1)
            question_ids = generate_question(path_save_model, relation_tag, model, combined_tensor)
            tagging = tokenizer.batch_decode(tagging_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            question = tokenizer.batch_decode(question_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print('context: ', story)
            # print('tagging: ', tagging)
            # print(tokenizer.batch_decode(combined_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
            # print(question)
            # input()
            type.append(tagging_type)
            tagging_.append(tagging)
            question_.append(question)
            context_.append(story)
    
    save_csv(type, context_, tagging_, question_, path = '.predict.csv')

def read_data(path):
    dataset = []
    for file_name in os.listdir(path):
        with open(os.path.join(path, file_name)) as f:
            line = f.read()
            data = text_segmentation(line)
        dataset += data
    return dataset

def text_segmentation(line):
    data = []
    split_len = 3
    sentences = re.split(r'[.,] ', line)
    for i in range(0, len(sentences), split_len):
        data.append(', '.join(sentences[i:i+split_len]))
    return data

def generate_tagging(path_save_model, relation_tag, model, input_ids):
    path_save_model = checkdir(path_save_model, relation_tag)
    best_pth = path_save_model + 'tagging.pth'
    model.load_state_dict(torch.load(best_pth))
    output = model.generate(input_ids, num_beams=2, min_length=0)
    return output

def generate_question(path_save_model, relation_tag, model, tagging_ids):
    path_save_model = checkdir(path_save_model, relation_tag)
    best_pth = path_save_model + 'question.pth'
    model.load_state_dict(torch.load(best_pth))
    output = model.generate(tagging_ids, num_beams=2, min_length=0)
    return output

def save_csv(type, context, tagging, question, path):
    row = ['ID', 'type', 'context', 'tagging', 'prediction']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(input)):
            writer.writerow([i, type[i], context[i], tagging[i], question[i]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--relation_tag', '-r', type=bool, default=False)
    parser.add_argument('--tagging', '-t', type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(device=device)