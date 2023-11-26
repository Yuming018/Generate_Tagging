from dataloader import Datasets
from model import create_model
from training import train_model, training
from inference import inference
from helper import checkdir
import argparse
import torch
import csv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

def main(batch_size = 4,
         epochs=10,
         path_save_model = 'save_model',
         device = 'cpu',
         test_mode = False,
         relation_tag = False,
         tagging = False,):
   
    model_name = "bigscience/mt0-large"
    
    path_save_model = checkdir(path_save_model, relation_tag, tagging)
    if tagging:
        file_name = path_save_model + 'tagging.csv'
    else:
        file_name = path_save_model + 'question.csv'
    print(path_save_model)
    print(file_name)
    
    train_data = Datasets('data/train.csv', model_name, relation_tag = relation_tag, tagging = tagging)
    valid_data = Datasets('data/valid.csv', model_name, relation_tag = relation_tag, tagging = tagging)
    test_data = Datasets('data/test.csv', model_name, relation_tag = relation_tag, tagging = tagging)
    train_dataloader = DataLoader(train_data, batch_size = batch_size, drop_last = True)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, drop_last = True)
    test_dataloader = DataLoader(test_data, batch_size = 1, drop_last = True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_model(model_name).to(device)
    if not test_mode:
        training(model, tokenizer, train_data, valid_data, path_save_model, epochs=epochs, batch_size = batch_size, tagging = tagging)
        # train_model(model, train_dataloader, valid_dataloader, device, tokenizer=tokenizer, epochs=epochs, path_save_model = best_pth)
    inference(model, tokenizer, test_data, device, path = file_name, path_save_model = path_save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=5)
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--test_mode', '-tm', type=bool, default=False)
    parser.add_argument('--relation_tag', '-r', type=bool, default=False)
    parser.add_argument('--tagging', '-t', type=bool, default=False)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(batch_size = args.batch_size, epochs = args.epoch, device=device, test_mode = args.test_mode, relation_tag = args.relation_tag, tagging = args.tagging)