from dataloader import Dataset
from model import BART
from training import train_model
from inference import inference
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration

def main(batch_size = 4,
         epochs=10,
         path_save_model = 'save_model/',
         device = 'cpu',
         test_mode = False,
         relation_tag = False,):
   
    model_name = "facebook/bart-large-cnn"

    train_data = Dataset('data/train.csv', model_name, relation_tag = relation_tag)
    valid_data = Dataset('data/valid.csv', model_name, relation_tag = relation_tag)
    test_data = Dataset('data/test.csv', model_name, relation_tag = relation_tag)
    train_dataloader = DataLoader(train_data, batch_size = batch_size, drop_last = True)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, drop_last = True)
    test_dataloader = DataLoader(test_data, batch_size = 1, drop_last = True)

    # model = BART(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    if not test_mode:
        train_model(model, train_dataloader, valid_dataloader, device, tokenizer=tokenizer, epochs=epochs, path_save_model = path_save_model)
    best_pth = path_save_model + 'best_train.pth'
    inference(model, tokenizer, test_dataloader, device, path = path_save_model, best_pth=best_pth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=2)
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--test_mode', '-tm', type=bool, default=False)
    parser.add_argument('--relation_tag', '-r', type=bool, default=False)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(batch_size = args.batch_size, epochs = args.epoch, device=device, test_mode = args.test_mode, relation_tag = args.relation_tag)