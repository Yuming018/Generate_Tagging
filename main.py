from dataloader import Extraction_Datasets, Question_generation_Datasets, Ranking_dataset
from model import create_model
from training import train_model, training
from inference import inference
from helper import checkdir
import argparse
import torch
from torch.utils.data import DataLoader

def main(batch_size = 4,
         epochs=10,
         max_len = 512,
         path_save_model = 'save_model',
         device = 'cpu',
         test_mode = False,
         event_or_relation = 'Event',
         Generation = 'tagging',
         model_name = "Mt0"):
   
    if Generation == 'tagging' :
        print('Tagging : ', event_or_relation)
        print('Generation : ', Generation)
    elif Generation == 'question':
        print('Generation : ', Generation)
    elif Generation == 'ranking':
        print('Generation : ', Generation)

    model, tokenizer = create_model(model_name)
    model.to(device)

    path_save_model = checkdir(path_save_model, event_or_relation, Generation, model_name)
    if Generation == 'tagging' :
        file_name = path_save_model + 'tagging.csv'
        train_data = Extraction_Datasets('data/train.csv', model_name, tokenizer, event_or_relation = event_or_relation, max_len = max_len)
        valid_data = Extraction_Datasets('data/valid.csv', model_name, tokenizer, event_or_relation = event_or_relation, max_len = max_len)
        test_data = Extraction_Datasets('data/test.csv', model_name, tokenizer, event_or_relation = event_or_relation, max_len = max_len)
    elif Generation == 'question':
        file_name = path_save_model + 'question.csv'
        train_data = Question_generation_Datasets('data/train.csv', model_name, tokenizer, max_len)
        valid_data = Question_generation_Datasets('data/valid.csv', model_name, tokenizer, max_len)
        test_data = Question_generation_Datasets('data/test.csv', model_name, tokenizer, max_len)
    elif Generation == 'ranking':
        file_name = path_save_model + 'ranking.csv'
        train_data = Ranking_dataset('data/Ranking/train_ranking.csv', model_name, tokenizer, max_len)
        valid_data = Ranking_dataset('data/Ranking/valid_ranking.csv', model_name, tokenizer, max_len)
        test_data = Ranking_dataset('data/Ranking/test_ranking.csv', model_name, tokenizer, max_len)
    
    if not test_mode:
        training(model, tokenizer, train_data, valid_data, path_save_model, epochs=epochs, batch_size = batch_size)
        # train_model(model, train_dataloader, valid_dataloader, device, tokenizer=tokenizer, epochs=epochs, path_save_model = best_pth)
    inference(model_name, model, tokenizer, test_data, test_data.paragraph, device, save_file_path = file_name, path_save_model = path_save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=5)
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--max_len', '-l', type=int, default=512)
    parser.add_argument('--test_mode', '-tm', type=bool, default=False)
    parser.add_argument('--event_or_relation', '-t', type=str, choices=['Event', 'Relation'], default='Event')
    parser.add_argument('--Generation', '-g', type=str, choices=['tagging', 'question', 'ranking'], default='tagging')
    parser.add_argument('--Model', '-m', type=str, choices=['Mt0', 'T5', 'Bart', 'roberta', 'gemma', 'flant5'], default='Mt0')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(batch_size = args.batch_size,
        epochs = args.epoch,
        max_len = args.max_len,
        device = device, 
        test_mode = args.test_mode, 
        event_or_relation = args.event_or_relation, 
        Generation = args.Generation,
        model_name = args.Model)