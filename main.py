from dataloader import Tagging_Datasets, Question_Datasets
from model import Mt0, T5
from training import train_model, training
from inference import inference
from helper import checkdir
import argparse
import torch
from torch.utils.data import DataLoader

def main(batch_size = 4,
         epochs=10,
         path_save_model = 'save_model',
         device = 'cpu',
         test_mode = False,
         event_or_relation = 'Event',
         Generation = 'tagging',
         model_name = "Mt0"):
   
    if Generation == 'tagging' :
        print('Tagging : ', event_or_relation)
    print('Generation : ', Generation)
    print('Model name :', model_name)

    create_model = {
        "Mt0" : Mt0(),
        "T5" : T5(),
    }

    model, tokenizer = create_model[model_name]
    model = model.to(device)

    path_save_model = checkdir(path_save_model, event_or_relation, Generation, model_name)
    if Generation == 'tagging' :
        file_name = path_save_model + 'tagging.csv'
        train_data = Tagging_Datasets('data/train.csv', tokenizer, event_or_relation = event_or_relation)
        valid_data = Tagging_Datasets('data/valid.csv', tokenizer, event_or_relation = event_or_relation)
        test_data = Tagging_Datasets('data/test.csv', tokenizer, event_or_relation = event_or_relation)
    elif Generation == 'question':
        file_name = path_save_model + 'question.csv'
        train_data = Question_Datasets('data/train.csv', tokenizer)
        valid_data = Question_Datasets('data/valid.csv', tokenizer)
        test_data = Question_Datasets('data/test.csv', tokenizer)
    
    if not test_mode:
        training(model, tokenizer, train_data, valid_data, path_save_model, epochs=epochs, batch_size = batch_size)
        # train_model(model, train_dataloader, valid_dataloader, device, tokenizer=tokenizer, epochs=epochs, path_save_model = best_pth)
    inference(model, tokenizer, test_data, test_data.paragraph, device, save_file_path = file_name, path_save_model = path_save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=5)
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--test_mode', '-tm', type=bool, default=False)
    parser.add_argument('--event_or_relation', '-t', type=str, choices=['Event', 'Relation'], default='Event')
    parser.add_argument('--Generation', '-g', type=str, choices=['tagging', 'question'], default='tagging')
    parser.add_argument('--Model', '-m', type=str, choices=['Mt0', 'T5'], default='Mt0')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(batch_size = args.batch_size,
        epochs = args.epoch,
        device = device, 
        test_mode = args.test_mode, 
        event_or_relation = args.event_or_relation, 
        Generation = args.Generation,
        model_name = args.Model)