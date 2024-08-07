from dataloader import Extraction_Datasets, Question_generation_Datasets, Answer_generation_dataset, Ranking_dataset
from model import create_model
from training import seq2seq_training, cls_training, ans_training
from inference import seq2seq_inference, cls_inference
from helper import checkdir
import argparse
import torch

def main(batch_size = 4,
         epochs=10,
         max_len = 512,
         path_save_model = 'save_model',
         device = 'cpu',
         test_mode = False,
         Generation = 'Event',
         gen_answer = False,
         model_name = "Mt0"):
   
    if Generation == 'question' and gen_answer:
        print('Generation : ', 'QA_pair')
    else:
        print('Generation : ', Generation)
    
    path_save_model = checkdir(path_save_model, Generation, model_name, gen_answer)
    model, tokenizer = create_model(model_name, Generation, test_mode, path_save_model)
    model.to(device)

    if Generation == 'Event' or Generation == 'Relation':
        file_name = path_save_model + 'tagging.csv'
        train_data = Extraction_Datasets('data/train.csv', model_name, tokenizer, event_or_relation = Generation, max_len = max_len)
        valid_data = Extraction_Datasets('data/valid.csv', model_name, tokenizer, event_or_relation = Generation, max_len = max_len)
        test_data = Extraction_Datasets('data/test.csv', model_name, tokenizer, event_or_relation = Generation, max_len = max_len)
    elif Generation == 'question':
        file_name = path_save_model + 'question.csv'
        train_data = Question_generation_Datasets('data/train.csv', model_name, tokenizer, max_len = 768, gen_answer = gen_answer)
        valid_data = Question_generation_Datasets('data/valid.csv', model_name, tokenizer, max_len = 768, gen_answer = gen_answer)
        test_data = Question_generation_Datasets('data/test.csv', model_name, tokenizer, max_len = 768, gen_answer = gen_answer)
    elif Generation == 'answer':
        file_name = path_save_model + 'answer.csv'
        train_data = Answer_generation_dataset('data/train.csv', model_name, tokenizer, max_len)
        valid_data = Answer_generation_dataset('data/valid.csv', model_name, tokenizer, max_len)
        test_data = Answer_generation_dataset('data/test.csv', model_name, tokenizer, max_len)
    elif Generation == 'ranking':
        file_name = path_save_model + 'ranking.csv'
        train_data = Ranking_dataset('data/Ranking/train_ranking.csv', model_name, tokenizer, max_len, gen_answer)
        valid_data = Ranking_dataset('data/Ranking/valid_ranking.csv', model_name, tokenizer, max_len, gen_answer)
        test_data = Ranking_dataset('data/Ranking/test_ranking.csv', model_name, tokenizer, max_len, gen_answer)

    print('Train : ', len(train_data))
    print('Valid : ', len(valid_data))
    print('Test : ', len(test_data))
    
    if not test_mode:
        if Generation == 'Event' or Generation == 'Relation' or Generation == 'question':
            seq2seq_training(model, tokenizer, train_data, valid_data, path_save_model, epochs=epochs, batch_size = batch_size)
        elif Generation == 'answer':
            ans_training(model, tokenizer, train_data, valid_data, path_save_model, epochs=epochs, batch_size = batch_size)
        elif Generation == 'ranking':
            cls_training(model, tokenizer, train_data, valid_data, path_save_model, epochs=epochs, batch_size = batch_size)

    if Generation == 'Event' or Generation == 'Relation' or Generation == 'question' or Generation == 'answer':
        seq2seq_inference(model_name, model, tokenizer, test_data, test_data.paragraph, device, save_file_path = file_name, path_save_model = path_save_model)
    elif Generation == 'ranking':
        cls_inference(model_name, model, tokenizer, test_data, test_data.paragraph, device, save_file_path = file_name, path_save_model = path_save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e',
                        type=int, 
                        default=5)
    parser.add_argument('--batch_size', '-b', 
                        type=int, 
                        default=2)
    parser.add_argument('--max_len', '-l', 
                        type=int, 
                        default=512)
    parser.add_argument('--test_mode', '-tm', 
                        type=bool, 
                        default=False)
    parser.add_argument('--Generation', '-g', 
                        type=str, 
                        choices=['Event', 'Relation', 'question', 'answer', 'ranking'], 
                        default='Event')
    parser.add_argument('--gen_answer', '-a', 
                        type=bool, 
                        default=False)
    parser.add_argument('--Model', '-m', 
                        type=str, 
                        choices=['Mt0', 'T5', 'T5_base', 'T5_small', 'bart', 'bert', 'gemma', 'flant5', 'roberta', 'distil', 'deberta'], 
                        default='Mt0')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(batch_size = args.batch_size,
        epochs = args.epoch,
        max_len = args.max_len,
        device = device, 
        test_mode = args.test_mode, 
        Generation = args.Generation,
        gen_answer = args.gen_answer,
        model_name = args.Model)