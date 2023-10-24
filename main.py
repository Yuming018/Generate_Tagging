from dataloader import Datasets
from model import create_model
from training import train_model
from inference import inference
from helper import checkdir
import argparse
import torch
import csv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

def main(batch_size = 4,
         epochs=10,
         path_save_model = 'save_model/',
         device = 'cpu',
         test_mode = False,
         relation_tag = False,
         tagging = False,):
   
    model_name = "bigscience/mt0-base"
    
    path_save_model = checkdir(path_save_model, relation_tag)
    if tagging:
        best_pth = path_save_model + 'tagging.pth'
        file_name = path_save_model + 'tagging.csv'
    else:
        best_pth = path_save_model + 'question.pth'
        file_name = path_save_model + 'question.csv'
    
    train_data = Datasets('data/train.csv', model_name, relation_tag = relation_tag, tagging = tagging)
    valid_data = Datasets('data/valid.csv', model_name, relation_tag = relation_tag, tagging = tagging)
    test_data = Datasets('data/test.csv', model_name, relation_tag = relation_tag, tagging = tagging)
    train_dataloader = DataLoader(train_data, batch_size = batch_size, drop_last = True)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, drop_last = True)
    test_dataloader = DataLoader(test_data, batch_size = 1, drop_last = True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_model(model_name).to(device)
    if not test_mode:
        collate_fn = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )

        args = Seq2SeqTrainingArguments(
            output_dir="./checkpoints",
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=5,
            learning_rate=1e-3,
            # optim='adafactor',
            fp16=False,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=100, # 保存checkpoint的step数
            save_total_limit=5, # 最多保存5个checkpoint
            predict_with_generate=True,
            weight_decay=0.01,
            include_inputs_for_metrics=True,
            lr_scheduler_type="polynomial",
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=valid_data,
            # compute_metrics=compute_metrics,
            args=args,
            data_collator=collate_fn,
            tokenizer=tokenizer,
        )
        trainer.train()
        model.save_pretrained(path_save_model)
        # train_model(model, train_dataloader, valid_dataloader, device, tokenizer=tokenizer, epochs=epochs, path_save_model = best_pth)
    inference(model, tokenizer, test_data, device, path = file_name, best_pth=best_pth, path_save_model = path_save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=3)
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--test_mode', '-tm', type=bool, default=False)
    parser.add_argument('--relation_tag', '-r', type=bool, default=False)
    parser.add_argument('--tagging', '-t', type=bool, default=False)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(batch_size = args.batch_size, epochs = args.epoch, device=device, test_mode = args.test_mode, relation_tag = args.relation_tag, tagging = args.tagging)