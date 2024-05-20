import csv
import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel
from helper import check_checkpoint

def seq2seq_inference(model_name, model, tokenizer, test_dataloader, test_data_paprgraph, device, save_file_path, path_save_model):

    generation_config = GenerationConfig(
        max_length=100,
        early_stopping=False,
        decoder_start_token_id=0,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id,
        penalty_alpha=0.6, 
        top_k=4, 
        temperature=0.5,
        max_new_tokens=512,
    )

    prediction, target, context, paragraph = [], [], [], []
    if model_name == 'gemma':
        model.half()
    model.to(device)
    model.eval()
    for data, para in tqdm(zip(test_dataloader, test_data_paprgraph)):
        input_ids, label = data['input_ids'], data['labels']
        input_ids = torch.tensor(input_ids).to(device)
        output = model.generate(input_ids=input_ids.unsqueeze(0), max_new_tokens=100)
        prediction.append(tokenizer.decode(output[0], skip_special_tokens=True))
        # print(tokenizer.decode(output[0], skip_special_tokens=True))
        # print(tokenizer.decode(input_ids, skip_special_tokens=True))
        # input()
        target.append(tokenizer.decode(label, skip_special_tokens=True))
        context.append(tokenizer.decode(input_ids, skip_special_tokens=True))
        paragraph.append(para)

    save_csv(prediction, target, context, paragraph, save_file_path)

def cls_inference(model_name, model, tokenizer, test_dataloader, test_data_paprgraph, device, save_file_path, path_save_model):

    id2label = {0: 'Can not answer', 1: 'Can answer'}
    prediction, target, context, paragraph = [], [], [], []
    model.to(device)
    for data, para in tqdm(zip(test_dataloader, test_data_paprgraph)):
        input_ids, label = data['input_ids'], data['label']
        input_ids = torch.tensor(input_ids).to(device)
        output = model(input_ids=input_ids.unsqueeze(0)).logits.argmax().item()
        prediction.append(id2label[output])
        target.append(id2label[label])
        context.append(tokenizer.decode(input_ids, skip_special_tokens=True))
        paragraph.append(para)

    save_csv(prediction, target, context, paragraph, save_file_path)

def save_csv(prediction, target, context, paragraph, path):
    row = ['Paragraph', 'Content', 'Prediction', 'Reference']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(prediction)):
            writer.writerow([paragraph[i], context[i], prediction[i], target[i]])

if __name__ == '__main__':
    pass