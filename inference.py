import csv
import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from helper import check_checkpoint

def inference(model_name, model, tokenizer, test_dataloader, test_data_paprgraph, device, save_file_path, path_save_model):
    model_path = check_checkpoint(path_save_model)
    if model_name == "Mt0":
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path)
    elif model_name == 'T5' or model_name == 'Bart' or model_name == 'flant5' :
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif model_name == 'roberta':
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif model_name == 'gemma':
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path)

    tagging_generation_config = GenerationConfig(
        max_length=100,
        # num_beams=4,
        early_stopping=False,
        decoder_start_token_id=0,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id,
        penalty_alpha=0.6, 
        # no_repeat_ngram_size=3,
        top_k=4, 
        temperature=0.5,
        max_new_tokens=256,
    )

    prediction, target, context, paragraph = [], [], [], []
    # model.load_state_dict(torch.load(best_pth))
    model.half()
    model.to(device)
    model.eval()
    for data, para in tqdm(zip(test_dataloader, test_data_paprgraph)):
        input_ids, label = data['input_ids'], data['labels']
        input_ids = torch.tensor(input_ids).to(device)
        output = model.generate(input_ids=input_ids.unsqueeze(0), generation_config=tagging_generation_config)
        prediction.append(tokenizer.decode(output[0], skip_special_tokens=True))
        target.append(tokenizer.decode(label, skip_special_tokens=True))
        context.append(tokenizer.decode(input_ids, skip_special_tokens=True))
        paragraph.append(para)

    # for step, batch in tqdm(enumerate(test_dataloader)):
    #     b_input_ids, _, b_labels = tuple(t.to(device) for t in batch)
    #     output = model.generate(input_ids = b_input_ids, generation_config=tagging_generation_config)
    #     output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     target_ = tokenizer.batch_decode(b_labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     context_ = tokenizer.batch_decode(b_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     prediction.append(output)
    #     # print(output)
    #     target.append(target_)
    #     context.append(context_)
    save_csv(prediction, target, context, paragraph, save_file_path)

def save_csv(prediction, target, context, paragraph, path):
    row = ['Paragraph', 'Content', 'Prediction', 'Reference']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(prediction)):
            writer.writerow([paragraph[i], context[i], prediction[i], target[i]])

if __name__ == '__main__':
    inference()