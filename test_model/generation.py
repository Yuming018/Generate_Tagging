from helper import checkdir
import torch
from peft import PeftConfig, PeftModel
from transformers import GenerationConfig, AutoModelForSeq2SeqLM

def create_generate_config(model):
    tagging_generation_config = GenerationConfig(
            num_beams=4,
            early_stopping=True,
            decoder_start_token_id=0,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.eos_token_id,
            penalty_alpha=0.6, 
            no_repeat_ngram_size=3,
            top_k=4, 
            temperature=0.5,
            max_new_tokens=128,
        )
    return tagging_generation_config

def generate_relation_tagging(path_save_model, model, input_ids, relation_tag):
    config = PeftConfig.from_pretrained(path_save_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"":0})
    model = PeftModel.from_pretrained(model, path_save_model, device_map={"":0})
    tagging_generation_config = create_generate_config(model)
    output = model.generate(input_ids=input_ids.unsqueeze(0), generation_config=tagging_generation_config)
    return output[0]

def generate_event_tagging(path_save_model, model, input_ids, relation_tag):
    config = PeftConfig.from_pretrained(path_save_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"":0})
    model = PeftModel.from_pretrained(model, path_save_model, device_map={"":0})
    tagging_generation_config = create_generate_config(model)
    output = model.generate(input_ids=input_ids.unsqueeze(0), generation_config=tagging_generation_config)
    return output[0]

def generate_question(path_save_model, relation_tag, model, tagging_ids):
    path_save_model = checkdir(path_save_model, relation_tag)
    best_pth = path_save_model + 'question.pth'
    model.load_state_dict(torch.load(best_pth))
    output = model.generate(tagging_ids, num_beams=2, min_length=0)
    return output

if __name__ == '__main__':
    pass