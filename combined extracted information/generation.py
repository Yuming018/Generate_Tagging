from helper import check_checkpoint
from peft import PeftConfig, PeftModel
from transformers import GenerationConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def create_generate_config(model, generate):
    if generate == 'tagging':
        generation_config = GenerationConfig(
                num_beams=3,
                early_stopping=True,
                decoder_start_token_id=0,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.eos_token_id,
                penalty_alpha=0.6, 
                no_repeat_ngram_size=3,
                top_k=300, 
                top_p = 0.95,
                temperature=1.5,
                max_new_tokens=128,
                do_sample=True,
                repetition_penalty = 1.8,
            )
    elif generate == 'question':
        generation_config = GenerationConfig(
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
                do_sample=True,
            )
    return generation_config

def set_model(model_name, model_path):
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
    
    return model

def create_tagging(path_save_model, model_name, input_ids, device):
    model_path = check_checkpoint(path_save_model)
    model = set_model(model_name, model_path).to(device)

    generation_config = create_generate_config(model, generate = 'tagging')
    output = model.generate(input_ids=input_ids.unsqueeze(0), generation_config=generation_config)
    return output[0]

def create_question(path_save_model, model_name, input_ids, device):
    model_path = check_checkpoint(path_save_model)
    model = set_model(model_name, model_path).to(device)

    generation_config = create_generate_config(model, generate = 'question')
    output = model.generate(input_ids=input_ids.unsqueeze(0), 
                            generation_config=generation_config
                            )
    return output[0]

if __name__ == '__main__':
    pass