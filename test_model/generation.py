from helper import check_checkpoint
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

def create_tagging(path_save_model, model, input_ids):
    model_path = check_checkpoint(path_save_model)
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"":0})
    model = PeftModel.from_pretrained(model, model_path, device_map={"":0})
    tagging_generation_config = create_generate_config(model)
    output = model.generate(input_ids=input_ids.unsqueeze(0), generation_config=tagging_generation_config)
    return output[0]

def create_question(path_save_model, model, input_ids):
    model_path = check_checkpoint(path_save_model)
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"":0})
    model = PeftModel.from_pretrained(model, model_path, device_map={"":0})
    tagging_generation_config = create_generate_config(model)
    output = model.generate(input_ids=input_ids.unsqueeze(0), generation_config=tagging_generation_config)
    return output[0]

if __name__ == '__main__':
    pass