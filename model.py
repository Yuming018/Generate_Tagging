import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, LoraConfig

def create_model(model_name):
    if model_name == 'Mt0':
        model, tokenizer, model_name = Mt0()
    elif model_name == 'T5':
        model, tokenizer, model_name = T5()
    elif model_name == 'flant5':
        model, tokenizer, model_name = flant5()
    elif model_name == 'Bart':
        model, tokenizer, model_name = Bart()
    elif model_name == 'roberta':
        model, tokenizer, model_name = roberta()
    elif model_name == 'gemma':
        model, tokenizer, model_name = gemma()
    
    print('Model name :', model_name, '\n')
    return model, tokenizer

def Mt0():
    model_name = "bigscience/mt0-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32, 
        lora_dropout=0.1,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.config.use_cache = False
    return peft_model, tokenizer, model_name

def T5():
    model_name = "google-t5/t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer, model_name

def flant5():
    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer, model_name

def Bart():
    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer, model_name

def roberta():
    model_name = "deepset/roberta-large-squad2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer, model_name

def gemma():
    model_name = "google/gemma-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=12,
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.config.use_cache = False
    return peft_model, tokenizer, model_name

if __name__ == '__main__':
    pass