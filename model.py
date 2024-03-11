import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, LoraConfig

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
    return peft_model, tokenizer

def T5():
    model_name = "google-t5/t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def Bart():
    model_name = "facebook/bart-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer 


if __name__ == '__main__':
    pass