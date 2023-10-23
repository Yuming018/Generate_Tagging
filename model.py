import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, LoraConfig

def create_model(model_name):
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
    return peft_model

if __name__ == '__main__':
    pass