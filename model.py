import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, LoraConfig

def create_model(model_name, Generation):
    if Generation == 'ranking' :
        if model_name != 'distil' and model_name != 'deberta':
            raise TypeError("Ranking model 只包含 distil, deberta")
    elif Generation == 'tagging' or Generation == 'question':
        if model_name == 'distil' and model_name != 'deberta':
            raise TypeError(f"{Generation} model 不包含 distil, deberta")

    if model_name == 'Mt0':
        model, tokenizer, model_name = Mt0()
    elif model_name == 'T5':
        model, tokenizer, model_name = T5()
    elif model_name == 'flant5':
        model, tokenizer, model_name = flant5()
    elif model_name == 'bart':
        model, tokenizer, model_name = Bart()
    elif model_name == 'bert':
        model, tokenizer, model_name = bert()
    elif model_name == 'gemma':
        model, tokenizer, model_name = gemma()
    elif model_name == 'distil':
        model, tokenizer, model_name = DistilBERT()
    elif model_name == 'deberta':
        model, tokenizer, model_name = deberta()
    
    print('Model name :', model_name, '\n')
    return model, tokenizer

def DistilBERT():
    model_name = "distilbert/distilbert-base-uncased"
    id2label = {0: 'Can not answer', 1: 'Can answer'}
    label2id = {'Can not answer': 0, 'Can answer': 1}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    return model, tokenizer, model_name

def deberta():
    model_name = "microsoft/deberta-v3-base"
    id2label = {0: 'Can not answer', 1: 'Can answer'}
    label2id = {'Can not answer': 0, 'Can answer': 1}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    return model, tokenizer, model_name

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

def bert():
    model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
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