import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from peft import get_peft_model, PeftModel, TaskType, LoraConfig
from helper import check_checkpoint

def create_model(model_name, Generation, test_mode, path_save_model):
    if Generation == 'ranking' :
        if model_name != 'distil' and model_name != 'deberta':
            raise TypeError("Ranking model 只包含 distil, deberta")
    elif Generation == 'tagging' or Generation == 'question':
        if model_name == 'distil' and model_name != 'deberta':
            raise TypeError(f"{Generation} model 不包含 distil, deberta")

    if test_mode:
        model_path = check_checkpoint(path_save_model)
    elif not test_mode:
        model_path = None

        
    if model_name == 'Mt0':
        model, tokenizer, model_name = Mt0(test_mode, model_path)
    elif model_name == 'T5':
        model, tokenizer, model_name = T5(test_mode, model_path)
    elif model_name == 'T5_base':
        model, tokenizer, model_name = T5_base(test_mode, model_path)
    elif model_name == 'T5_small':
        model, tokenizer, model_name = T5_small(test_mode, model_path)
    elif model_name == 'flant5':
        model, tokenizer, model_name = flant5(test_mode, model_path)
    elif model_name == 'bart':
        model, tokenizer, model_name = Bart(test_mode, model_path)
    elif model_name == 'bert':
        model, tokenizer, model_name = bert(test_mode, model_path)
    elif model_name == 'roberta':
        model, tokenizer, model_name = roberta(test_mode, model_path)
    elif model_name == 'gemma':
        model, tokenizer, model_name = gemma(test_mode, model_path)
    elif model_name == 'distil':
        model, tokenizer, model_name = DistilBERT(test_mode, model_path)
    elif model_name == 'deberta':
        model, tokenizer, model_name = deberta(test_mode, model_path)
    
    print('Model name :', model_name, '\n')
    return model, tokenizer

def DistilBERT(test_mode, model_path):
    model_name = "distilbert/distilbert-base-uncased"
    id2label = {0: 'Can not answer', 1: 'Can answer'}
    label2id = {'Can not answer': 0, 'Can answer': 1}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    elif test_mode:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer, model_name

def deberta(test_mode, model_path):
    model_name = "microsoft/deberta-v3-base"
    id2label = {0: 'Can not answer', 1: 'Can answer'}
    label2id = {'Can not answer': 0, 'Can answer': 1}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    elif test_mode:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer, model_name

def roberta(test_mode, model_path):
    model_name = "deepset/roberta-base-squad2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif test_mode:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer, model_name

def Mt0(test_mode, model_path):
    model_name = "bigscience/mt0-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=12,
        lora_alpha=32, 
        lora_dropout=0.1,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.config.use_cache = False

    if test_mode:
        peft_model = PeftModel.from_pretrained(model, model_path)

    return peft_model, tokenizer, model_name

def T5(test_mode, model_path):
    model_name = "google-t5/t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer, model_name

def T5_base(test_mode, model_path):
    model_name = "google-t5/t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer, model_name

def T5_small(test_mode, model_path):
    model_name = "google-t5/t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer, model_name

def flant5(test_mode, model_path):
    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer, model_name

def Bart(test_mode, model_path):
    model_name = "facebook/bart-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif test_mode:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer, model_name

def bert(test_mode, model_path):
    model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not test_mode:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif test_mode:
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    return model, tokenizer, model_name

def gemma(test_mode, model_path):
    model_name = "google/gemma-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=12,
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_model = get_peft_model(model, lora_config)
    peft_model.config.use_cache = False
    if test_mode:
        peft_model = PeftModel.from_pretrained(model, model_path)

    return peft_model, tokenizer, model_name

if __name__ == '__main__':
    pass