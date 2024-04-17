import os
import re

def checkdir(path_save_model, event_or_relation, Generation, model_name):
    
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    
    if Generation == 'tagging' :
        path_save_model += f'/{event_or_relation}'
        if not os.path.isdir(path_save_model):
            os.mkdir(path_save_model)
        path_save_model += '/tagging'
    elif Generation == 'question':
        path_save_model += '/Question'
    elif Generation == 'ranking':
        path_save_model += '/Ranking'
    
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    
    path_save_model += f'/{model_name}/'

    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)

    return path_save_model

def text_segmentation(data):
        min_b, max_b = float('inf'), float('-inf')
        for i in range(7, len(data)):
            if data[i] != '':
                numbers = re.findall(r'\d+', data[i])
                min_b = min(min_b, int(numbers[-2]))
                max_b = max(max_b, int(numbers[-1]))
        words = re.findall(r'\S+|[\s]+', data[1])
        if words[min_b] == ' ':
            min_b -= 1
        return "".join(words[min_b:max_b])

def enconder(tokenizer, max_len=256, text = ''):
    encoded_sent = tokenizer.encode_plus(
        text = text,  
        add_special_tokens=True,
        truncation=True,  
        padding = 'max_length',   
        max_length = max_len,        
        #return_tensors='pt',           
        return_attention_mask=True      
        )
    return encoded_sent

def check_checkpoint(path_save_model):
    checkpoints_list = [int(f.split('-')[1]) for f in os.listdir(path_save_model + 'checkpoints')]
    model_path = path_save_model + f'checkpoints/checkpoint-{max(checkpoints_list)}/'
    return model_path

def create_prompt(model_name, tagging_type, generate_type, context):

    if generate_type == 'tagging':
        if model_name == 'Mt0' or model_name == 'gemma':
            text = f"Please utilize the provided context to generate {tagging_type} 1 key information for this context [Context] {context} [END]"
        elif model_name == 'T5' or model_name == 'flant5' or model_name == 'Bart':
            text = f"[Context] {context} [END]"
        elif model_name == 'roberta':
            question = f'What {tagging_type} key information is included in this context and explain their subjects, objects, and their possible types?'
            text = (question, context)
    elif generate_type == 'question':
        if model_name == 'Mt0' or model_name == 'gemma':
            text = f"Please utilize the provided context and key information to generate question for this context [Context] {context} "
        elif model_name == 'T5' or model_name == 'flant5'  or model_name == 'Bart' or model_name == 'roberta':
            text = f"[Context] {context} "
    elif generate_type == 'ranking':
        if model_name == 'Mt0' or model_name == 'gemma':
            text = f"Please use the provided context to determine whether the question can be answered [Context] {context} "
        elif model_name == 'T5' or model_name == 'flant5'  or model_name == 'Bart' or model_name == 'roberta':
            text = f"[Context] {context} "
    
    return text

if __name__ == '__main__':
    pass