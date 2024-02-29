import os
import re

def checkdir(path_save_model, event_or_relation, Generation):
    
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    
    if event_or_relation == 'Relation' :
        path_save_model += '/Relation'
    elif event_or_relation == 'Event' :
        path_save_model += '/Event'
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    
    if Generation == 'tagging' :
        path_save_model += '/tagging/'
    elif Generation == 'question':
        path_save_model += '/question/'
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
        padding = 'longest',   
        max_length = max_len,        
        #return_tensors='pt',           
        return_attention_mask=True      
        )
    return encoded_sent

def check_checkpoint(path_save_model):
    checkpoints_list = [int(f.split('-')[1]) for f in os.listdir(path_save_model + 'checkpoints')]
    model_path = path_save_model + f'checkpoints/checkpoint-{max(checkpoints_list)}/'
    return model_path

if __name__ == '__main__':
    pass