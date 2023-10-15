import os

def checkdir(path_save_model, relation_tag):
    if relation_tag :
        target_path = 'Relation/'
    else:
        target_path = 'Event/'

    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    path_save_model = path_save_model + target_path
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    return path_save_model

def enconder(tokenizer, max_len=256, text = ''):
    encoded_sent = tokenizer.encode_plus(
        text = text,  
        add_special_tokens=True,
        truncation=True,  
        padding = 'longest' if max_len < 512 else 'max_length',   
        # max_length='longest' if max_len < 512 else 512,        
        #return_tensors='pt',           
        return_attention_mask=True      
        )
    return encoded_sent

if __name__ == '__main__':
    pass