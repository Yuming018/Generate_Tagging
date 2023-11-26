import os

def checkdir(path_save_model, relation_tag, tagging):
    
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    
    if relation_tag :
        path_save_model += '/Relation'
    else:
        path_save_model += '/Event'
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    
    if tagging :
        path_save_model += '/tagging/'
    else:
        path_save_model += '/question/'
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)

    return path_save_model

def enconder(tokenizer, max_len=256, text = ''):
    encoded_sent = tokenizer.encode_plus(
        text = text,  
        add_special_tokens=True,
        truncation=True,  
        padding = 'longest' if max_len < 512 else 'max_length',   
        max_length = max_len,        
        #return_tensors='pt',           
        return_attention_mask=True      
        )
    return encoded_sent

if __name__ == '__main__':
    pass