import os
import re
import google.generativeai as genai
from openai import OpenAI
import configparser

def checkdir(path_save_model, Generation, model_name, gen_answer):
    
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    
    if Generation == 'Event' or Generation == 'Relation':
        path_save_model += f'/{Generation}'
    elif Generation == 'question':
        if gen_answer:
            path_save_model += '/QA_pair'
        elif not gen_answer:
            path_save_model += '/Question'
    elif Generation == 'answer':
        path_save_model += '/Answer'
    elif Generation == 'ranking':
        if gen_answer:
            path_save_model += '/Ranking_w_ans'
        elif not gen_answer:
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
        # padding = 'max_length',   
        max_length = max_len,        
        #return_tensors='pt',           
        return_attention_mask=True      
        )
    return encoded_sent

def check_checkpoint(path_save_model):
    checkpoints_list = [int(f.split('-')[1]) for f in os.listdir(path_save_model + 'checkpoints')]
    model_path = path_save_model + f'checkpoints/checkpoint-{max(checkpoints_list)}/'
    return model_path

def create_prompt(model_name, generate_type, context, question_type = None, gen_Answer = False):

    if generate_type == 'Event' or generate_type == 'Relation':
        if model_name == 'Mt0' or model_name == 'gemma' or model_name == 'flant5':
            text = f"Please utilize the provided context to generate {generate_type} 1 key information for this context [Context] {context} [END]"
        elif model_name == 'T5' or model_name == 'Bart':
            text = f"[Context] {context} [END]"
        elif model_name == 'roberta':
            question = f'What {generate_type} key information is included in this context and explain their subjects, objects, and their possible types?'
            text = (question, context)
    elif generate_type == 'question':
        if model_name == 'Mt0' or model_name == 'gemma' or model_name == 'flant5':
            if not gen_Answer:
                text = f"Please utilize the provided context, answer and key information to generate {question_type} question for this context [Context] {context} "
            elif gen_Answer:
                text = f"Please utilize the provided context and key information to generate {question_type} question and answer for this context [Context] {context} "
        elif model_name == 'T5' or model_name == 'Bart' or model_name == 'roberta':
            text = f"[Question type] {question_type} [Context] {context} "
    elif generate_type == 'ranking':
        if model_name == 'distil':
            text = f"[CLS] {context} "
    
    return text

def check_config():
    config = configparser.ConfigParser()
    current_directory = os.getcwd()
    if current_directory.endswith("combined extracted information"):
        config_path = "../config.ini"
    else:
        config_path = "config.ini"
    config.read(config_path)
    return config

def gptapi(content, version = 3.5, temperature = 0.5, seed = 1234):
    config = check_config()
    client = OpenAI(
        api_key = config['key']['openai_api_key'],
        organization = config['key']['Organization_ID']
    )
    
    completion = client.chat.completions.create(
        model=f"gpt-{version}-turbo",
        temperature = temperature,
        seed = seed,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}],
        )
    return completion.choices[0].message.content

def geminiapi(content, temperature = 0):
    config = check_config()
    genai.configure(api_key=config['key']['google_api_key'])
    safetySettings =  [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
            ]
    generationConfig =  {
        "temperature": temperature, # 控制輸出的隨機性
        "maxOutputTokens": 500
        # "topP": 0.8,
        # "topK": 10
    }

    model = genai.GenerativeModel('gemini-pro', safety_settings = safetySettings)    
    response = model.generate_content(content)
    return response.text

if __name__ == '__main__':
    pass