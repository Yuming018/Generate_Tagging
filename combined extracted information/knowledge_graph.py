import sys 
sys.path.append("..") 
from helper import enconder, text_segmentation, create_prompt
from coreference_resolution import corf_resolution

import re
from functools import lru_cache
from tqdm import tqdm
from collections import defaultdict
from generation import create_tagging, create_question
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import torch
import itertools

sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

question_keyword = {
    'who' : ['Speaker', 'Actor', 'Addressee', 'Agent', 'Entity'],
    'where' : ['Place'],
    'what' : ['Direct Object', 'Topic(indirect)', 'Msg(Direct)', 'key', 'value', 'Topic'],
    'when' : ['Time'],
    'how' : ['Emotion', 'Trigger_Word'],
    # 'why' : [],
}

# fairytale_question_type = {
#     'character', 'setting', 'causal relationship', 'outcome resolution', 'action', 'feeling', 'prediction'
# }

fairytale_question_type = {
    'Action' : ['character', 'action', 'setting', 'outcome resolution', 'prediction'], 
    'State' : ['character', 'setting'],
    'Causal Effect' : ['feeling', 'causal relationship', 'outcome resolution', 'prediction'],
    'Temporal' : ['outcome resolution'],
}

event_type = [
    'Action - Action Verb', 'Action - Direct Speech Act', 'Action - Indirect Speech Act',
    'State - Emotion', 'State - Thought', 'State - Characteristic',
]

relation_type = [
    'Temporal - the same', 'Temporal - isAfter', 'Temporal - isBefore',
    'Causal Effect - X intent', 'Causal Effect - X need', 'Causal Effect - X attribute', 
    'Causal Effect - Effect on X', 'Causal Effect - X want', 'Causal Effect - X reaction',
    'Causal Effect - Other reaction', 'Causal Effect - Other want', 'Causal Effect - Effect on other',
]

def create_knowledge_graph(gen_answer, 
                           context, 
                           target, 
                           tokenizer, 
                           device, 
                           model_name, 
                           Event_count):

    lines = re.split(r'[,.]', context)
    new_lines = []
    remind_line = ""
    for line in lines:
        remind_line += line
        if len(remind_line.split(" ")) > 10:
            new_lines.append(remind_line)
            remind_line = ""
            
    if len(lines) > 10:
        range_para = 3 + (len(lines)//10)*1
    else:
        range_para = 3
    
    Event_graph, Relation_graph = defaultdict(defaultdict), defaultdict(defaultdict)
    
    #old
    print('\nEvent extraction...')
    for idx in tqdm(range(len(new_lines))):
        temp_context = new_lines[idx]
        text = create_prompt(model_name, generate_type = 'Event', context = temp_context)
        tagging = generate_tagging(f'../save_model/Event/{model_name}/', model_name, text, tokenizer, device)
        Event_graph[idx]['text'] = temp_context
        for tag in tagging.split('[')[1:-1]:
            try:
                key, entity = tag.split(']')
                Event_graph[idx][key] = entity
            except:
                print(tagging)
    
    #new
    # event_set = set()
    # event_idx, event_count = 0, 0
    # while event_count < 5:
    #     text = create_prompt(model_name, generate_type = 'Event', context = context)
    #     tagging = generate_tagging(f'../save_model/Event/{model_name}/', model_name, text, tokenizer, device)
    #     print(event_idx)
    #     if tagging not in event_set:
    #         Event_graph[event_idx]['text'] = context
    #         for tag in tagging.split('[')[1:-1]:
    #             try:
    #                 key, entity = tag.split(']')
    #                 Event_graph[event_idx][key] = entity
    #             except:
    #                 print(tagging)
    #         event_idx += 1
    #         event_count = 0
    #         event_set.add(tagging)
    #     else:
    #         event_count += 1
    
    #old
    # relation_set = set()
    # for idx in tqdm(range(len(lines) - range_para + 1)):
    #     temp_context = ".".join(lines[idx : idx + range_para])
    #     text = create_prompt(model_name, generate_type = 'Relation', context = temp_context)
    #     count = 0

    #     while count < 5:
    #         all_keys = []
    #         tagging = generate_tagging(f'../save_model/Relation/{model_name}/', model_name, text, tokenizer, device)
    #         Relation_graph[idx]['text'] = temp_context
    #         for tag in tagging.split('[')[1:-1]:
    #             try:
    #                 key, entity = tag.split(']')
    #                 if key == 'EVENT1':
    #                     key = 'Event1'
    #                 if key == 'EVENT2':
    #                     key = 'Event2' 
    #                 all_keys.append(key)
    #             except:
    #                 continue

    #         if all_keys == ['Relation 1', 'Event1', 'Event2']:
    #             overlap_flag = False
    #             for temp_tag in relation_set:
    #                 if check_high_overlap(tagging, temp_tag):
    #                     overlap_flag = True
    #             if not overlap_flag:
    #                 break
    #         count += 1
        
    #     if not overlap_flag and all_keys == ['Relation 1', 'Event1', 'Event2']:
    #         for tag in tagging.split('[')[1:-1]:  
    #             try:
    #                 key, entity = tag.split(']')
    #                 if key == 'EVENT1':
    #                     key = 'Event1'
    #                 if key == 'EVENT2':
    #                     key = 'Event2' 
    #                 Relation_graph[idx][key] = entity
    #             except:
    #                 print(tagging)
    #         relation_set.add(tagging)
    
    # new
    print('\nRelation extraction...')
    relation_set = set()
    relation_idx, relation_count = 0, 0
    while relation_count < 5:
        text = create_prompt(model_name, generate_type = 'Relation', context = context)
        tagging = generate_tagging(f'../save_model/Relation/{model_name}/', model_name, text, tokenizer, device)
        print(relation_idx)
        temp = dict()
        all_keys = []
        temp['text'] = context
        for tag in tagging.split('[')[1:-1]:
            try:
                key, entity = tag.split(']')
                if key == 'EVENT1' or key == 'event1':
                    key = 'Event1'
                if key == 'EVENT2' or key == 'event2':
                    key = 'Event2' 
                all_keys.append(key)
                temp[key] = entity
            except:
                print(tagging)
        
        if all_keys == ['Relation 1', 'Event1', 'Event2']:
            overlap_flag = False 
            for temp_tag in relation_set:
                if check_high_overlap(tagging, temp_tag):
                    overlap_flag = True
            # temp_tuple = tuple(temp.items())
            if not overlap_flag :
                Relation_graph[relation_idx] = temp
                relation_idx += 1
                relation_count = 0
                relation_set.add(tagging)
            else:
                relation_count += 1
    
    # segment
    # Event_graph = {0: defaultdict(None, {'text': 'long  long ago there lived in kyoto a brave soldier named kintoki ', 'Event 1': ' Action - Action Verb ', 'Actor': ' kintoki ', 'Place': ' in kyoto ', 'Time': ' long long ago ', 'Trigger_Word': ' lived '}), 1: defaultdict(None, {'text': ' now he fell in love with a beautiful lady and married her ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Direct Object': ' her ', 'Trigger_Word': ' married '}), 2: defaultdict(None, {'text': ' not long after this  through the malice of some of his friends ', 'Event 1': ' Action - Action Verb ', 'Actor': ' some of his friends ', 'Direct Object': ' the malice of ', 'Time': ' not long after this ', 'Trigger_Word': ' through '}), 3: defaultdict(None, {'text': ' he fell into disgrace at court and was dismissed ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Trigger_Word': ' was dismissed '}), 4: defaultdict(None, {'text': ' this misfortune so preyed upon his mind that he did not long survive his dismissal ', 'Event 1': ' Action - Action Verb ', 'Actor': ' this misfortune ', 'Direct Object': ' his mind ', 'Tool or Method': ' so preyed upon ', 'Trigger_Word': ' that he did not long survive his dismissal '}), 5: defaultdict(None, {'text': ' he died  leaving behind him his beautiful young wife to face the world alone ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Direct Object': ' his beautiful young wife ', 'Tool or Method': ' to face the world alone ', 'Trigger_Word': ' leaving behind him '}), 6: defaultdict(None, {'text': " fearing her husband 's enemies  she fled to the ashigara mountains as soon as her husband was dead ", 'Event 1': ' Action - Action Verb ', 'Actor': ' she ', 'Place': ' the ashigara mountains ', 'Time': ' as soon as her husband was dead ', 'Trigger_Word': ' fled to '}), 7: defaultdict(None, {'text': ' there in the lonely forests where no one ever came except woodcutters ', 'Event 1': ' State - Characteristic ', 'Entity': ' there ', 'Key': ' woodcutters ', 'Trigger_Word': ' came ', 'Value': ' no one ever '}), 8: defaultdict(None, {'text': ' a little boy was born to her  she called him kintaro or the golden boy ', 'Event 1': ' Action - Action Verb ', 'Actor': ' she ', 'Direct Object': ' a little boy ', 'Indirect Object (Direct)': ' kintaro or the golden boy ', 'Trigger_Word': ' called '}), 9: defaultdict(None, {'text': ' now the remarkable thing about this child was his great strength ', 'Event 1': ' State - Characteristic ', 'Entity': ' this child ', 'Trigger_Word': ' was ', 'Value': ' his great strength '}), 10: defaultdict(None, {'text': ' and as he grew older he grew stronger and stronger ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Time': ' as he grew older ', 'Tool or Method': ' stronger and stronger ', 'Trigger_Word': ' grew '}), 11: defaultdict(None, {'text': ' by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters ', 'Event 1': ' Action - Action Verb ', 'Actor': ' he ', 'Direct Object': ' trees ', 'Tool or Method': ' as quickly as the woodcutters ', 'Time': ' by the time he was eight years of age ', 'Trigger_Word': ' cut down '}), 12: defaultdict(None, {'text': ' then his mother gave him a large ax  and he used to go out in the forest and help the woodcutters ', 'Event 1': ' Action - Action Verb ', 'Actor': ' his mother ', 'Direct Object': ' him ', 'Indirect Object': ', a large ax ', 'Tool or Method': ' go out in the forest and help the woodcutters ', 'Trigger_Word': ' gave '}), 13: defaultdict(None, {'text': ' who called him " wonder - child  " and his mother the " old nurse of the mountains ', 'Event 1': ' Action - Action Verb ', 'Actor': ' who ', 'Direct Object': ' him ', 'Indirect Object (Direct)': ' the " old nurse of the mountains ', 'Trigger_Word': ' called '}), 14: defaultdict(None, {'text': ' " for they did not know her high rank ', 'Event 1': ' Action - Action Verb ', 'Actor': ' they ', 'Direct Object': ' her high rank ', 'Trigger_Word': ' did not know '}), 15: defaultdict(None, {'text': " another favorite pastime of kintaro 's was to smash up rocks and stones ", 'Event 1': ' Action - Action Verb ', 'Actor': " another favorite pastime of kintaro's ", 'Direct Object': ' rocks and stones ', 'Trigger_Word': ' smash up '})}
    # all paragraph
    # Event_graph = {0: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Key': ' the wonder child ', 'Time': ' long, long ago ', 'Trigger_Word': ' was ', 'Value': ' strong '}), 1: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' Action - Action Verb ', 'Actor': ' kintaro ', 'Direct Object': ' rocks and stones ', 'Time': ' long, long ago ', 'Trigger_Word': ' smash up '}), 2: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Key': " her husband's enemies ", 'Place': ' the ashigara mountains ', 'Trigger_Word': ' fled to '}), 3: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Key': ' his mother gave him a large ax ', 'Trigger_Word': ' used to go out in the forest and help the woodcutters '}), 4: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Key': ' his mother gave him a large ax ', 'Time': ' long, long ago ', 'Trigger_Word': ' used to go out in the forest and help the woodcutters '}), 5: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Key': ' the great strength ', 'Time': ' long, long ago ', 'Trigger_Word': ' was ', 'Value': ' to smash up rocks and stones '}), 6: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Key': ' the distinguished thing about this child was his great strength ', 'Time': ' long, long ago ', 'Trigger_Word': ' was ', 'Value': ' smash up rocks and stones '}), 7: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro or the golden boy ', 'Key': ' they did not know her high rank ', 'Time': ' long, long ago '}), 8: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Time': ' long, long ago ', 'Key': ' his mother gave him a large ax ', 'Trigger_Word': ' used to go out in the forest and help the woodcutters '}), 9: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Key': ' a large ax ', 'Time': ' long, long ago ', 'Trigger_Word': ' used to go out in the forest and help the woodcutters '}), 10: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Key': ' the golden boy ', 'Time': ' long, long ago ', 'Trigger_Word': ' was ', 'Value': ' strong '}), 11: defaultdict(None, {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Event 1': ' State - Characteristic ', 'Entity': ' kintaro ', 'Time': ' long, long ago ', 'Trigger_Word': ' was ', 'Value': ' strong '})}
    # Relation_graph = {0: {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' he fell into disgrace at court and was dismissed', 'Event2': ' his misfortune so preyed upon his mind '}, 1: {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' this misfortune so preyed upon his mind that he did not long survive his dismissal', 'Event2': ' she fled to the ashigara mountains as soon as her husband was dead '}, 2: {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' this misfortune so preyed upon his mind that he did not long survive his dismissal', 'Event2': ' he died '}, 3: {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' through the malice of some of his friends', 'Event2': ' he fell into disgrace at court and was dismissed '}, 4: {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' he fell into disgrace at court and was dismissed', 'Event2': ' this misfortune so preyed upon his mind that he did not long survive his dismissal '}, 5: {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Relation 1': ' Causal Effect - Effect on X ', 'Event1': ' his mother gave him a large ax ', 'Event2': ' he used to go out in the forest and help the woodcutters '}, 6: {'text': 'long , long ago there lived in kyoto a brave soldier named kintoki . now he fell in love with a beautiful lady and married her . not long after this , through the malice of some of his friends , he fell into disgrace at court and was dismissed . this misfortune so preyed upon his mind that he did not long survive his dismissal . he died , leaving behind him his beautiful young wife to face the world alone . fearing her husband \'s enemies , she fled to the ashigara mountains as soon as her husband was dead . there in the lonely forests where no one ever came except woodcutters , a little boy was born to her . she called him kintaro or the golden boy . now the remarkable thing about this child was his great strength , and as he grew older he grew stronger and stronger . by the time he was eight years of age he was able to cut down trees as quickly as the woodcutters . then his mother gave him a large ax , and he used to go out in the forest and help the woodcutters , who called him " wonder - child , " and his mother the " old nurse of the mountains , " for they did not know her high rank . another favorite pastime of kintaro \'s was to smash up rocks and stones . you can imagine how strong he was !', 'Relation 1': ' Causal Effect - Effect on other ', 'Event1': ' through the malice of some of his friends, he fell into disgrace at court and was dismissed', 'Event2': ' this misfortune so preyed upon his mind that he did not long survive his dismissal '}}
    
    # print(Event_graph, '\n')
    # print(Relation_graph)
    # input()

    print('Done !')
    question_set, score_set, text_set, question_difficulty, question_5w1h, generate_question_type = [], [], [], [], [], []
    allennlp_pred = corf_resolution(context)

    print('\nEvent question generation...')
    event_question, event_score, event_text, event_difficulty, event_question_type = connect_event_graph(gen_answer, context, Event_graph, tokenizer, device, target, model_name, allennlp_pred, new_lines, Event_count)
    for question, score, text, difficulty, q_type in zip(event_question, event_score, event_text, event_difficulty, event_question_type):
        if question not in question_set:
            question_set.append(question)
            score_set.append(score)
            text_set.append(text)
            question_difficulty.append(difficulty)
            question_5w1h.append('Event')
            generate_question_type.append(q_type)

    print('\nRelation question generation...')
    relation_question, relation_score, relation_text, relation_difficulty, relation_question_type = connect_relation_graph(gen_answer, context, Relation_graph, tokenizer, device, target, model_name, Event_count)
    for question, score, text, difficulty, q_type in zip(relation_question, relation_score, relation_text, relation_difficulty, relation_question_type):
        if question not in question_set:
            question_set.append(question)
            score_set.append(score)
            text_set.append(text)
            question_difficulty.append(difficulty)
            question_5w1h.append('Relation')
            generate_question_type.append(q_type)

    return question_set, score_set, text_set, question_difficulty, question_5w1h, generate_question_type, Event_graph, Relation_graph

def check_high_overlap(tagging, tagging2):
    tag_dict, tag_dict_2 = defaultdict(str), defaultdict(str)
    for tag in tagging.split('[')[1:-1]:  
        try:
            key, entity = tag.split(']')
            if key == 'EVENT1' or key == 'event1':
                key = 'Event1'
            if key == 'EVENT2' or key == 'event2':
                key = 'Event2' 
            tag_dict[key] = entity
        except:
            print(tagging)
    
    for tag in tagging2.split('[')[1:-1]:  
        key, entity = tag.split(']')
        if key == 'EVENT1' or key == 'event1':
            key = 'Event1'
        if key == 'EVENT2' or key == 'event2':
            key = 'Event2'
        tag_dict_2[key] = entity
    try:
        max_overlap = max(longest_common_subsequence(tag_dict['Event1'], tag_dict_2['Event1']) + longest_common_subsequence(tag_dict['Event2'], tag_dict_2['Event2']), 
                        longest_common_subsequence(tag_dict['Event1'], tag_dict_2['Event2']) + longest_common_subsequence(tag_dict['Event2'], tag_dict_2['Event1']))
    except:
        print(tag_dict)
        print(tag_dict_2)
        max_overlap = 2
    return max_overlap > 1.6

def generate_tagging(path_save_model, model_name, text, tokenizer, device):
    input_ids = enconder(tokenizer, max_len=1024, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    tagging_ids = create_tagging(path_save_model = path_save_model, model_name = model_name, input_ids = input_ids, device = device)
    tagging = tokenizer.decode(tagging_ids, skip_special_tokens=True)

    return tagging

def generate_question(model_name, gen_answer, text, target, tokenizer, device):
    if gen_answer:
        path_save_model = f'../save_model/QA_pair/{model_name}/'
    elif not gen_answer:
        path_save_model = f'../save_model/question/{model_name}/'
    input_ids = enconder(tokenizer, max_len=1024, text=text)
    input_ids = torch.tensor(input_ids.get('input_ids')).to(device)

    question_ids = create_question(path_save_model = path_save_model, model_name = model_name, input_ids = input_ids, device = device)
    question = tokenizer.decode(question_ids, skip_special_tokens=True)
    score = eval(question, target)

    return question, score

def connect_event_graph(gen_answer, context, Event_graph, tokenizer, device, target, model_name, allennlp_pred, new_lines, Event_count):
    
    event_question_set, event_score, event_text, event_difficulty, event_question_type = [], [], [], [], []
    for idx in tqdm(Event_graph):
        if 'Action' in Event_graph[idx]['Event 1']:
            current_question_type = fairytale_question_type['Action']
        elif 'State' in Event_graph[idx]['Event 1']:
            current_question_type = fairytale_question_type['State']

        if len(Event_graph[idx]['text'].split(' ')) <= 3:
            continue

        for q_type in current_question_type:
            answer_list = set()
            text = create_prompt(model_name, generate_type = 'question', context = context, question_type = q_type)
            for key, entity in Event_graph[idx].items():
                if key != 'text':
                    text += f'[{key}] {entity}'
                if 'Action' in Event_graph[idx]['Event 1'] and key in ['Direct Object', 'Actor', 'Msg (Direct)', 'Place', 'Tool or Method']:
                    answer_list.add(entity)
                if 'State' in Event_graph[idx]['Event 1'] and key in ['Emotion', 'Value', 'Entity', 'Key', 'Topic (Indirect)']:
                    answer_list.add(entity)                
            
            have_arg = 0
            corf_range = 0
            for arg in ['Speaker', 'Actor', 'Direct Object', 'Addressee', 'Agent', 'Entity']:
                if arg in Event_graph[idx].keys():
                    subject = " ".join(Event_graph[idx][arg].split(' ')[1:-1])
                    if len(subject):
                        corf, final_subject, difficulty = check_corf(subject, allennlp_pred, new_lines, idx)
                    if corf != "":
                        text += corf
                        have_arg += 1
                        corf_range = max(corf_range, difficulty)
                        if Event_graph[idx][arg] in answer_list:
                            answer_list.remove(Event_graph[idx][arg])
                            answer_list.add(final_subject)
            
            if have_arg == Event_count - 1:
                if not gen_answer:
                    for answer in answer_list:
                        fianl_text = text + f'[Answer] {answer} [END]'
                        event_question, e_score = generate_question(model_name, gen_answer, fianl_text, target, tokenizer, device)
                        event_question_set.append(event_question)
                        event_score.append(e_score)
                        event_text.append(fianl_text)
                        event_difficulty.append(corf_range)
                        event_question_type.append(q_type)
                else:
                    fianl_text = text + f' [END]'
                    event_question, e_score = generate_question(model_name, gen_answer, fianl_text, target, tokenizer, device)
                    event_question_set.append(event_question)
                    event_score.append(e_score)
                    event_text.append(fianl_text)
                    event_difficulty.append(corf_range)
                    event_question_type.append(q_type)
                
                                        
    return event_question_set, event_score, event_text, event_difficulty, event_question_type

def check_corf(subject, allennlp_pred, new_lines, idx):
    max_ratio = 0
    
    text_len = len(new_lines[idx].split(' '))
    for word_idx in range(len(allennlp_pred['document'])):
        text = ' '.join(allennlp_pred['document'][word_idx:word_idx+text_len])
        match_len = longest_common_subsequence_entity(text, new_lines[idx])
        ratio = match_len / len(new_lines[idx])
        if max_ratio < ratio:
            max_ratio = ratio
            boundary_front = word_idx
            boundary_back = word_idx + text_len
    
    max_ratio = 0
    pronoun, final_subject = "", ""
    final_subject_idx, pronoun_idx = 0, 0
    for entity_name in allennlp_pred['coreference']:
        for span in allennlp_pred['coreference'][entity_name]:
            entity = ' '.join(allennlp_pred['document'][span[0]:span[1]+1])
            match_len = longest_common_subsequence_entity(subject, entity)
            ratio = match_len / len(subject)
            if ratio > 0.8 and max_ratio < ratio and span[0] >= boundary_front and span[1] < boundary_back:
                max_ratio = ratio
                final_subject = entity
                final_subject_idx = span[0]
                pronoun = entity_name
                pronoun_idx = allennlp_pred['coreference'][entity_name][0][0]
    
    corf_range = 0
    if pronoun != "":
        for word_idx in range(min(pronoun_idx, final_subject_idx), max(pronoun_idx, final_subject_idx)):
            if ',' in allennlp_pred['document'][word_idx] or '.' in allennlp_pred['document'][word_idx]:
                corf_range += 1

    return (f'[Relation1] Coreference - Coref [Arg1] {final_subject} [Arg2] {pronoun} ', pronoun, corf_range) if pronoun != "" and final_subject != pronoun else ("", "", 0)

def longest_common_subsequence_entity(entity1, entity2):
    event1_len = len(entity1)
    event2_len = len(entity2)
    longest = 0
    lcs_table = [[0] * (event2_len + 1) for _ in range(event1_len + 1)]

    for i in range(1, event1_len + 1):
        for j in range(1, event2_len + 1):
            if entity1[i - 1] == entity2[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                longest = max(longest, lcs_table[i][j])
            else:
                lcs_table[i][j] = 0
    return longest

def connect_relation_graph(gen_answer, context, Relation_graph, tokenizer, device, target, model_name, Event_count):
    graph = defaultdict(list)
    if len(Relation_graph) > 30:
        return ['too many'], ['too many'], ['too many'], ['too many'], ['too many']
    
    for idx in Relation_graph:
        if ['text', 'Relation 1', 'Event1', 'Event2'] != list(Relation_graph[idx].keys()):
            continue
        relation_dfs(idx, [], Relation_graph, graph[idx], set())
    

    event_chain_list = []
    for idx in Relation_graph:
        max_len = 0
        for event_chain in graph[idx]:
            if len(event_chain) >= Event_count and max_len < len(event_chain):
                max_len = len(event_chain)
                max_event_chain = event_chain
        if max_len >= Event_count:
            event_chain_list.append(max_event_chain)

    lines = re.split(r'[,.]', context)
    
    relation_question_set, relation_score, relation_text, relation_difficulty, relation_question_type = [], [], [], [], []
    combination_set = set()
    for event_chain in tqdm(event_chain_list):
        # event_combinations = list(itertools.combinations(event_chain, Event_count))
        event_combinations = []
        for idx in range(len(event_chain)- Event_count + 1):
            event_combinations.append(deepcopy(event_chain[idx : idx+Event_count]))

        for combination in event_combinations:
            tuple_combination = tuple(combination)
            if tuple_combination in combination_set:
                continue
            combination_set.add(tuple_combination)

            current_question_type = []
            for idx in combination:
                if 'Causal Effect' in Relation_graph[idx]["Relation 1"]:
                    current_question_type += fairytale_question_type['Causal Effect']
                if 'Temporal' in Relation_graph[idx]["Relation 1"]:
                    current_question_type += fairytale_question_type['Temporal']
            current_question_type = list(set(current_question_type))

            for q_type in current_question_type:
                min_event_idx, max_event_idx, count = 1000, 0, 1
                answer_list = set()
                text = create_prompt(model_name, generate_type = 'question', context = context, question_type = q_type)
                for idx, relation_idx in enumerate(combination):
                    text += f'[Relation {count}] {Relation_graph[relation_idx]["Relation 1"]} [Event 1] {Relation_graph[relation_idx]["Event1"]} [Event 2] {Relation_graph[relation_idx]["Event2"]}'
                    
                    # if 'Causal Effect' in Relation_graph[relation_idx]["Relation 1"]:
                    #     answer_list.add(Relation_graph[relation_idx]["Event1"])
                    #     answer_list.add(Relation_graph[relation_idx]["Event2"])
                    # elif 'Temporal' in Relation_graph[relation_idx]["Relation 1"]:
                    #     answer_list.add(Relation_graph[relation_idx]["Event2"])
                    if idx == 0:
                        answer_list.add(Relation_graph[relation_idx]["Event1"])
                    elif idx == len(combination) - 1:
                        answer_list.add(Relation_graph[relation_idx]["Event2"])
                    
                    count += 1
                    try:
                        min_event_idx = min(check_event_idx(lines, Relation_graph[relation_idx]["Event1"]), min_event_idx)
                        min_event_idx = min(check_event_idx(lines, Relation_graph[relation_idx]["Event2"]), min_event_idx)
                        max_event_idx = max(check_event_idx(lines, Relation_graph[relation_idx]["Event1"]), max_event_idx)
                        max_event_idx = max(check_event_idx(lines, Relation_graph[relation_idx]["Event2"]), max_event_idx)
                    except:
                        print(lines)
                        print(Relation_graph[relation_idx]["Event1"], '\n')
                        print(Relation_graph[relation_idx]["Event2"], '\n')
                        print(check_event_idx(lines, Relation_graph[relation_idx]["Event1"]), check_event_idx(lines, Relation_graph[relation_idx]["Event2"]))
                        # input() 
                
                if not gen_answer:
                    for answer in answer_list:
                        fianl_text = text + f'[Answer] {answer} [END]'

                        relation_question, r_score = generate_question(model_name, gen_answer, fianl_text, target, tokenizer, device)
                        relation_question_set.append(relation_question)
                        relation_score.append(r_score)
                        relation_text.append(fianl_text)
                        relation_difficulty.append(max_event_idx - min_event_idx)
                        relation_question_type.append(q_type)
                else:
                    fianl_text = text + f' [END]'
                    relation_question, r_score = generate_question(model_name, gen_answer, fianl_text, target, tokenizer, device)
                    relation_question_set.append(relation_question)
                    relation_score.append(r_score)
                    relation_text.append(fianl_text)
                    relation_difficulty.append(max_event_idx - min_event_idx)
                    relation_question_type.append(q_type)
    
    if not relation_question_set:
        relation_question_set.append('question')
        relation_score.append(0)
        relation_text.append('text')
        relation_difficulty.append(0)
        relation_question_type.append('type')
            
    return relation_question_set, relation_score, relation_text, relation_difficulty, relation_question_type

@lru_cache(maxsize=None)
def cached_lcs(event1, event2):
    return longest_common_subsequence(event1, event2)

def relation_dfs(idx, chains, Relation_graph, graph, visited_chains): 
    if idx in chains:
        return
    chains.append(idx)

    orinigal_event1 = Relation_graph[idx]['Event1']
    orinigal_event2 = Relation_graph[idx]['Event2']
    for next_idx in Relation_graph:
        if next_idx in chains or set(Relation_graph[next_idx].keys()) != {'text', 'Relation 1', 'Event1', 'Event2'}:
            continue
        next_event1 = Relation_graph[next_idx]['Event1']
        next_event2 = Relation_graph[next_idx]['Event2']
        if (cached_lcs(orinigal_event1, next_event1) > 0.8 or 
            cached_lcs(orinigal_event1, next_event2) > 0.8 or 
            cached_lcs(orinigal_event2, next_event1) > 0.8 or 
            cached_lcs(orinigal_event2, next_event2) > 0.8 ):
            relation_dfs(next_idx, chains, Relation_graph, graph, visited_chains)
    
    chain_tuple = tuple(chains)
    if chain_tuple not in visited_chains:
        graph.append(deepcopy(chains))
        visited_chains.add(chain_tuple)
    chains.pop()
    return 

def longest_common_subsequence(event1, event2):
    event1_len = len(event1)
    event2_len = len(event2)
    longest = 0
    lcs_table = [[0] * (event2_len + 1) for _ in range(event1_len + 1)]

    for i in range(1, event1_len + 1):
        for j in range(1, event2_len + 1):
            if event1[i - 1] == event2[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                longest = max(longest, lcs_table[i][j])
            else:
                lcs_table[i][j] = 0
    return longest / (event1_len + event2_len - longest)

def check_event_idx(lines, text):
    # text_list = re.split(r'[,.]', text) # 擷取出的資訊有逗號，會導致 mapping 不到
    # for text in text_list:  # 避免擷取到錯誤資訊
    match_ratio, match_idx = 0, 0
    for idx, line in enumerate(lines):
        ratio = longest_common_subsequence(line, text)
        if match_ratio < ratio:
            match_ratio = ratio
            match_idx = idx
    return match_idx

def eval(pred, tar):
    """
    Input : 預測出來的問題，以及 Fairytale QA 的原始問題
    Output : 返回兩者的相似性分數
    """
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    query_embedding = model.encode(pred)
    passage_embedding = model.encode(tar)
    result = util.dot_score(query_embedding, passage_embedding)
    return round(result[0][0].item(), 2)

if __name__ == '__main__':

    tag_dict, tag_dict_2 = defaultdict(str), defaultdict(str)
    tag_dict['Event1'] = ' his mother gave him a large ax '
    tag_dict['Event2'] = ''
    tag_dict_2['Event1'] = ' this misfortune so preyed upon his mind'
    tag_dict_2['Event2'] = ''

    max_overlap = max(longest_common_subsequence(tag_dict['Event1'], tag_dict_2['Event1']) + longest_common_subsequence(tag_dict['Event2'], tag_dict_2['Event2']), 
                      longest_common_subsequence(tag_dict['Event1'], tag_dict_2['Event2']) + longest_common_subsequence(tag_dict['Event2'], tag_dict_2['Event1']))