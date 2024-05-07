
import configparser
import argparse
import pandas as pd
import csv
from tqdm import tqdm
from helper import checkdir, text_segmentation, gptapi, geminiapi

config = configparser.ConfigParser()
config.read('config.ini')

legal_tagging = ['Causal Effect',
            'Temporal',
            # 'Coreference'
            'State',
            'Action'
]

Event_definition = {
    'Action - Action Verb' : "Describe an action performed by a person. The available tags are as follows : Trigger_Word, Actor, Direct Object, Indirect Object, Time, Place, Tool or Method .",
    'Action - Indirect Speech Act' : "Describe a person speaking indirectly. The available tags are as follows : Trigger_Word, Addressee, Speaker, Topic (Indirect), Time, Place .",
    'Action - Direct Speech Act' : "Describe a person speaking directly. The available tags are as follows : Trigger_Word, Addressee, Speaker, Msg (Direct), Time, Place .",
    'State - Thought' : "Describe a person's thoughts. The available tags are as follows : Trigger_Word, Time, Place, Agent, Topic .",
    'State - Emotion' : "Describe a person's emotions. The available tags are as follows : Trigger_Word, Time, Place, Agent, Emotion, Emotion_Type .",
    'State - Characteristic' : "Describe the characteristics of a person or an event. The available tags are as follows : Trigger_Word, Time, Place, Entity, Key, Value .",
}

Relation_definition = {
    'Causal Effect - X intent' : "Why does X cause the event?",
    'Causal Effect - X reaction' : "How does X feel after the event?",
    'Causal Effect - Other reaction' : "How do others' feel after the event?",
    'Causal Effect - X attribute' : "How would X be described?",
    'Causal Effect - X need' : "What does X need to do before the event?",
    'Causal Effect - Effect on X' : "What effects does the event have on X?",
    'Causal Effect - X want' : "What would X likely want to do after the event?",
    'Causal Effect - Other want' : "What would others likely want to do after the event?",
    'Causal Effect - Effect on other' : "What effects does the event have on others?",
    'Temporal - isBefore' : "No causal relationship exists; Event1 occurs before Event2",
    'Temporal - the same' : "No causal relationship exists; Event1 and Event2 occur simultaneously",
    'Temporal - isAfter' : "No causal relationship exists; Event1 occurs after Event2",
}

prompt = {
    'Event' : f"You are currently a tagger. Please help me extract the subject, verb, object, and some emotional words from the given text. Some sentences contain emotional words, please extract them as well.\
        Here are some examples:\
        Example 1 :\
        Content: were only born yesterday know very little . i can see that in you . i have age and experience . i know every one here in the house , and i know there was once a time when i did not lie out here in the cold , fastened to a chain . away , away ! ' ' the cold is delightful , ' said the snow man\
        Result :  [Event 1] Action - Action Verb [Actor] the snow man [Time] yesterday [Trigger_Word] were only born  [END]\
        Example 2 :\
        Content: they would tell you that long , long ago you would have met fishes on the land\
        Result : [Event 1] Action - Indirect Speech Act [Addressee] you [Speaker] they [Topic (Indirect)]  long , long ago you would have met fishes on the land [Trigger_Word] tell  [END]\
        Example 3 :\
        Content: the youngest of the tribe , bowed himself before thuggai , saying , ' ask my father , guddhu the cod , to light the fire . he is skilled in magic more than most fishes . '\
        Result : [Event 1] Action - Direct Speech Act [Msg (Direct)]  ' ask my father , guddhu the cod , to light the fire . he is skilled in magic more than most fishes . ' [Speaker] the youngest of the tribe [Trigger_Word] saying  [END]\
        Example 4 :\
        Content: william thought it must have strayed from the flock\
        Result : [Event 1] State - Thought [Agent] william [Topic] it must have strayed from the flock [Trigger_word] thought  [END]\
        Example 5 :\
        Content: the great confusion of the wearer\
        Result : [Event 1] State - Emotion [Agent]  the wearer [Emotion] the great confusion   [END]\
        Example 6 :\
        Content: him below at dungannon .\
        Result : [Event 1] State - Characteristic [Entity] him [Place] below at dungannon .  [END]\
        Event type definition : ",

    'Relation' : f"You are currently a tagger. Please help me extract the relationship types of two events from the given text and what the two events are respectively.\
        Here are some examples:\
        Example 1 :\
        Content: ' she has stroked my back many times , and he has given me a bone of meat . i never bite those two\
        Result : [Relation 1] Causal Effect - X intent  [Event1]  i never bite those two  [Event2]  she has stroked my back many times [END]\
        Example 2 : \
        Content: ' certainly people who were only born yesterday know very little\
        Result : [Relation 1] Causal Effect - Effect on X  [Event1] certainly people who were only born yesterday [Event2] know very little [END]  \
        Example 3 :\
        Content: i grew too big for them , and they sent me away to the housekeeper 's room\
        Result : [Relation 1] Causal Effect - Effect on other  [Event1] i grew too big for them [Event2] they sent me away to the housekeeper 's room [END] \
        Example 4 :\
        Content: the whole fish tribe came back very tired from a hunting expedition , and looked about for a nice , cool spot in which to pitch their camp\
        Result : [Relation 1] Temporal - isBefore  [Event1] the whole fish tribe came back very tired from a hunting expedition [Event2] looked about for a nice , cool spot in which to pitch their camp [END] \
        Relation type definition : "
}

class Dataset:
    def __init__(self, path, event_or_relation) -> None:
        self.path = path
        self.tagging_type = event_or_relation
        if event_or_relation == 'Event':
            self.index = 5 #index
        elif event_or_relation == 'Relation':
            self.index = 6 #index
    
        self.dataset = self.get_data(path)
        self.input_, self.target, self.paragraph = [], [], []
        self.create_dataset()

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data.fillna('')
        return data.values
    
    def create_dataset(self):
        story_name = "-".join(self.dataset[0][4].split('-')[:-1])
        story_list = []
        for idx in tqdm(range(len(self.dataset))):
            tag_type = self.dataset[idx][self.index].split(' - ')[0]
            if self.tagging_type == 'Event' or self.path == 'data/test.csv':
                if tag_type in legal_tagging:
                    self.input_.append(self.create_input([idx]))
                    self.target.append(self.create_target([idx]))
                    self.paragraph.append(self.dataset[idx][4])
            elif self.tagging_type == 'Relation':
                current_story_name = "-".join(self.dataset[idx][4].split('-')[:-1])
                if current_story_name != story_name and story_list and tag_type in legal_tagging:
                    self.input_.append(self.create_input(story_list))
                    self.target.append(self.create_target(story_list))
                    self.paragraph.append(story_name)

                    story_name = current_story_name
                    story_list = []
                if tag_type in legal_tagging:
                    story_list.append(idx)
        return  
    
    def create_input(self, story_list):
        text = prompt[self.tagging_type]

        definition = Relation_definition if self.tagging_type == 'Relation' else Event_definition
        for title, defin in definition.items():
            text += f" {title} : {defin}, "

        text += "Now the context I give you is as follows. Please help me extract 1 Event for it. Context : "
        if self.tagging_type == 'Event' or self.tagging_type == 'Relation':
            text += text_segmentation(self.dataset[story_list[0]])
            # text += self.dataset[story_list[0]][1]
        elif self.tagging_type == 'Relation':
            text += self.dataset[story_list[0]][1]
        return text

    def create_target(self, story_list):
        text = ""
        for idx, story_idx in enumerate(story_list):
            text += f"[{self.tagging_type} {idx+1}] {self.dataset[story_idx][self.index]} "
            for i in range(7, len(self.dataset[story_idx])):
                if self.dataset[story_idx][i] != '':
                    left_parenthesis_index = self.dataset[story_idx][i].rfind('(')
                    if self.tagging_type == 'Relation' and i == 7:
                            text += " [Event1] " + "".join(self.dataset[story_idx][i][:left_parenthesis_index])
                    elif self.tagging_type == 'Relation' and i == 8:
                        text += " [Event2] " + "".join(self.dataset[story_idx][i][:left_parenthesis_index])
                    elif self.tagging_type == 'Event':
                        temp = " ".join(self.dataset[story_idx][i][:left_parenthesis_index].split(' - ')[1:])
                        text += f"[{self.dataset[story_idx][i][:left_parenthesis_index].split(' - ')[0]}] {temp} "
        text += " [END]"
        return text
    
    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.input_[idx], self.target[idx], self.paragraph[idx]

def main(event_or_relation = 'Event',
        Generation = 'tagging',
        model_name = "openai"):

    save_model_path = checkdir('save_model', event_or_relation, Generation, model_name)
    dataset = Dataset('data/test.csv', event_or_relation)
    
    prediction, target, context, paragraph = [], [], [], []
    for data in tqdm(dataset):
        text = data[0]
        try:
            if model_name == "openai":
                response = gptapi(text)
            elif model_name == "gemini":
                response = geminiapi(text)
            # print(response)
            # input()
            count = 0
            match_str = '[Event 1]' if event_or_relation == 'Event' else '[Relation 1]'
            match_len = 9 if event_or_relation == 'Event' else 12
            while '\n' in response or '[END]' not in response or response[:match_len] != match_str:
                if model_name == "openai":
                    response = gptapi(text)
                elif model_name == "gemini":
                    response = geminiapi(text)
                
                if count == 10:
                    break
                count += 1
            prediction.append(response)
        except:
            prediction.append("")

        target.append(data[1])
        context.append(text)
        paragraph.append(data[2])
    
    save_csv(prediction, target, context, paragraph, save_model_path + 'tagging.csv')
    return

def save_csv(prediction, target, context, paragraph, path):
    row = ['Paragraph', 'Content', 'Prediction', 'Reference']

    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(prediction)):
            writer.writerow([paragraph[i], context[i], prediction[i], target[i]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_or_relation', '-t', type=str, choices=['Event', 'Relation'], default='Event')
    parser.add_argument('--Generation', '-g', type=str, choices=['tagging', 'question'], default='tagging')
    parser.add_argument('--Model', '-m', type=str, choices=['openai', 'gemini'], default='openai')
    args = parser.parse_args()

    main(event_or_relation = args.event_or_relation, Generation = args.Generation, model_name = args.Model)