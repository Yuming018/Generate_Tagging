import sys 
sys.path.append("..") 
from helper import checkdir
import torch

def generate_question(path_save_model, relation_tag, model, tagging_ids):
    path_save_model = checkdir(path_save_model, relation_tag)
    best_pth = path_save_model + 'question.pth'
    model.load_state_dict(torch.load(best_pth))
    output = model.generate(tagging_ids, num_beams=2, min_length=0)
    return output

if __name__ == '__main__':
    pass