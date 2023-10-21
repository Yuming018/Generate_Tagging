from helper import checkdir
import torch

def generate_relation_tagging(path_save_model, model, input_ids, relation_tag):
    path_save_model = checkdir(path_save_model, relation_tag)
    best_pth = path_save_model + 'tagging.pth'
    model.load_state_dict(torch.load(best_pth))
    output = model.generate(input_ids, num_beams=2, min_length=0)
    return output

def generate_event_tagging(path_save_model, model, input_ids, relation_tag):
    path_save_model = checkdir(path_save_model, relation_tag)
    best_pth = path_save_model + 'tagging.pth'
    model.load_state_dict(torch.load(best_pth))
    output = model.generate(input_ids, num_beams=2, min_length=0)
    return output

if __name__ == '__main__':
    pass