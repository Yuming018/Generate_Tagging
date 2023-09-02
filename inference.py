import csv
import torch
from tqdm import tqdm

def inference(model, tokenizer, test_dataloader, device, path, best_pth):
    input, target = [], []
    model.load_state_dict(torch.load(best_pth))
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader)):
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        output = model.generate(b_input_ids, num_beams=2, min_length=0)
        output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        target_ = tokenizer.batch_decode(b_labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        input.append(output)
        target.append(target_)
    save_csv(input, target, path)

def save_csv(input, target, path):
    row = ['ID', 'prediction', 'reference']
    
    path = path + 'predict.csv'
    with open(path, 'w', newline = '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row)
        for i in range(len(input)):
            writer.writerow([i, input[i], target[i]])

if __name__ == '__main__':
    inference()