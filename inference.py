
def inference(model, test_dataloader, device):
    model.eval()
    for step, batch in enumerate(test_dataloader):
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

if __name__ == '__main__':
    inference()