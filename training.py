import torch
import time
import numpy as np
from transformers import GenerationConfig

loss_fn = torch.nn.CrossEntropyLoss()

def train_model(model, train_dataloader, val_dataloader, device, tokenizer, epochs = 10, path_save_model= 'save_model/'):
    tagging_generation_config = GenerationConfig(
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=0,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id,
        penalty_alpha=0.6, 
        no_repeat_ngram_size=3,
        top_k=4, 
        temperature=0.5,
        max_new_tokens=128,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)
    min_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Elapsed':^9}")
        print("-"*60)

        model.train()
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss = 0, 0
        batch_counts = 0

        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            b_labels = b_labels.type(torch.LongTensor)
            b_labels = b_labels.to(device)
            
            # print(tokenizer.batch_decode(b_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
            # print(tokenizer.batch_decode(b_labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
            # model.zero_grad()
            output = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels = b_labels)
            
            # out = model.generate(input_ids = b_input_ids, generation_config=tagging_generation_config)
            # print(tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
            # input()
            loss = output['loss']
            # loss = loss_fn(output, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            time_elapsed = time.time() - t0_batch
            
            if (step % 200 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                val_loss = evaluate(model, val_dataloader, device)
                print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {val_loss:^10.6f} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
                if val_loss < min_val_loss:
                    torch.save(model.state_dict(), path_save_model)
                    min_val_loss = val_loss
                scheduler.step(loss)
                

        avg_train_loss = total_loss / len(train_dataloader)
        time_elapsed = time.time() - t0_epoch
        val_loss = evaluate(model, val_dataloader, device)
        print(f"{epoch + 1:^7} | {step:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {time_elapsed:^9.2f}")

def evaluate(model, val_dataloader, device):
    val_loss = []

    model.eval()
    for setp, batch in enumerate(val_dataloader):
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        b_labels = b_labels.type(torch.LongTensor)
        b_labels = b_labels.to(device)

        model.zero_grad()
        output = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels = b_labels)
        loss = output['loss']
        val_loss.append(loss.item())
        
    return np.mean(val_loss)

if __name__ == '__main__':
    train_model()