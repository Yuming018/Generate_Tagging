import torch
import time
import numpy as np
from rouge_score import rouge_scorer, scoring
import evaluate
from datasets import load_metric
from transformers import GenerationConfig
from transformers import Seq2SeqTrainer, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, TrainingArguments, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer, util

def seq2seq_training(model, tokenizer, train_data, valid_data, path_save_model, epochs, batch_size):    
    
    args = Seq2SeqTrainingArguments(
        output_dir= path_save_model + "checkpoints",
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=2e-5,
        # optim='adafactor',
        fp16=False,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100, 
        save_total_limit=1, 
        load_best_model_at_end = True,
        metric_for_best_model = 'eval_loss',
        predict_with_generate = True,
        weight_decay=0.01,
        include_inputs_for_metrics=True,
        lr_scheduler_type="polynomial",
        dataloader_prefetch_factor = None,
        report_to="none",
    )

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
        
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        # compute_metrics=compute_metrics,
        args=args,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    trainer.train()
    # model.save_pretrained(path_save_model)
    return

def ans_training(model, tokenizer, train_data, valid_data, path_save_model, epochs, batch_size):
    def Sen_T_metric(eval_pred):
        predictions, labels, _ = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        query_embedding = model.encode(decoded_preds)
        passage_embedding = model.encode(decoded_labels)
        result = util.dot_score(query_embedding, passage_embedding)
        rows, cols = result.shape
        anti_diagonal_sum = sum(result[i, cols - 1 - i] for i in range(rows))
        return {'Sentence_Transformer': anti_diagonal_sum/rows}
    
    def rouge_metrics(eval_pred):
        rouge = evaluate.load("rouge")
        predictions, labels, _ = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    args = TrainingArguments(
        output_dir= path_save_model + "checkpoints",
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=5e-6,
        # optim='adafactor',
        fp16=False,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100, 
        save_total_limit=1, 
        load_best_model_at_end = True,
        metric_for_best_model = 'eval_loss',
        # metric_for_best_model = 'eval_rougeL',
        # predict_with_generate = True,
        weight_decay=0.01,
        include_inputs_for_metrics=True,
        lr_scheduler_type="polynomial",
        dataloader_prefetch_factor = None,
        report_to="none",
    )

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
        
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        # compute_metrics=Sen_T_metric,
        args=args,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    trainer.train()
    return

def cls_training(model, tokenizer, train_data, valid_data, path_save_model, epochs, batch_size):
    def compute_metrics(eval_pred):
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")
        
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}
        
    args = TrainingArguments(
            output_dir= path_save_model + "checkpoints",
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=2e-5,
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500, 
            save_total_limit=1, 
            load_best_model_at_end = True,
            metric_for_best_model = 'eval_accuracy',
            weight_decay=0.01,
        )
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return

if __name__ == '__main__':
    pass