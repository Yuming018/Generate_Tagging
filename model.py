import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, LoraConfig

class Mt0(nn.Module):
    def __init__(self, model_name, freeze_bert=False) -> None:
        super(Mt0, self).__init__()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
            tokenizer_name_or_path=self.model,
        )
        self.peft_model = get_peft_model(self.model, self.peft_config)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.peft_model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
        print(outputs)
        return outputs

def create_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32, 
        lora_dropout=0.1,
        bias = "lora_only",
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model


if __name__ == '__main__':
    pass