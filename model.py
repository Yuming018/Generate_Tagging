import torch.nn as nn
from transformers import AutoModel, BartForConditionalGeneration

class BART(nn.Module):
    def __init__(self, model_name, freeze_bert=False) -> None:
        super(BART, self).__init__()

        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, decoder_attention_mask=None, lm_labels=None):
        outputs = self.model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            decoder_attention_mask = decoder_attention_mask,
                            labels = lm_labels)
        return outputs

if __name__ == '__main__':
    pass