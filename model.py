import torch.nn as nn
from transformers import AutoModel, BartForConditionalGeneration

class BART(nn.Module):
    def __init__(self, model_name, freeze_bert=False) -> None:
        super(BART, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def generate(self):
        super(BART, self).geneate()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

if __name__ == '__main__':
    pass