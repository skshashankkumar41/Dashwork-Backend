import torch
from transformers import BertModel, BertPreTrainedModel

class BertIntentModel(BertPreTrainedModel):
    def __init__(self, num_labels, conf):
        super(BertIntentModel, self).__init__(conf)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = torch.nn.Dropout(0.40)
        self.out = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.drop(pooled_output)
        return self.out(output)