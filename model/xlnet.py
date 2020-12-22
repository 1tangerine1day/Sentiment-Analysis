from transformers import XLNetModel
from transformers import XLNetTokenizer
import torch.nn as nn
import torch

class xlnet_classifier(nn.Module):
    def __init__(self, num_class, pretrain_model):
        super(xlnet_classifier, self).__init__()
        self.pretrain_model = pretrain_model
        self.num_class = num_class

        self.xlnet = XLNetModel.from_pretrained(pretrain_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.xlnet.config.hidden_size, num_class)
        
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrain_model)
        
    def forward(self, input_ids, attention_mask=None):
        
        last_hidden_states  = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled_output = self.pool_hidden_state(last_hidden_states)
        drop_output = self.drop(pooled_output)
        linear_output = self.out(drop_output)
        return linear_output

    def get_num_class(self):
        return self.num_class

    def get_pretrain_model_name(self):
        return self.pretrain_model

    def pool_hidden_state(self, last_hidden_state):
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state
    
    def get_tokenizer(self):
        return self.tokenizer