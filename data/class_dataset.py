from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class dataset(Dataset):
    def __init__(self, path, tokenizer, cls_token):
        self.tokenizer = tokenizer
        self.cls = cls_token
        self.name = path
        self.read_csv(path)
     

    def __getitem__(self, idx):
        label, text = self.df.iloc[idx, :3].values
        
        label_tensor = torch.tensor(label)

        word_pieces = [self.cls]
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > 500:
            tokens = tokens[:200] + tokens[-300:]
        word_pieces += tokens
        
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
            
        return tokens_tensor, label_tensor
    
    def __len__(self):
        return len(self.df)

    def read_csv(self, path):
        self.df = pd.read_csv(path)

        # five to three classes
        self.df['truth'] = np.where(self.df['truth'].values == 0, 0, self.df['truth'])
        self.df['truth'] = np.where(self.df['truth'].values == 1, 0, self.df['truth'])
        self.df['truth'] = np.where(self.df['truth'].values == 2, 1, self.df['truth'])
        self.df['truth'] = np.where(self.df['truth'].values == 3, 2, self.df['truth'])
        self.df['truth'] = np.where(self.df['truth'].values == 4, 2, self.df['truth'])

        print("Size :",len(self.df))
        # self.draw_df_label()

    def get_df(self):
        return self.df

    def get_len(self):
        return len(self.df)

    