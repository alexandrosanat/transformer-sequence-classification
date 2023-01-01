import pandas as pd
import pytorch_lightning as pl
import torch
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from transformers import BertTokenizer


class ImdbDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'label': torch.tensor(target, dtype=torch.long),
            'input_ids': (encoding['input_ids']).flatten(),
            'attention_mask': (encoding['attention_mask']).flatten(),
            'token_type_ids': (encoding['token_type_ids']).flatten()
        }


def clean_imdb_dataset(df):

    df = df.drop_duplicates(subset="SentenceId", keep="first", ignore_index=True)

    return df


if __name__ == "__main__":

    # Import tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Import data
    data = pd.read_csv("train.tsv", sep="\t")
    data = clean_imdb_dataset(data)
    print(len(data))
    texts = data["Phrase"].tolist()
    labels = data["Sentiment"].tolist()

    # Convert to Pytorch dataset
    dataset = ImdbDataset(texts, labels, tokenizer, 50)
    print(dataset.__getitem__(1))

