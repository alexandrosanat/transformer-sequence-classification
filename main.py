import pandas as pd
import pytorch_lightning as pl
import torch
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from transformers import BertTokenizer, AutoModel
from math import floor
import numpy as np
from pytorch_lightning import Trainer
import torchmetrics


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
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text": text,
            "input_ids": (encoding["input_ids"]).flatten(),
            "attention_mask": (encoding["attention_mask"]).flatten(),
            "label": torch.tensor(target, dtype=torch.long),
        }


def clean_imdb_dataset(df):
    """Drops duplicate rows and returns just the text sequences and the labels."""
    df = df.drop_duplicates(subset="SentenceId", keep="first", ignore_index=True)
    return df["Phrase"].tolist(), df["Sentiment"].tolist()


class SentimentClassificationModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 32):
        super().__init__()

        self.bert = AutoModel.from_pretrained("bert-base-cased")
        # Freeze the Bert model
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = torch.nn.Dropout(0.1)
        self.Bidirectional = torch.nn.LSTM(
            input_size=768,
            hidden_size=1,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        hidden_size = 1
        self.classifier = torch.nn.Linear(hidden_size, 5)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        embeddings = outputs[0]  # (bs, seq_len, dim)
        X = self.dropout(embeddings)
        output, (hidden, cell) = self.Bidirectional(X)
        out = self.classifier(hidden[-1])
        # out = self.softmax(out)
        return out

    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        label = batch["label"]
        attention_mask = batch["attention_mask"]
        # Forward
        y_hat = self(input_ids, attention_mask, label)
        loss = self.loss_fct(y_hat, label)

        accuracy_epoch = self.accuracy(y_hat, label)
        self.log('train_acc_step', accuracy_epoch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        label = batch["label"]
        attention_mask = batch["attention_mask"]
        y_hat = self(input_ids, attention_mask, label)
        loss = self.loss_fct(y_hat, label)
        self.log(value=loss, name="val_loss", on_epoch=True)

    def configure_optimizers(self):
        # The first arg is required so that the bert layer won't be unfrozen
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=0.01, eps=1e-08
        )
        return optimizer


if __name__ == "__main__":

    N_EPOCHS = 15
    SENTENCE_LEN = 50

    # Import tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Import data
    data = pd.read_csv("train.tsv", sep="\t")
    texts, labels = clean_imdb_dataset(data)

    # Convert to Pytorch dataset
    dataset = ImdbDataset(texts, labels, tokenizer, SENTENCE_LEN)
    print(dataset.__getitem__(1))
    print(dataset.__len__())

    BATCH_SIZE = 32
    DATASET_LEN = dataset.__len__()
    NO_BATCHES = floor(DATASET_LEN / BATCH_SIZE)
    TRAIN_PROP = 0.9
    TRAIN_LEN = floor(TRAIN_PROP * NO_BATCHES * BATCH_SIZE)
    VAL_LEN = DATASET_LEN - TRAIN_LEN

    train_set, val_set = torch.utils.data.random_split(dataset, [TRAIN_LEN, VAL_LEN])
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8)

    trainer = Trainer(max_epochs=N_EPOCHS, fast_dev_run=True, log_every_n_steps=1)
    model = SentimentClassificationModel()

    trainer.fit(model, train_dataloader, val_dataloader)
