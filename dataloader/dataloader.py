
from torch.utils.data import Dataset
import json
import numpy as np
import pandas as pd
import torch

class MyDataset(Dataset):
    def __init__(self, config, file):
        self.config = config
        with open(self.config.schema, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)[0]
        with open(self.config.vocab, "r", encoding="utf-8") as ff:
            self.vocab = json.load(ff)

        self.df = pd.read_csv(file)
        self.sentence = self.df.sentence.tolist()
        self.label = self.df.label.tolist()


    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        token = list(sentence)
        token_len = len(token)
        if token_len <= 4:
            token.extend(["pad"] *(5-token_len))
            token_len = 5

        label = self.label[idx]

        tokeni2d = [self.vocab.get(i, self.vocab['unk']) for i in token]
        input_ids = np.array(tokeni2d)
        mask = [1] * token_len
        attention_mask = np.array(mask)

        label_init = [0.] * self.config.num_label
        if type(label) is float:
            label2id = label_init
        else:
            label_list = label.split("、")
            for i in label_list:
                label_init[self.label2id[i]] = 1
            label2id = label_init

        return sentence, label, label2id, attention_mask, input_ids, token_len

def collate_fn(batch):
    sentence, label, label2id, attention_mask, input_ids, token_len = zip(*batch)
    cur_batch = len(batch)
    max_len = max(token_len)

    batch_input_ids = torch.LongTensor(cur_batch, max_len).zero_()
    batch_attention_mask = torch.LongTensor(cur_batch, max_len).zero_()

    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_attention_mask[i, :token_len[i]].copy_(torch.from_numpy(attention_mask[i]))
    return {"input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "sentence": sentence,
            "label": label,
            "target": torch.tensor(label2id)}