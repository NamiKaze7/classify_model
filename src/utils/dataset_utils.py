import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from src.utils.processor import CLASSIFYProcessor, convert_examples_to_features, convert_example
from src.utils.processor import fine_grade_tokenize
from transformers import BertTokenizer
import os
import pickle


class ClassifyDataset(Dataset):
    def __init__(self, train_path, opt, mode):
        self.train_data = []
        self.train_data.extend(pickle.load(open(train_path, 'rb')))
        self.num_tags = opt.num_tags

        self.mode = mode

    def y_onehot(self, y):
        l = np.zeros(self.num_tags)
        l[y] = 1
        return torch.tensor(list(l)).long()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        data = {'token_ids': torch.LongTensor(self.train_data[index].token_ids),
                'attention_masks': torch.FloatTensor(self.train_data[index].attention_masks),
                'token_type_ids': torch.LongTensor(self.train_data[index].token_type_ids),
                'labels': torch.tensor(self.train_data[index].label),
                'raw_text': self.train_data[index].raw_text}

        return data


def text2token(raw_text, tokenizer, opt):
    tokens = fine_grade_tokenize(raw_text, tokenizer)
    assert len(tokens) == len(raw_text)

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=opt.max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    return encode_dict


class ClassifyInferDataset(Dataset):
    def __init__(self, dev_path):
        self.dev_data = pickle.load(open(dev_path, 'rb'))

    def __len__(self):
        return len(self.dev_data)

    def __getitem__(self, index):
        data = {'token_ids': torch.LongTensor(self.dev_data[index].token_ids),
                'attention_masks': torch.FloatTensor(self.dev_data[index].attention_masks),
                'token_type_ids': torch.LongTensor(self.dev_data[index].token_type_ids),
                'labels': torch.tensor(self.dev_data[index].label),
                'raw_text': self.dev_data[index].raw_text}

        return data


class ClassifyTestDataset(Dataset):
    def __init__(self, data):
        self.test_data = data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        data = {'token_ids': torch.LongTensor(self.test_data[index].token_ids),
                'attention_masks': torch.FloatTensor(self.test_data[index].attention_masks),
                'token_type_ids': torch.LongTensor(self.test_data[index].token_type_ids),
                'raw_text': self.test_data[index].raw_text}

        return data
