import os
import re
import json
import logging
import numpy as np
from transformers import BertTokenizer
from collections import defaultdict
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

ENTITY_TYPES = ['DRUG', 'DRUG_INGREDIENT', 'DISEASE', 'SYMPTOM', 'SYNDROME', 'DISEASE_GROUP',
                'FOOD', 'FOOD_GROUP', 'PERSON_GROUP', 'DRUG_GROUP', 'DRUG_DOSAGE', 'DRUG_TASTE',
                'DRUG_EFFICACY']
PAD_token = 0


def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

class InputExample:
    def __init__(self,
                 text,
                 label=None):
        self.text = text
        self.label = label


class InputTestExample:
    def __init__(self,
                 text):
        self.text = text


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 raw_text):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.raw_text = raw_text


class ClassifyFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 raw_text,
                 label
                 ):
        super(ClassifyFeature, self).__init__(token_ids=token_ids,
                                              attention_masks=attention_masks,
                                              token_type_ids=token_type_ids,
                                              raw_text=raw_text)
        self.label = label


class CLASSIFYProcessor:
    def __init__(self, max_x_length=100):
        self.max_x_length = max_x_length

    def get_examples(self, raw_examples):
        examples = []
        for i,d in tqdm(raw_examples.iterrows()):

            sent = d['卖点'][:self.max_x_length - 2]
            label = d['label']
            examples.append(InputExample(text=sent, label=label))

        return examples


class CLASSIFYTestProcessor:
    def __init__(self, bert_dir, max_seq_len=100):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    def get_examples(self, raw_examples):
        examples = []
        for d in raw_examples.iterrows():
            d = d[1]
            sent = d['卖点'][:self.max_seq_len - 2]
            examples.append(InputTestExample(text=sent))

        return examples

    def convert_examples_to_features(self, examples):

        callback_info = []

        logger.info(f'Convert {len(examples)} examples to features')

        for i, example in enumerate(examples):

            tmp_callback = self.convert_example(
                example=example
            )

            if tmp_callback is None:
                continue

            callback_info.append(tmp_callback)

        return callback_info

    def convert_example(self, example: InputTestExample):
        raw_text = example.text

        tokens = fine_grade_tokenize(raw_text, self.tokenizer)
        assert len(tokens) == len(raw_text)

        encode_dict = self.tokenizer.encode_plus(text=tokens,
                                                 padding='max_length',
                                                 max_length=512,
                                                 pad_to_max_length=True,
                                                 is_pretokenized=True,
                                                 return_token_type_ids=True,
                                                 return_attention_mask=True)

        token_ids = np.array(encode_dict['input_ids'])
        attention_masks = np.array(encode_dict['attention_mask'])
        token_type_ids = np.array(encode_dict['token_type_ids'])

        callback_info = BaseFeature(token_ids=token_ids,
                                    attention_masks=attention_masks,
                                    token_type_ids=token_type_ids,
                                    raw_text=raw_text
                                    )

        return callback_info


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens


def convert_example(example: InputExample, tokenizer: BertTokenizer, max_seq_len):
    raw_text = example.text

    tokens = fine_grade_tokenize(raw_text, tokenizer)
    assert len(tokens) == len(raw_text)

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        padding='max_length',
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    callback_info = ClassifyFeature(token_ids=token_ids,
                                    attention_masks=attention_masks,
                                    token_type_ids=token_type_ids,
                                    raw_text=raw_text, label=example.label
                                    )

    return callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):

        tmp_callback = convert_example(
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer
        )

        if tmp_callback is None:
            continue

        callback_info.append(tmp_callback)

    return callback_info
