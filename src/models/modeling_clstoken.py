import os
from typing import Dict

import torch
import torch.nn as nn
# from transformers import BertModel
from src.utils.model_utils import LabelSmoothingCrossEntropy, FocalLoss
from src.utils.metrics import MetricsF1
from src.utils.allennlp import replace_masked_values


class BaseModel(nn.Module):
    def __init__(self,
                 bert_model,
                 dropout_prob):
        super(BaseModel, self).__init__()

        self.bert_module = bert_model

        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


class ClassifyModel_CLS(BaseModel):
    def __init__(self,
                 bert_model,
                 num_tags,
                 args,
                 dropout_prob=0.1,
                 loss_type='ls_ce',
                 max_seq_len=100,
                 **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(ClassifyModel_CLS, self).__init__(bert_model, dropout_prob=dropout_prob)

        self._metrics = MetricsF1(num_tags)
        out_dims = self.bert_config.hidden_size

        self.num_tags = num_tags
        self.perm = torch.arange(0, num_tags).unsqueeze(0)
        if args.cuda:
            self.perm = self.perm.cuda()

        self.tags_fc = nn.Linear(out_dims, num_tags)

        weight = [.5]*(num_tags-1) + [.5*num_tags]
        self.weight = torch.tensor(weight)
        if args.cuda:
            self.weight = torch.tensor(weight).cuda()
        reduction = 'none'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            self.criterion = FocalLoss(reduction=reduction, weight=self.weight)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)

        init_blocks = [self.tags_fc]

        self._init_weights(init_blocks)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                labels,
                raw_text,
                mode='train'):
        output_dict = {}
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
        seq_out = bert_outputs[0]  # last_hidden_state
        seq_out = replace_masked_values(seq_out, attention_masks.unsqueeze(-1), 0)
        logits = self.tags_fc(seq_out[:, 0, :])

        if labels is not None:
            loss = self.criterion(logits, labels).mean()
        else:
            loss = -1
        if self.num_tags != 1:
            output_dict['logits'] = logits
            output_dict['loss'] = loss
            pred = torch.max(logits, 1).indices.unsqueeze(1)
            output_dict['pred'] = pred
        else:
            output_dict['logits'] = logits
            output_dict['loss'] = loss
            pred = logits
            output_dict['pred'] = logits
        self._metrics(labels, pred, raw_text)
        return output_dict

    def predict(self,
                token_ids,
                attention_masks,
                token_type_ids,
                raw_text):
        output_dict = {}
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
        seq_out = bert_outputs[0]  # last_hidden_state

        logits = self.tags_fc(seq_out[:, 0, :])
        if self.num_tags != 1:
            prob = torch.softmax(logits, -1)
            top = torch.topk(prob, 1)
            output_dict['raw_text'] = raw_text
            output_dict['pred_label'] = top.indices
            output_dict['pred_score'] = (prob * self.perm).sum(-1).unsqueeze(1)
        else:
            output_dict['raw_text'] = raw_text
            output_dict['pred_label'] = logits
            output_dict['pred_score'] = logits
        return output_dict

    def get_metrics(self, logger=None):
        metrics = self._metrics.get_overall_metric()
        precision = metrics['p']
        recall = metrics['r']
        f1 = metrics['f1']
        acc = metrics['acc']
        df = metrics['dataframe']
        if logger is not None:
            logger.info(f"precision:{precision} ")
            logger.info(f"recall:{recall} ")
            logger.info(f"f1:{f1} ")
            logger.info(f"acc:{acc} ")

        return metrics

    def reset(self):
        self._metrics.reset()

    def get_raw_details(self):
        return self._metrics.get_raw()
