import argparse
import json
import re
import time
import os
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel, RobertaModel
import xlsxwriter

from src.models.model import TagtreePredictModel
from src.models.modeling_cls import ClassifyModel
from src.utils import options
from src.utils.dataset_utils import ClassifyTestDataset
from src.utils.train_utils import create_logger, set_environment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import logging
import pandas as pd

from src.utils.functions_utils import get_time_dif
from src.utils.processor import CLASSIFYTestProcessor

parser = argparse.ArgumentParser("training task.")
options.add_raw_args(parser)
options.add_train_args(parser)
options.add_data_args(parser)
options.add_bert_args(parser)
options.add_test_args(parser)
args = parser.parse_args()

if not os.path.exists(args.test_save_dir):
    os.mkdir(args.test_save_dir)
logger = create_logger("Model Testing", log_file=os.path.join(args.test_save_dir, args.test_log_file))

args.cuda = args.gpu_num > 0
args_path = os.path.join(args.test_save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump((vars(args)), f)
pprint(args)
set_environment(args.seed, args.cuda)


def load_model(args):
    bert_dir = ''
    bert_model = None
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained(args.bert_model)
        bert_dir = args.bert_model
    elif args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
        bert_dir = args.roberta_model

    network = ClassifyModel(bert_model=bert_model, num_tags=args.num_tags,
                            dropout_prob=args.dropout,
                            loss_type=args.loss_type,
                            max_seq_len=args.max_seq_len)
    model = TagtreePredictModel(args, network)
    load_prefix = os.path.join(args.load_dir, "checkpoint_best.pt")
    model.load(load_prefix)

    return model, bert_dir


def get_onesp(processor, model, bert_dir, test_raw):
    test_examples = processor.get_examples(test_raw)
    test_features = processor.convert_examples_to_features(test_examples, args.max_seq_len, bert_dir)
    test_dataset = ClassifyTestDataset(test_features, args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size)
    pred_lis = model.predict(test_loader)
    res_df = pd.DataFrame(pred_lis, columns=['卖点', '标签', '分数'])

    ret = pd.merge(test_raw, res_df).sort_values(by='分数', ascending=False)

    return ret


def just_chinese(strings):
    regStr = ".*?([\u4E00-\u9FA5]+).*?"
    expr = ''.join(re.findall(regStr, strings))
    if expr:
        return expr
    return '\n'


def hand_raw_text(df, g_id, g_name):
    l = []
    for i, row in df.iterrows():
        v = row['review_body']
        blis = re.split('，|,| +|。|！|、', v)
        for w in blis:
            w = just_chinese(w)
            if 6 >= len(w) >= 3:
                l.append([row[g_id], row[g_name], w])
    return pd.DataFrame(l, columns=[g_id, g_name, '卖点'])


def main():
    start_time = time.time()
    logger.info('----------------开始计时----------------')
    logger.info('----------------------------------------')
    if args.group_name == 'category5_id':
        g_id = 'category5_id'
        g_name = 'category5_name'
    else:
        g_id = 'base_sku_id'
        g_name = 'base_sku_name'
    raw_df = pd.read_csv(args.test_path, sep='\t')[[g_id, g_name,
                                                    'review_body']][:100]
    logger.info('total raw data size: {}\n'.format(len(raw_df)))
    df = hand_raw_text(raw_df, g_id, g_name)
    logger.info('total data size: {}\n'.format(len(df)))
    model, bert_dir = load_model(args)
    processor = CLASSIFYTestProcessor(args.max_seq_len)

    cate_ids = list(set(df[g_id].values))
    reslis = []
    for i in tqdm(range(len(cate_ids))):
        cate_id = cate_ids[i]
        raw = df[df[g_id] == cate_id]
        raw = raw.drop_duplicates(subset=['卖点'])
        ret = get_onesp(processor, model, bert_dir, raw)
        name = raw.iloc[0][g_name]
        good = ret[ret['分数'] > args.limit_score]
        if g_id == 'base_sku_id':
            goodsp = good.head()['卖点'].to_list()
        else:
            goodsp = good.head(20)['卖点'].to_list()
        reslis.append([cate_id, name, goodsp])
    if not os.path.exists(args.test_save_dir):
        os.mkdir(args.test_save_dir)
    save_file = os.path.join(args.test_save_dir, 'result_{}.xlsx'.format(args.group_name))
    total_ret = pd.DataFrame(reslis, columns=[g_id, g_name, 'selling_points'])
    total_ret.to_excel(save_file, engine='xlsxwriter')
    logging.info("----------本次容器运行时长：{}-----------".format(get_time_dif(start_time)))


if __name__ == "__main__":
    main()
