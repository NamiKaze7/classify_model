import argparse
import json
import time
import os
from pprint import pprint

from torch.utils.data import DataLoader
from transformers import BertModel, RobertaModel

from src.models.model import PredictModel
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


def main():
    start_time = time.time()
    logger.info('----------------开始计时----------------')
    logger.info('----------------------------------------')
    test_raw = pd.read_csv(args.test_path, sep='\t')
    processor = CLASSIFYTestProcessor(args.max_seq_len)
    test_examples = processor.get_examples(test_raw)
    bert_dir = ''
    bert_model = None
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained(args.bert_model)
        bert_dir = args.bert_model
    elif args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
        bert_dir = args.roberta_model
    test_features = processor.convert_examples_to_features(test_examples, args.max_seq_len, bert_dir)
    test_dataset = ClassifyTestDataset(test_features, args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size)

    network = ClassifyModel(bert_model=bert_model, num_tags=args.num_tags,
                            dropout_prob=args.dropout,
                            loss_type=args.loss_type,
                            max_seq_len=args.max_seq_len)
    model = PredictModel(args, network)
    load_prefix = os.path.join(args.load_dir, "checkpoint_best.pt")
    model.load(load_prefix)
    pred_lis = model.predict(test_loader)
    res_df = pd.DataFrame(pred_lis, columns=['卖点', '标签', '分数'])

    ret = pd.merge(test_raw, res_df).sort_values(by='分数', ascending=False)
    save_file = os.path.join(args.test_save_dir, 'result.xlsx')
    ret.to_excel(save_file)
    logging.info("----------本次容器运行时长：{}-----------".format(get_time_dif(start_time)))


if __name__ == "__main__":
    main()
