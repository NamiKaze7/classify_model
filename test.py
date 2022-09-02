import argparse
import json

from src.models.modeling_clstoken import ClassifyModel_CLS
from src.utils.filter_tag import taglist
import time
import os
from pprint import pprint
from tqdm import tqdm
from src.utils.data_gen import PredictBatchGen
from torch.utils.data import DataLoader
from transformers import BertModel, RobertaModel
import xlsxwriter
from src.utils.data_tools import just_chinese, cut_sentence
from src.models.model import PredictModel
from src.models.modeling_cls import ClassifyModel
from src.utils import options
from src.utils.dataset_utils import ClassifyTestDataset
from src.utils.train_utils import create_logger, set_environment
import torch

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

cpu_num = args.num_workers
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

if not os.path.exists(args.test_save_dir):
    os.makedirs(args.test_save_dir)
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

    if args.cls_model == 'CLS':
        network = ClassifyModel_CLS(bert_model=bert_model, num_tags=args.num_tags,
                                    dropout_prob=args.dropout,
                                    loss_type=args.loss_type,
                                    max_seq_len=args.max_seq_len, args=args)
    else:
        network = ClassifyModel(bert_model=bert_model, num_tags=args.num_tags,
                                dropout_prob=args.dropout,
                                loss_type=args.loss_type,
                                max_seq_len=args.max_seq_len, args=args)
    model = PredictModel(args, network)
    load_prefix = os.path.join(args.load_dir, "checkpoint_best.pt")
    model.load(load_prefix)

    return model, bert_dir


def main():
    if os.path.exists('run.sh'):
        with open('run.sh', 'r') as f:
            logger.info("Run Code:\r\n {}".format(f.read()))
    start_time = time.time()
    logger.info('----------------开始计时----------------')
    logger.info('----------------------------------------')
    raw_df = pd.read_csv(args.test_path, sep='\t')
    logger.info('total raw data size: {}\n'.format(len(raw_df)))
    model, bert_dir = load_model(args)
    processor = CLASSIFYTestProcessor(bert_dir, args.max_seq_len)
    logger.info('model_has_build')
    test_examples = processor.get_examples(raw_df)
    test_features = processor.convert_examples_to_features(test_examples)
    test_dataset = ClassifyTestDataset(test_features, args)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)
    pred_lis = model.predict(test_loader)
    res_df = pd.DataFrame(pred_lis, columns=['text', 'label', 'score'])[['label', 'score']]
    res = pd.concat([raw_df, res_df], axis=1)
    save_file = os.path.join(args.test_save_dir, 'result.txt')
    res.to_csv(save_file, sep='\t')


if __name__ == "__main__":
    main()
