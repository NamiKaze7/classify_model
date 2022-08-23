import pickle
import time
import os
import json
import logging
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.utils.trainer import train
from src.utils.prepare_options import Args
from src.utils.model_utils import build_model
from src.utils.dataset_utils import ClassifyDataset
from src.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel, get_time_dif
from src.utils.processor import CLASSIFYProcessor, convert_examples_to_features
from src.utils.processor import fine_grade_tokenize
import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


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


if __name__ == '__main__':
    start_time = time.time()
    logging.info('----------------开始计时----------------')
    logging.info('----------------------------------------')

    opt = Args().get_parser()

    opt.output_dir = os.path.join(opt.output_dir, opt.bert_type)

    set_seed(opt.seed)

    train_path = os.path.join(opt.data_dir, 'sp_train.txt')
    dev_path = os.path.join(opt.data_dir, 'sp_test.txt')

    # prepare_train
    logger.info('read train data from {}'.format(train_path))
    train_raw_examples = pd.read_csv(train_path, sep='\t').dropna()
    processor = CLASSIFYProcessor(opt.max_seq_len)
    train_examples = processor.get_examples(train_raw_examples)
    train_features = convert_examples_to_features(train_examples, opt.max_seq_len, opt.bert_dir)
    save_path = os.path.join(opt.save_dir, 'train.pkl')
    pickle.dump(train_features, open(save_path, 'wb'))
    logger.info('save train data in {}'.format(save_path))
    # prepare_test
    logger.info('read dev data from {}'.format(dev_path))
    dev = pd.read_csv(dev_path, sep='\t')
    processor = CLASSIFYProcessor(opt.max_seq_len)
    dev_examples = processor.get_examples(dev)
    dev_features = convert_examples_to_features(dev_examples, opt.max_seq_len, opt.bert_dir)
    save_path =os.path.join(opt.save_dir, 'dev.pkl')
    pickle.dump(dev_features, open(save_path, 'wb'))
    logger.info('save dev data in {}'.format(save_path))

    time_dif = get_time_dif(start_time)
    logging.info("----------本次容器运行时长：{}-----------".format(time_dif))
