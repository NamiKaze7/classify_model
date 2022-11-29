import argparse
import pickle
import time
import os
from src.utils import options
import logging
import pandas as pd

from src.utils.train_utils import set_environment, get_time_dif
from src.utils.processor import CLASSIFYProcessor, convert_examples_to_features
from src.utils.processor import fine_grade_tokenize

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser("prepare task.")
options.add_prepare_args(parser)
args = parser.parse_args()
set_environment(args.seed)
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def text2token(raw_text, tokenizer):
    tokens = fine_grade_tokenize(raw_text, tokenizer)
    assert len(tokens) == len(raw_text)

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=args.max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    return encode_dict


def prepare_from_path(mode='train'):
    # prepare data to pickle file (get token ids, attention mask)
    path = os.path.join(args.data_dir, 'sp_{}.txt'.format(mode))
    logger.info('read {0} data from {1}'.format(mode, path))
    raw_examples = pd.read_csv(path, sep='\t').dropna()
    raw_examples = processor.get_examples(raw_examples)
    logger.info('load pretrained model from {}'.format(args.bert_dir))
    features = convert_examples_to_features(raw_examples, args.max_seq_len, args.bert_dir)
    save_path = os.path.join(args.save_dir, '{}.pkl'.format(mode))
    pickle.dump(features, open(save_path, 'wb'))
    logger.info('save {0} data in {1}'.format(mode, save_path))


if __name__ == '__main__':
    logger.info(args)
    start_time = time.time()
    logging.info('----------------开始计时----------------')
    logging.info('----------------------------------------')
    args.save_dir = os.path.join(args.save_dir, args.bert_type)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cache_list = ['train', 'dev']
    if args.test:
        cache_list.append('test')

    # prepare data
    processor = CLASSIFYProcessor(args.max_seq_len)
    for data_mode in cache_list:
        prepare_from_path(data_mode)

    time_dif = get_time_dif(start_time)
    logging.info("----------本次容器运行时长：{}-----------".format(time_dif))
