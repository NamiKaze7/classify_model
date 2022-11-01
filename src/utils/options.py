import argparse
import torch


def add_device_args(parser):
    parser.add_argument("--gpu_num", default=torch.cuda.device_count(), type=int, help="training gpu num.")
    parser.add_argument("--num_workers", default=4, type=int, help="num workers for dataloader.")


def add_data_args(parser: argparse.ArgumentParser):
    # data & model directory
    parser.add_argument('--data_dir', type=str, default='data/train_long_com',
                        help='data dir')
    parser.add_argument("--save_dir", default="./model_saved/checkpoint-lc-1", type=str, help="save dir.")
    parser.add_argument("--load_dir", default="./model_saved/checkpoint-lc-1", type=str, help="load dir.")
    parser.add_argument("--log_file", default="train.log", type=str, help="train log file.")
    parser.add_argument("--get_result", default="", type=str, help="get xlsx file.")


def add_train_args(parser: argparse.ArgumentParser):
    # data batch
    parser.add_argument("--max_epoch", default=5, type=int, help="max epoch.")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size.")
    parser.add_argument('--eval_batch_size', type=int, default=8, help="eval batch size.")
    parser.add_argument("--test", default=False, type=bool, help="do test or not.")
    parser.add_argument("--log_per_updates", default=20, type=int, help="log pre update size.")
    # optimizer
    parser.add_argument("--optimizer", default="adam", type=str, help="train optimizer.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="learning rate.")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="weight decay.")
    parser.add_argument("--grad_clipping", default=1.0, type=float, help="gradient clip.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--warmup', type=float, default=0.06,
                        help="Proportion of training to perform linear learning rate warm up for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_schedule", default="warmup_linear", type=str, help="warmup schedule.")
    parser.add_argument("--pretrained", default=False, type=bool, help="pretrained or not")
    # loss
    # parser.add_argument("--eps", default=1e-8, type=float, help="ema gamma.")
    parser.add_argument('--loss_type', default='ls_ce',
                        help='loss type')


def add_augment_args(parser: argparse.ArgumentParser):
    # AT-train
    parser.add_argument("--attack", default="", type=str, help="attack methods.")
    # R-Drop
    parser.add_argument("--r_drop", default=False, type=bool, help="r_drop methods.")
    parser.add_argument("--kl_weight", default=1.0, type=float, help="r_drop_weight methods.")
    # manifold mix_up
    parser.add_argument("--alpha", default=1, type=int, help="manifold_mix_up p")
    parser.add_argument("--mix_up", default=False, type=bool, help="manifold_mix_up")


def add_model_args(parser: argparse.ArgumentParser):
    # bert model
    parser.add_argument("--encoder", type=str, default='roberta')
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument("--roberta_model", type=str, help="robert model path.",
                        default="../pretrained_models/chinese-roberta-wwm-ext")
    parser.add_argument("--bert_model", type=str, help="robert model path.",
                        default="../pretrained_models/bert-base-chinese")
    parser.add_argument("--bert_learning_rate", default=1.5e-5, type=float, help="bert learning rate.")
    parser.add_argument("--bert_weight_decay", default=0.01, type=float, help="bert weight decay.")
    # full-connection network
    parser.add_argument("--cls_model", default='mid', type=str, help="cls network.")
    parser.add_argument("--num_tags", default=5, type=int, help="cls label count")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for data shuffling, embedding init, etc.')


def add_test_args(parser):
    parser.add_argument('--test_path', default='./file/raw_data1/sp_test.txt', type=str)
    parser.add_argument('--test_save_dir', default='./model_result/test1', type=str)
    parser.add_argument('--test_log_file', default='test.log', type=str)
    parser.add_argument('--eval_batch_size', type=int, default=8, help="eval batch size.")
    parser.add_argument('--limit_score', default=0.8, type=float)
    parser.add_argument('--group_name', default='base_sku_id', type=str)
    parser.add_argument('--top_sp', default=20, type=int)
    parser.add_argument("--load_dir", default="./model_saved/checkpoint-lc-1", type=str, help="load dir.")
    parser.add_argument("--log_file", default="train.log", type=str, help="train log file.")


# prepare data args
def add_prepare_args(parser):
    parser.add_argument('--data_dir', default='./file/long_com_train',
                        help='source data dir to prepare')
    parser.add_argument('--save_dir', default='./data/train_long_com',
                        help='source data dir to prepare')
    # end prepare
    parser.add_argument('--bert_dir', default='../pretrained_models/chinese-roberta-wwm-ext',
                        help='bert dir for ernie / roberta-wwm / uer')

    parser.add_argument('--bert_type', default='roberta',
                        help='roberta_wwm / ernie_1 / uer_large')

    parser.add_argument('--max_seq_len', default=128, type=int,
                        help='max sequence length')
    # other args
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--test', type=bool, default=False,
                        help='prepare test or not')


# training args
def add_args(parser, mode='train'):
    if mode == 'train':
        func_list = [add_train_args, add_data_args, add_device_args, add_model_args, add_augment_args]
    else:
        func_list = [add_test_args, add_device_args, add_model_args]
    for func in func_list:
        func(parser)
