import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        # data dir

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
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()

    def get_parser_notebook(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args(args=[])
