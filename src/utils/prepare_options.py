import argparse

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):

        parser.add_argument('--output_dir', default='./out/',
                            help='the output dir for models checkpoints')
        # data dir
        
        parser.add_argument('--data_dir', default='./file',
                           help='source data dir to prepare')
        
        parser.add_argument('--save_dir', default='./data',
                           help='source data dir to prepare')
        # end prepare
        parser.add_argument('--bert_dir', default='../pretrained_models/chinese-roberta-wwm-ext',
                            help='bert dir for ernie / roberta-wwm / uer')

        parser.add_argument('--bert_type', default='roberta_wwm',
                            help='roberta_wwm / ernie_1 / uer_large')

        parser.add_argument('--max_seq_len', default=8,
                            help='max sequence length')

        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
    
    def get_parser_notebook(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args(args=[])