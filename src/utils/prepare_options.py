import argparse

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):

        parser.add_argument('--output_dir', default='./out/',
                            help='the output dir for model checkpoints')
        # data dir
        
        parser.add_argument('--data_dir', default='./data/20210704/dev_snapshoot_0727.txt',
                           help='source data dir to prepare')
        
        parser.add_argument('--save_dir', default='./data/20210704/dev_snapshoot_0727.pkl',
                           help='source data dir to prepare')
        # end prepare
        parser.add_argument('--bert_dir', default='../chinese_roberta_wwm_ext_pytorch',
                            help='bert dir for ernie / roberta-wwm / uer')

        parser.add_argument('--bert_type', default='roberta_wwm',
                            help='roberta_wwm / ernie_1 / uer_large')

        parser.add_argument('--use_type_embed', default=False, action='store_true',
                            help='weather to use soft label in span loss')

        parser.add_argument('--use_fp16', default=False, action='store_true',
                            help='weather to use fp16 during training')

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