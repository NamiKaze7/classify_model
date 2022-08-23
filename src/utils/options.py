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
        parser.add_argument('--dev_raw_path', type=str, default='',
                            help='dev raw data path')
        parser.add_argument('--data_aug_path', type=str, default='',
                            help='data aug path')
        parser.add_argument('--train_path', type=str, default='',
                            help='train data path')
        parser.add_argument('--dev_path', type=str, default='',
                            help='dev data path')
        # prepare
        parser.add_argument('--prepare_mode', default='train')
        
        parser.add_argument('--src_dir', default='./data/20210704/dev_snapshoot_0727.txt',
                           help='source data dir to prepare')
        
        parser.add_argument('--save_dir', default='./data/20210704/dev_snapshoot_0727.pkl',
                           help='source data dir to prepare')
        # end prepare
        parser.add_argument('--bert_dir', default='../chinese_roberta_wwm_ext_pytorch',
                            help='bert dir for ernie / roberta-wwm / uer')

        parser.add_argument('--bert_type', default='roberta_wwm',
                            help='roberta_wwm / ernie_1 / uer_large')

        parser.add_argument('--model_step', default='100000')

        parser.add_argument('--loss_type', default='ls_ce',
                            help='loss type for span bert')

        parser.add_argument('--use_type_embed', default=False, action='store_true',
                            help='weather to use soft label in span loss')

        parser.add_argument('--use_fp16', default=False, action='store_true',
                            help='weather to use fp16 during training')


        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default='3',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
        
        parser.add_argument('--mode', type=str, default='train',
                            help='train / stack')
        
        parser.add_argument('--train_batch_size', default=108, type=int)
        parser.add_argument('--dev_batch_size', default=3000, type=int)
        parser.add_argument('--max_seq_len', default=68, type=int)
        parser.add_argument('--log_loss_steps', default=20, type=int)
    
        parser.add_argument('--swa_start', default=3, type=int,
                            help='the epoch when swa start')

        # train args
        parser.add_argument('--train_epochs', default=12, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.02, type=float,
                            help='drop out probability')

        parser.add_argument('--lr', default=2e-5, type=float,
                            help='learning rate for the bert module')

        parser.add_argument('--other_lr', default=2e-3, type=float,
                            help='learning rate for the module except bert')

        parser.add_argument('--max_grad_norm', default=1.0, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.00, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        
        
        
        parser.add_argument('--eval_model', default=True, action='store_true',
                            help='whether to eval model after training')

        parser.add_argument('--attack_train', default='', type=str,
                            help='fgm / pgd attack train when training')


        # test args
        parser.add_argument('--ckpt_dir', default='', type=str)
        parser.add_argument('--eval_save_name', default='', type=str,
                           help='save name of eval output')
        parser.add_argument('--checkpoint', default='100000', type=str,
                           help='')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
    
    def get_parser_notebook(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args(args=[])