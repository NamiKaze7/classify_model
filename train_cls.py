import os
import json
import argparse
from datetime import datetime

from torch.utils.data import DataLoader, RandomSampler

from src.models.model import FineTuningModel
from src.utils import options
from pprint import pprint
import torch

from src.utils.dataset_utils import ClassifyDataset, ClassifyInferDataset
from src.utils.train_utils import create_logger, set_environment

from transformers import RobertaModel, BertModel
from src.models.modeling_cls import ClassifyModel
import pandas as pd


from tqdm import tqdm

parser = argparse.ArgumentParser("training task.")
options.add_raw_args(parser)
options.add_train_args(parser)
options.add_data_args(parser)
options.add_bert_args(parser)
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

cpu_num = args.num_workers
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

args.cuda = args.gpu_num > 0
args_path = os.path.join(args.save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump((vars(args)), f)

args.batch_size = args.batch_size // args.gradient_accumulation_steps

logger = create_logger("Model Training", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.seed, args.cuda)


def main():
    best_result = float("-inf")
    logger.info("num_workers:{0}.... gpu_nums:{1}".format(args.num_workers, args.gpu_num))
    logger.info("Loading data...")
    train_dataset = ClassifyDataset(args.train_path, args, 'train')
    dev_dataset = ClassifyInferDataset(args.dev_path, args)
    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.num_workers)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size)
    logger.info('----------------Build model...----------------')

    num_train_steps = int(args.max_epoch * len(train_loader) / args.gradient_accumulation_steps)
    logger.info("Model update steps {}!".format(num_train_steps))

    logger.info(f"Build Cls {args.encoder} model.")
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained(args.bert_model)
    elif args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)

    network = ClassifyModel(bert_model=bert_model, num_tags=args.num_tags,
                            dropout_prob=args.dropout,
                            loss_type=args.loss_type,
                            max_seq_len=args.max_seq_len)
    model = FineTuningModel(args, network, num_train_steps=num_train_steps)
    epoch_pre = 0
    if args.pretrained:
        load_prefix = os.path.join(args.load_dir, "checkpoint_best")
        model.load(load_prefix)
        other_path = load_prefix + ".ot"
        epoch_pre = torch.load(other_path)['epoch']
    train_start = datetime.now()

    for epoch in range(1 + epoch_pre, args.max_epoch + epoch_pre + 1):
        model.reset()

        logger.info('At epoch {}'.format(epoch))
        for step, batch in enumerate(train_loader):

            model.update(batch)
            if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                logger.info(
                    "Updates[{0:6}] train loss[{1:.5f}] remaining[{2}].\r\n ".format(
                        model.updates, model.train_loss.avg,
                        str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0])
                )

                model.avg_reset()

        df = model.get_metrics(logger)['dataframe']
        if args.get_result:
            detail_df = model.get_raw_details()
            output_metric_path = os.path.join(args.save_dir, 'Experiments_train.xlsx')
            with pd.ExcelWriter(output_metric_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='metrics')
                detail_df.to_excel(writer, sheet_name='details')
        model.reset()
        model.avg_reset()
        logger.info("====== Begin to Evaluate on Dev set...")
        model.evaluate(dev_loader)
        logger.info("Evaluate epoch:[{0:6}] eval loss[{1:.5f}]\r\n".format(epoch, model.dev_loss.avg))

        metrics = model.get_metrics(logger)
        if args.get_result:
            output_metric_path = os.path.join(args.save_dir, 'Experiments_dev.xlsx')
            detail_df = model.get_raw_details()
            with pd.ExcelWriter(output_metric_path, engine='xlsxwriter') as writer:
                metrics['dataframe'].to_excel(writer, sheet_name='metrics')
                detail_df.to_excel(writer, sheet_name='details')
        if metrics["acc"] > best_result:
            save_prefix = os.path.join(args.save_dir, "checkpoint_best")
            model.save(save_prefix, epoch)
            best_result = metrics["acc"]
            logger.info("Best eval F1 {} at epoch {}.\r\n".format(best_result, epoch))

        model.avg_reset()
        model.reset()


if __name__ == "__main__":
    main()
