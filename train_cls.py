import os
import json
import argparse
import time
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler
from src.models.modeling_clstoken import ClassifyModel_CLS
from src.models.model import FineTuningModel
from src.utils import options
from pprint import pprint
import torch

from src.utils.dataset_utils import ClassifyDataset, ClassifyInferDataset
from src.utils.train_utils import create_logger, set_environment, get_time_dif

from transformers import RobertaModel, BertModel
from src.models.modeling_cls import ClassifyModel
import pandas as pd
from transformers import logging
logging.set_verbosity_error()
# args parser
parser = argparse.ArgumentParser("training task.")
options.add_args(parser, mode='train')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# CPU workers
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


def save_result(metrics, model, data='train'):
    if args.get_result:
        df = metrics['dataframe']
        detail_df = model.get_raw_details()
        output_metric_path = os.path.join(args.save_dir, 'Experiments_{}.xlsx'.format(data))
        with pd.ExcelWriter(output_metric_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='metrics')
            detail_df.to_excel(writer, index=False, sheet_name='details')
        logger.info("Model result save in {}!".format(output_metric_path))


def build_model(num_train_steps):
    logger.info("Model update steps {}!".format(num_train_steps))
    logger.info(f"Build Cls {args.encoder} model.")
    bert_model = None
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained(args.bert_model)
    elif args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
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
    model = FineTuningModel(args, network, num_train_steps=num_train_steps)
    # Load parameter
    epoch_pre = 0
    if args.pretrained:
        load_prefix = os.path.join(args.load_dir, "checkpoint_best")
        model.load(load_prefix)
        other_path = load_prefix + ".ot"
        epoch_pre = torch.load(other_path)['epoch']
    return model, epoch_pre


def save_shell_script():
    if os.path.exists('train.sh'):
        with open('train.sh', 'r') as file:
            ret = file.read()
            logger.info("Run Code:\r\n {}".format(ret))
        with open(os.path.join(args.save_dir, "code.sh"), 'w') as file:
            file.write(ret)


def main():
    start_time = time.time()
    logger.info('----------------开始计时----------------')
    logger.info('----------------------------------------')
    best_eval_acc = float("-inf")
    best_train_acc = best_test_acc = 0.
    save_shell_script()
    logger.info("num_workers:{0}.... gpu_nums:{1}".format(args.num_workers, args.gpu_num))

    # Build Dataloader
    logger.info("Loading data...")
    args.data_dir = os.path.join(args.data_dir, args.encoder)
    args.train_path, args.dev_path, args.test_path = [os.path.join(args.data_dir, file_name) for file_name in
                                                      ["train.pkl", "dev.pkl", "test.pkl"]]

    train_dataset, dev_dataset = ClassifyDataset(args.train_path, args), ClassifyInferDataset(args.dev_path, args)
    test_dataset = ClassifyInferDataset(args.test_path, args) if args.test else None

    train_sampler = RandomSampler(train_dataset)
    train_loader, dev_loader, test_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler), DataLoader(
        dataset=dev_dataset, batch_size=args.eval_batch_size), DataLoader(
        dataset=test_dataset, batch_size=args.eval_batch_size) if args.test else None

    # Build Model
    logger.info('----------------Build model...----------------')
    num_train_steps = int(args.max_epoch * len(train_loader) / args.gradient_accumulation_steps)
    model, epoch_pre = build_model(num_train_steps)
    train_start = datetime.now()

    # Begin to train
    logger.info('----------------Begin to train model...----------------')
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
        train_metrics = model.get_metrics(logger)
        if train_metrics['acc'] > best_train_acc:
            best_train_acc = train_metrics['acc']
            save_result(train_metrics, model, data='train')
        model.reset()
        model.avg_reset()

        # Model eval
        logger.info("====== Begin to Evaluate on Dev set...")
        model.evaluate(dev_loader)
        logger.info("Evaluate epoch:[{0:6}] eval loss[{1:.5f}]\r\n".format(epoch, model.dev_loss.avg))
        dev_metrics = model.get_metrics(logger)
        if dev_metrics["acc"] > best_eval_acc:
            save_prefix = os.path.join(args.save_dir, "checkpoint_best")
            model.save(save_prefix, epoch)
            best_eval_acc = dev_metrics["acc"]
            logger.info("Best eval ACC {} at epoch {}.\r\n".format(best_eval_acc, epoch))
            save_result(dev_metrics, model, data='dev')
        model.avg_reset()
        model.reset()

        # Model Testing
        if test_loader is not None:
            model.evaluate(test_loader)
            test_metrics = model.get_metrics(logger)
            if test_metrics["acc"] > best_test_acc:
                best_test_acc = test_metrics["acc"]
                logger.info("Best test ACC {} at epoch {}.\r\n".format(best_test_acc, epoch))
                save_result(test_metrics, model, data='test')
        model.avg_reset()
        model.reset()

    logger.info('----------------------------------------')
    logger.info("----------本次容器运行时长：{}-----------".format(get_time_dif(start_time)))


if __name__ == "__main__":
    main()
