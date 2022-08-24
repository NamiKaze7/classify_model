import os
import copy
import torch
import logging
from torch.cuda.amp import autocast as ac
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from src.utils.attack_train_utils import FGM, PGD
from src.utils.functions_utils import load_model_and_parallel, swa
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_model(opt, model, global_step, best_acc):
    output_dir = os.path.join(opt.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of models distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info(f'Saving models & optimizer & scheduler checkpoint to {output_dir}')
    state = {'state_dict': model_to_save.state_dict(), 'dev_acc': best_acc}
    torch.save(state, os.path.join(output_dir, 'models.pt'))


def build_optimizer_and_scheduler(opt, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def train(opt, model, train_dataset, dev_dataset):
    swa_raw_model = copy.deepcopy(model)

    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              sampler=train_sampler,
                              num_workers=0)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=opt.dev_batch_size)

    scaler = None
    if opt.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    model, device, best_acc = load_model_and_parallel(model, opt.gpu_ids)

    use_n_gpus = False
    if hasattr(model, "module"):
        use_n_gpus = True

    t_total = len(train_loader) * opt.train_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)

    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {opt.train_epochs}")
    logger.info(f"  Total training batch size = {opt.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0

    model.zero_grad()

    fgm, pgd = None, None
    attack_train_mode = opt.attack_train.lower()
    if attack_train_mode == 'fgm':
        fgm = FGM(model=model)
    elif attack_train_mode == 'pgd':
        pgd = PGD(model=model)

    pgd_k = 3
    log_loss_steps = opt.log_loss_steps
    avg_loss = 0.

    for epoch in range(opt.train_epochs):
        model.train()
        for step, batch_data in tqdm(enumerate(train_loader)):

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)

            if opt.use_fp16:
                with ac():
                    logits, loss = model(**batch_data)
            else:
                logits, loss = model(**batch_data)

            if use_n_gpus:
                loss = loss.mean()

            if opt.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward(loss.clone().detach())

            if fgm is not None:
                fgm.attack()

                if opt.use_fp16:
                    with ac():
                        loss_adv = model(**batch_data)[-1]
                else:
                    loss_adv = model(**batch_data)[-1]

                if use_n_gpus:
                    loss_adv = loss_adv.mean()

                if opt.use_fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()

                fgm.restore()

            elif pgd is not None:
                pgd.backup_grad()

                for _t in range(pgd_k):
                    pgd.attack(is_first_attack=(_t == 0))

                    if _t != pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()

                    if opt.use_fp16:
                        with ac():
                            loss_adv = model(**batch_data)[-1]
                    else:
                        loss_adv = model(**batch_data)[-1]

                    if use_n_gpus:
                        loss_adv = loss_adv.mean()

                    if opt.use_fp16:
                        scaler.scale(loss_adv).backward()
                    else:
                        loss_adv.backward()

                pgd.restore()

            if opt.use_fp16:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

            # optimizer.step()
            if opt.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            model.zero_grad()

            global_step += 1

            if global_step % log_loss_steps == 0:

                # 计算 acc1
                rightCnt = sum(torch.topk(logits, 1)[1].reshape(-1) == batch_data['labels']).item()
                acc1 = rightCnt / batch_data['labels'].shape[0]

                avg_loss /= log_loss_steps
                logger.info(
                    'Step: %d / %d ----> total loss: %.5f ----> acc1: %.5f' % (
                        global_step, t_total, avg_loss, acc1))
                avg_loss = 0.
            else:
                avg_loss += loss.item()
        
        logger.info(">>>>>> begin to evaluate >>>>>>")
        if isinstance(model,torch.nn.DataParallel):
            model_p = model.module
        else:
            model_p = model
        with torch.no_grad():
            dev_acc = 0.
            dev_ac = dev_count = 0
            model_p.eval()
            for step, batch_data in tqdm(enumerate(dev_loader)):
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(device)
                acc3 = model_p.predict(batch_data['token_ids'], batch_data['attention_masks'], batch_data['token_type_ids'])
                pred = acc3[1][:, 0].squeeze(-1)
                assert pred.size() == batch_data['labels'].size()
                dev_ac += (pred == batch_data['labels']).sum().item()
                dev_count += pred.size(0)
            dev_acc = dev_ac / dev_count

            if dev_acc > best_acc:
                best_acc = dev_acc
            save_model(opt, model, global_step, dev_acc)

            logger.info('------------- epoch: {0} dev_acc: {1} best_acc: {2}------------'.format(epoch, dev_acc, best_acc))
    
    swa(swa_raw_model, opt.output_dir, swa_start=opt.swa_start)

    # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()
    logger.info('Train done')
