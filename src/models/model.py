import torch
import torch.nn as nn
from src.models.optimizer import BertAdam as Adam
from src.utils.model_utils import AverageMeter
from tqdm import tqdm
from src.utils.attack_train_utils import FGSM, FGM, PGD, FreeAT
import torch.nn.functional as F


class PredictModel():
    def __init__(self, args, network):
        self.args = args
        self.train_loss = AverageMeter()
        self.dev_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network

        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network

        if self.args.gpu_num > 0:
            self.network.cuda()
            self.network.perm.cuda()
            self.network.weight.cuda()

    def avg_reset(self):
        self.train_loss.reset()
        self.dev_loss.reset()

    @torch.no_grad()
    def evaluate(self, dev_data_list):
        self.network.eval()
        for batch in tqdm(dev_data_list):
            output_dict = self.network(**batch)
            loss = output_dict["loss"]
            self.dev_loss.update(loss.item(), 1)

    @torch.no_grad()
    def predict(self, test_data_list):
        self.network.eval()
        pred_list = []
        for batch in tqdm(test_data_list):
            output_dict = self.network.predict(**batch)
            text = output_dict['raw_text']
            label = output_dict['pred_label']
            score = output_dict['pred_score']
            for i in range(len(label)):
                rec_lis = [text[i], int(label[i]), round(float(score[i]), 3)]
                pred_list.append(rec_lis)
        return pred_list

    def reset(self):
        self.mnetwork.reset()

    def get_raw_details(self):
        return self.mnetwork.get_raw_details()

    def get_metrics(self, logger=None):
        return self.mnetwork.get_metrics_predict(logger)

    def load(self, state_path):
        self.network.load_state_dict(torch.load(state_path))


class FineTuningModel():
    def __init__(self, args, network, state_dict=None, num_train_steps=1):
        self.args = args
        self.train_loss = AverageMeter()
        self.dev_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network
        self.kl_weight = args.kl_weight
        if state_dict is not None:
            print("Load Model!")
            self.network.load_state_dict(state_dict["state"])
        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network

        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.bert_module.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': args.bert_weight_decay, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.bert_module.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.named_parameters() if not n.startswith("bert_module.")],
             "weight_decay": args.weight_decay, "lr": args.learning_rate}
        ]
        self.optimizer = Adam(optimizer_parameters,
                              lr=args.learning_rate,
                              warmup=args.warmup,
                              t_total=num_train_steps,
                              max_grad_norm=args.grad_clipping,
                              schedule=args.warmup_schedule)
        if self.args.gpu_num > 0:
            self.network.cuda()

        if self.args.attack == 'FGSM':
            self.attack = FGSM(self.mnetwork)
        elif self.args.attack == 'FGM':
            self.attack = FGM(self.mnetwork)
        elif self.args.attack == 'PGD':
            self.attack = PGD(self.mnetwork)
            self.attack_time = 3
        elif self.args.attack == 'FreeAT':
            self.attack = FreeAT(self.mnetwork)
            self.attack_time = 5

    def avg_reset(self):
        self.train_loss.reset()
        self.dev_loss.reset()

    def update(self, tasks):
        self.network.train()
        # r-drop
        loss = self.r_drop(tasks)
        self.train_loss.update(loss.item(), 1)
        if self.args.gradient_accumulation_steps > 1:
            loss /= self.args.gradient_accumulation_steps
        loss.backward()
        # attack training
        self.attack_train(tasks)

        if (self.step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
        self.step += 1

    @torch.no_grad()
    def evaluate(self, dev_dataloader):
        self.network.eval()
        with torch.no_grad():
            for batch in tqdm(dev_dataloader):
                output_dict = self.network(**batch, mode='eval')
                loss = output_dict["loss"]
                self.dev_loss.update(loss.item(), 1)
        # self.network.train()

    @torch.no_grad()
    def predict(self, test_data_list):
        self.network.eval()
        with torch.no_grad():
            for batch in tqdm(test_data_list):
                self.network.predict(**batch, mode="eval")

    def attack_train(self, tasks):
        if self.args.attack == 'FGSM' or self.args.attack == 'FGM':
            # 备份，加扰动
            self.attack.attack()
            # 对抗样本前向传播
            loss_adv = self.mnetwork(**tasks)['loss']
            loss_adv.backward()
            # 恢复扰动前状况
            self.attack.restore()
        elif self.args.attack == 'PGD' or self.args.attack == 'FreeAT':
            self.attack.backup_grad()
            for _t in range(self.attack_time):
                self.attack.attack(is_first_attack=(_t == 0))
                if _t != self.attack_time - 1:
                    self.mnetwork.zero_grad()
                else:
                    self.attack.restore_grad()
                loss_adv = self.mnetwork(**tasks)['loss']
                loss_adv.backward()
            self.attack.restore()

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def r_drop(self, tasks):
        output_dict = self.mnetwork(**tasks, mode='train')
        if self.args.r_drop:
            output_dict2 = self.mnetwork(**tasks)
            # cross entropy loss for classifier
            ce_loss = 0.5 * output_dict['loss'] + 0.5 * output_dict2['loss']
            kl_loss = self.compute_kl_loss(output_dict['logits'], output_dict2['logits'])
            loss = ce_loss + self.kl_weight * kl_loss
        else:
            loss = output_dict["loss"]
        return loss

    def reset(self):
        if isinstance(self.mnetwork, nn.DataParallel):
            models = self.mnetwork.module
        else:
            models = self.mnetwork
        models.reset()

    def get_raw_details(self):
        if isinstance(self.mnetwork, nn.DataParallel):
            models = self.mnetwork.module
        else:
            models = self.mnetwork
        return models.get_raw_details()

    def get_metrics(self, logger=None):
        if isinstance(self.mnetwork, nn.DataParallel):
            models = self.mnetwork.module
        else:
            models = self.mnetwork
        return models.get_metrics(logger)

    def save(self, prefix, epoch):

        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        other_params = {
            'optimizer': self.optimizer.state_dict(),
            'config': self.args,
            'epoch': epoch
        }
        state_path = prefix + ".pt"
        other_path = prefix + ".ot"
        torch.save(other_params, other_path)
        torch.save(network_state, state_path)
        print('model saved to {}'.format(prefix))

    def load(self, prefix):
        state_path = prefix + ".pt"
        other_path = prefix + ".ot"
        self.network.load_state_dict(torch.load(state_path))
        self.optimizer.load_state_dict(torch.load(other_path)['optimizer'])
