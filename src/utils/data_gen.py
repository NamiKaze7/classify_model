import torch
import random


class PredictBatchGen(object):
    def __init__(self, args, data, data_mode='test'):
        self.is_train = data_mode == "train"
        self.args = args

        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item.token_ids)
            attention_mask = torch.from_numpy(item.attention_masks)
            token_type_ids = torch.from_numpy(item.token_type_ids)
            raw_text = item.raw_text
            all_data.append((input_ids, attention_mask, token_type_ids, raw_text))
        self.data = PredictBatchGen.make_batches(all_data,
                                                 args.batch_size if self.is_train else args.eval_batch_size,
                                                 self.is_train)
        self.offset = 0

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            input_ids_batch, attention_mask_batch, token_type_ids_batch, raw_text_batch = zip(*batch)
            bsz = len(batch)
            token_ids = torch.LongTensor(bsz, self.args.max_seq_len)
            attention_masks = torch.LongTensor(bsz, self.args.max_seq_len)
            token_type_ids = torch.LongTensor(bsz, self.args.max_seq_len).fill_(0)
            raw_texts = []

            for i in range(bsz):
                token_ids[i] = input_ids_batch[i]
                attention_masks[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]
                raw_texts.append(raw_text_batch[i])
            out_batch = {"token_ids": token_ids, "attention_masks": attention_masks, "token_type_ids": token_type_ids,
                         "raw_text": raw_texts
                         }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield out_batch
