from paddle.io import Dataset as PDDataset
from torch.utils.data import Dataset as PTDataset
import random
from tqdm import tqdm
import csv
from config import mnli_label2id, mnli_id2label


def _truncate_segments(segments, max_num_tokens, rng):
    """
    Truncate sequence pair according to original BERT implementation:
    https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
    """
    while True:
        if sum(len(s) for s in segments) <= max_num_tokens:
            break

        segments = sorted(segments, key=lambda s: len(s), reverse=True)
        trunc_tokens = segments[0]

        assert len(trunc_tokens) >= 1

        if rng and rng.random() < 0.5:
            trunc_tokens.pop(0)
        else:
            trunc_tokens.pop()
    return segments


# class MNLIDatasetPD(PDDataset):
#
#     def __init__(self, tokenizer, data_path, max_length):
#         super(MNLIDatasetPD, self).__init__()
#
#         with open(data_path, 'r', encoding='utf-8') as f:
#             reader = csv.reader(f, delimiter="\t", quotechar=None)
#             lines = []
#             for line in reader:
#                 lines.append(line)
#         lines = lines[1:]
#         sentence1 = [line[8] for line in lines]
#         sentence2 = [line[9] for line in lines]
#         self.labels = [mnli_label2id[line[11]] for line in lines]
#
#         self.len_ = len(sentence1)
#
#         self.input_ids = []
#         self.type_ids = []
#         self.position_ids = []
#         self.input_mask = []
#
#         for i in tqdm(range(self.len_), desc='transform data to ids'):
#             sen1 = sentence1[i]
#             sen2 = sentence2[i]
#             segments = _truncate_segments([tokenizer.tokenize(sen1), tokenizer.tokenize(sen2)],
#                                           max_length,
#                                           None)
#             tokens = ['[CLS]']
#             type_ids = [0]
#             for j, s in enumerate(segments):
#                 tokens.extend(s)
#                 tokens.append('[SEP]')
#                 type_ids.extend([j] * (len(s) + 1))
#
#             token_ids = tokenizer.convert_tokens_to_ids(tokens)
#             pos_ids = list(range(len(token_ids)))
#             input_mask = [1] * len(token_ids)
#             self.input_ids.append(token_ids)
#             self.type_ids.append(type_ids)
#             self.position_ids.append(pos_ids)
#             self.input_mask.append(input_mask)
#
#     def __getitem__(self, item):
#         return {'input_ids': self.input_ids[item],
#                 'type_ids': self.type_ids[item],
#                 'position_ids': self.position_ids[item],
#                 'attention_mask': self.input_mask[item],
#                 'labels': self.labels[item]}
#
#     def __len__(self):
#         return self.len_

class MNLIDatasetPD(PDDataset):

    def __init__(self, tokenizer, data_path, max_length):
        super(MNLIDatasetPD, self).__init__()

        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = []
            for line in reader:
                lines.append(line)
        lines = lines[1:]
        sentence1 = [line[8] for line in lines]
        sentence2 = [line[9] for line in lines]
        self.labels = [mnli_label2id[line[11]] for line in lines]

        self.len_ = len(sentence1)

        encode_batch = tokenizer(text=sentence1,
                                 text_pair=sentence2,
                                 max_seq_len=max_length)

        self.input_ids = [e_b['input_ids'] for e_b in encode_batch]
        self.type_ids = [e_b['token_type_ids'] for e_b in encode_batch]

    def __getitem__(self, item):
        return {'input_ids': self.input_ids[item],
                'type_ids': self.type_ids[item],
                'labels': self.labels[item]}

    def __len__(self):
        return self.len_


class MNLIDatasetPT(PTDataset):

    def __init__(self, tokenizer, data_path, max_length):
        super(MNLIDatasetPT, self).__init__()

        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = []
            for line in reader:
                lines.append(line)
        lines = lines[1:]
        sentence1 = [line[8] for line in lines]
        sentence2 = [line[9] for line in lines]
        self.labels = [mnli_label2id[line[11]] for line in lines]

        self.len_ = len(sentence1)

        self.input_ids = []
        self.type_ids = []
        self.position_ids = []
        self.input_mask = []

        for i in tqdm(range(self.len_), desc='transform data to ids'):
            sen1 = sentence1[i]
            sen2 = sentence2[i]
            segments = _truncate_segments([tokenizer.tokenize(sen1), tokenizer.tokenize(sen2)],
                                          max_length,
                                          None)
            tokens = ['[CLS]']
            type_ids = [0]
            for j, s in enumerate(segments):
                tokens.extend(s)
                tokens.append('[SEP]')
                type_ids.extend([j] * (len(s) + 1))

            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            pos_ids = list(range(len(token_ids)))
            input_mask = [1] * len(token_ids)
            self.input_ids.append(token_ids)
            self.type_ids.append(type_ids)
            self.position_ids.append(pos_ids)
            self.input_mask.append(input_mask)

    def __getitem__(self, item):
        return {'input_ids': self.input_ids[item],
                'type_ids': self.type_ids[item],
                'position_ids': self.position_ids[item],
                'attention_mask': self.input_mask[item],
                'labels': self.labels[item]}

    def __len__(self):
        return self.len_

