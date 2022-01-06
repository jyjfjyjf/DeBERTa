from paddle.io import Dataset as pd_dataset
from torch.utils.data import Dataset as pt_dataset
import random
from tqdm import tqdm


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

        if rng.random() < 0.5:
            trunc_tokens.pop(0)
        else:
            trunc_tokens.pop()
    return segments


class MNLIDatasetPD(pd_dataset):

    def __init__(self, tokenizer, samples, max_length):
        super(MNLIDatasetPD, self).__init__()

        sentence1 = [d['sentence1'] for d in samples.data]
        sentence2 = [d['sentence2'] for d in samples.data]
        self.labels = []
        for d in samples.data:
            if d['labels'] == 0:
                self.labels.append(2)
            elif d['labels'] == 1:
                self.labels.append(0)
            else:
                self.labels.append(1)

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
                                          random)
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


class MNLIDatasetPT(pt_dataset):

    def __init__(self, tokenizer, samples, max_length):
        super(MNLIDatasetPT, self).__init__()

        sentence1 = [d['premise'] for d in samples]
        sentence2 = [d['hypothesis'] for d in samples]
        self.labels = [d['label'] for d in samples]

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
                                          random)
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

