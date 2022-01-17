from functools import partial

import numpy as np
import paddle
import paddlenlp
import torch
from datasets import load_dataset
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import load_dataset as ppnlp_load_dataset
from reprod_log import ReprodDiffHelper, ReprodLogger
from transformers import DataCollatorWithPadding, DebertaV2Tokenizer
from config import max_length, vocab_path, model_dir
from tools.my_datasets import _truncate_segments
from tools.spm_tokenizer import SPMTokenizer
import random


def build_paddle_data_pipeline():
    def read():
        train_data, dev_data = paddlenlp.datasets.load_dataset('glue', 'mnli', splits=("train", "dev_mismatched"))
        # ['contradiction', 'entailment', 'neutral']
        for d in train_data.data:
            if d['labels'] == 0:
                d['labels'] = 2
            elif d['labels'] == 1:
                d['labels'] = 0
            else:
                d['labels'] = 1
            yield {"sentence1": d["sentence1"], 'sentence2': d['sentence2'], "labels": d["labels"]}

    def convert_example(example, tokenizer, max_length=max_length):
        labels = np.array([example["labels"]], dtype="int64")
        segments = _truncate_segments([tokenizer.tokenize(example["sentence1"]), tokenizer.tokenize(example["sentence2"])],
                                      max_length,
                                      random)
        tokens = ['[CLS]']
        type_ids = [0]
        for j, s in enumerate(segments):
            tokens.extend(s)
            tokens.append('[SEP]')
            type_ids.extend([j] * (len(s) + 1))

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        return {
            "input_ids": np.array(
                token_ids, dtype="int64"),
            "token_type_ids": np.array(
                type_ids, dtype="int64"),
            "labels": labels,
        }

    # load tokenizer
    tokenizer = SPMTokenizer(vocab_path)
    # load data
    dataset_test = ppnlp_load_dataset(
        read, lazy=False)
    trans_func = partial(convert_example, tokenizer=tokenizer, max_length=128)
    # tokenize data
    dataset_test = dataset_test.map(trans_func, lazy=False)
    collate_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "labels": Stack(dtype="int64"), }): fn(samples)
    test_sampler = paddle.io.SequenceSampler(dataset_test)
    test_batch_sampler = paddle.io.BatchSampler(
        sampler=test_sampler, batch_size=4)
    data_loader_test = paddle.io.DataLoader(
        dataset_test,
        batch_sampler=test_batch_sampler,
        num_workers=0,
        collate_fn=collate_fn, )

    return dataset_test, data_loader_test


def build_torch_data_pipeline():
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)

    def preprocess_function(examples):
        result = tokenizer(
            text=examples["premise"],
            text_pair=examples["hypothesis"],
            padding=False,
            max_length=max_length,
            truncation=True,
            return_token_type_ids=True, )
        if "label" in examples:
            result["labels"] = [examples["label"]]
        return result

    # load data
    mnli_dataset = load_dataset('glue', 'mnli')
    train_dataset = mnli_dataset['train']  # ["entailment", "neutral", "contradiction"]
    valid_dataset = mnli_dataset['validation_matched']
    dataset_test = train_dataset.map(
        preprocess_function,
        batched=False,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on dataset", )
    dataset_test.set_format(
        "np", columns=["input_ids", "token_type_ids", "labels"])
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=4,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=collate_fn, )
    return dataset_test, data_loader_test


def test_data_pipeline():
    diff_helper = ReprodDiffHelper()
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline()
    torch_dataset, torch_dataloader = build_torch_data_pipeline()

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    # random choose 5 images and check
    for idx in range(5):
        rnd_idx = np.random.randint(0, len(paddle_dataset))
        for k in ["input_ids", "token_type_ids", "labels"]:

            logger_paddle_data.add(f"dataset_{idx}_{k}",
                                   paddle_dataset[rnd_idx][k])

            logger_torch_data.add(f"dataset_{idx}_{k}",
                                  torch_dataset[rnd_idx][k])

    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx >= 5:
            break
        for i, k in enumerate(["input_ids", "token_type_ids", "labels"]):
            logger_paddle_data.add(f"dataloader_{idx}_{k}",
                                   paddle_batch[i].numpy())
            logger_torch_data.add(f"dataloader_{idx}_{k}",
                                  torch_batch[k].cpu().numpy())

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report()


if __name__ == "__main__":
    test_data_pipeline()
