import os.path

from my_datasets import MNLIDatasetPT
from tools.sift_torch import hook_sift_layer, AdversarialLearner
from spm_tokenizer import SPMTokenizer
from datasets import load_dataset
from config import vocab_path, batch_size, max_length, model_dir, model_path, \
    lr, adv_weight, root_path, valid_batch_size, seed, train_data_path, valid_data_path
from transformers.models.deberta_v2 import DebertaV2ForSequenceClassification, \
    DebertaV2Config, DebertaV2Tokenizer
import torch
import time
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from collections.abc import Sequence
from tqdm import tqdm
import numpy as np
import random


def setup_torch_seed(sts_seed):
    torch.manual_seed(sts_seed)
    torch.cuda.manual_seed_all(sts_seed)
    torch.cuda.manual_seed(sts_seed)
    np.random.seed(sts_seed)
    random.seed(sts_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


setup_torch_seed(seed)


def padding(indice, max_len, pad_idx=0):
    """
        补齐方法
    """
    pad_indice = [item + [pad_idx] * max(0, max_len - len(item)) for item in indice]
    return torch.tensor(pad_indice)


def mnli_collate_fn(batch):
    input_ids = [mcf_d['input_ids'] for mcf_d in batch]
    token_type_ids = [mcf_d['type_ids'] for mcf_d in batch]

    attention_mask = [mcf_d['attention_mask'] for mcf_d in batch]
    labels = [mcf_d['labels'] for mcf_d in batch]

    mcf_max_length = max(len(t) for t in input_ids)
    input_ids_padded = padding(input_ids, mcf_max_length)
    token_type_ids_padded = padding(token_type_ids, mcf_max_length)

    attention_mask_padded = padding(attention_mask, mcf_max_length)

    return input_ids_padded, token_type_ids_padded, attention_mask_padded, \
           torch.tensor(labels)


tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)
# mnli_dataset = load_dataset('glue', 'mnli')
# train_dataset = mnli_dataset['train']
# valid_dataset = mnli_dataset['validation_matched']
# metric = load_metric('glue', 'mnli')

train_dataset = MNLIDatasetPT(tokenizer, train_data_path, max_length)
valid_dataset = MNLIDatasetPT(tokenizer, valid_data_path, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mnli_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True, collate_fn=mnli_collate_fn)

config = DebertaV2Config.from_pretrained(model_dir)
config.output_attentions = False
config.output_hidden_states = False
config.num_labels = 3

epochs = 1

warmup_proportion = 0.01

weight_decay = 0.01

model = DebertaV2ForSequenceClassification(config)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint, strict=False)

model.to('cuda')

num_training_steps = len(train_loader) * epochs
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_proportion * num_training_steps,
    num_training_steps=(1 - warmup_proportion) * num_training_steps
)

global_step = 0
for epoch in range(1, epochs + 1):
    model.train()

    adv_modules = hook_sift_layer(model,
                                  hidden_size=model.config.hidden_size,
                                  learning_rate=1e-4,
                                  init_perturbation=1e-2)

    adv = AdversarialLearner(model, adv_modules)

    correct = 0
    data_num = 0
    batch_id = 0
    model.train()
    pre_acc = 0
    for data in tqdm(train_loader, desc=f'epoch {epoch} training'):
        batch_id += 1
        input_ids = data[0].to('cuda')
        token_type_ids = data[1].to('cuda')
        attention_mask = data[2].to('cuda')
        labels = data[3].to('cuda')

        input_data = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'attention_mask': attention_mask,
                      'labels': labels}

        outputs = model(**input_data)

        if adv_weight > 0:
            def pert_logits_fn(plf_model,
                               **plf_input_data):
                o = plf_model(**plf_input_data)
                plf_logits = o['logits']
                if isinstance(plf_logits, Sequence):
                    plf_logits = plf_logits[-1]
                return plf_logits

        logits = outputs['logits']
        loss = outputs['loss']
        loss += adv.loss(logits, pert_logits_fn,
                         loss_fn="symmetric-kl", **input_data) * adv_weight
        loss = loss / 2

        probs = F.softmax(logits, dim=-1)
        correct += sum(probs.argmax(dim=-1) == labels).item()
        data_num += labels.shape[0]
        acc = correct / data_num

        # if pre_acc != 0 and pre_acc - acc < 0.001:
        #     break

        pre_acc = acc

        global_step += 1
        if global_step % 100 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, learning rate: %.7f, time: %s"
                  % (global_step, epoch, batch_id, loss, acc, scheduler.get_last_lr()[0], time.asctime()))

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # model.load_state_dict(torch.load(os.path.join(root_path, 'model\\pytorch_model.bin')))
    model.eval()
    correct = 0
    data_num = 0
    with torch.no_grad():
        for data in tqdm(valid_loader, desc=f'epoch {epoch} dev'):
            input_ids = data[0].to('cuda')
            token_type_ids = data[1].to('cuda')
            attention_mask = data[2].to('cuda')
            labels = data[3].to('cuda')
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

            logits = outputs['logits']

            probs = F.softmax(logits, dim=-1)
            correct += sum(probs.argmax(dim=-1) == labels).item()
            data_num += labels.shape[0]

    acc = correct / data_num

    print("epoch: %d, dev accuracy: %.5f, time: %s" % (epoch, acc, time.asctime()))

torch.save(model.state_dict(), os.path.join(root_path, 'model\\pytorch_model.bin'))
