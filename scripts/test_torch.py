from my_datasets import MNLIDatasetPT
from spm_tokenizer import SPMTokenizer
from datasets import load_dataset, load_metric
from config import vocab_path, batch_size, max_length, model_dir, model_path, lr
from transformers import DebertaV2ForSequenceClassification, DebertaV2Config
import torch
import time
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F


def padding(indice, max_len, pad_idx=0):
    """
        补齐方法
    """
    pad_indice = [item + [pad_idx] * max(0, max_len - len(item)) for item in indice]
    return torch.tensor(pad_indice)


def mnli_collate_fn(batch):
    input_ids = [mcf_d['input_ids'] for mcf_d in batch]
    token_type_ids = [mcf_d['type_ids'] for mcf_d in batch]
    position_ids = [mcf_d['position_ids'] for mcf_d in batch]
    attention_mask = [mcf_d['attention_mask'] for mcf_d in batch]
    labels = [mcf_d['labels'] for mcf_d in batch]

    mcf_max_length = max(len(t) for t in input_ids)
    input_ids_padded = padding(input_ids, mcf_max_length)
    token_type_ids_padded = padding(token_type_ids, mcf_max_length)
    position_ids_padded = padding(position_ids, mcf_max_length)
    attention_mask_padded = padding(attention_mask, mcf_max_length)

    return input_ids_padded, token_type_ids_padded, attention_mask_padded, position_ids_padded, \
        torch.tensor(labels)


tokenizer = SPMTokenizer(vocab_path)
mnli_dataset = load_dataset('glue', 'mnli')
train_dataset = mnli_dataset['train']
valid_dataset = mnli_dataset['validation_matched']
# metric = load_metric('glue', 'mnli')

train_dataset = MNLIDatasetPT(tokenizer, train_dataset, max_length)
valid_dataset = MNLIDatasetPT(tokenizer, valid_dataset, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mnli_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=mnli_collate_fn)

config = DebertaV2Config.from_pretrained(model_dir)
config.output_attentions = False
config.output_hidden_states = False
config.num_labels = 3

epochs = 3

warmup_proportion = 0.1

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

model.train()
global_step = 0
for epoch in range(1, epochs + 1):
    correct = 0
    data_num = 0
    model.train()
    for batch_id, data in enumerate(train_loader, start=1):
        input_ids = data[0].to('cuda')
        token_type_ids = data[1].to('cuda')
        attention_mask = data[2].to('cuda')
        position_ids = data[3].to('cuda')
        labels = data[4].to('cuda')
        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        position_ids=position_ids)

        logits = outputs['logits']
        loss = outputs['loss']

        probs = F.softmax(logits, dim=1)
        correct += sum(probs.argmax(dim=1) == labels).item()
        data_num += labels.shape[0]
        acc = correct / data_num

        global_step += 1
        if global_step % 100 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, time: %s"
                  % (global_step, epoch, batch_id, loss, acc, time.asctime()))

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad:
        for batch_id, data in enumerate(valid_loader, start=1):
            input_ids = data[0].to('cuda')
            token_type_ids = data[1].to('cuda')
            attention_mask = data[2].to('cuda')
            position_ids = data[3].to('cuda')
            labels = data[4].to('cuda')
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            position_ids=position_ids)

            logits = outputs['logits']
            loss = outputs['loss']

            probs = F.softmax(logits, dim=1)
            correct += sum(probs.argmax(dim=1) == labels).item()
            data_num += labels.shape[0]
            acc = correct / data_num

            global_step += 1
            if global_step % 100 == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, time: %s"
                      % (global_step, epoch, batch_id, loss, acc, time.asctime()))

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

