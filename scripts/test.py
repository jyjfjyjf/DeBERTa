import numpy as np
from my_datasets import MNLIDatasetPD
from scripts.sift import hook_sift_layer, AdversarialLearner
from spm_tokenizer import SPMTokenizer
from collections.abc import Sequence
from config import vocab_path, batch_size, max_length, model_dir, model_path, \
    paddle_model_path, lr, adv_weight
import paddlenlp
import paddle
from transformers import DebertaV2ForSequenceClassification as torch_DebertaV2ForSequenceClassification
import torch
from deberta_config import DebertaV2Config
from sequence_classification import SequenceClassificationModel
from tqdm import tqdm
import paddle.nn.functional as F
from paddlenlp.transformers import LinearDecayWithWarmup


def padding(indice, max_len, pad_idx=0):
    """
        补齐方法
    """
    pad_indice = [item + [pad_idx] * max(0, max_len - len(item)) for item in indice]
    return paddle.to_tensor(pad_indice)


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
           paddle.to_tensor(labels)


vocab_path = vocab_path

tokenizer = SPMTokenizer(vocab_path)
train_data, dev_data = paddlenlp.datasets.load_dataset('glue', 'mnli', splits=("train", "dev_mismatched"))

train_dataset = MNLIDatasetPD(tokenizer=tokenizer,
                              samples=train_data,
                              max_length=max_length)

dev_dataset = MNLIDatasetPD(tokenizer=tokenizer,
                            samples=dev_data,
                            max_length=max_length)

train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mnli_collate_fn)
dev_loader = paddle.io.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=mnli_collate_fn)

"""模型torch转paddle"""
# torch_dict = torch.load(model_path)
#
# paddle_dict = {}
#
# fc_names = ['self.query', 'self.key', 'self.value', 'dense', 'self.pos_query_proj',
#             'self.pos_key_proj']
#
# for key in torch_dict:
#     weight = torch_dict[key].cpu().numpy()
#     flag = [i in key for i in fc_names]
#     if any(flag):
#         print('weight {} need to be trans'.format(key))
#         weight = weight.transpose()
#     paddle_dict[key] = weight.astype('float32')
#
# paddle.save(paddle_dict, paddle_model_path)

config = DebertaV2Config.from_pretrain(model_dir)
config.use_return_dict = False
config.output_attentions = False
config.output_hidden_states = False


def test_forward(tf_config):
    from transformers import DebertaV2Config
    config_torch = DebertaV2Config.from_pretrained(model_dir)
    model_torch = torch_DebertaV2ForSequenceClassification(config_torch)
    model_paddle = SequenceClassificationModel(tf_config)
    model_torch.eval()
    model_paddle.eval()
    torch_checkpoint = torch.load(model_path)
    model_torch.load_state_dict(torch_checkpoint, strict=False)
    paddle_checkpoint = paddle.load(paddle_model_path)
    model_paddle.set_state_dict(paddle_checkpoint)

    x = np.random.randint(1, 10, size=(2, 10))
    input_torch = torch.tensor(x, dtype=torch.long)
    out_torch = model_torch(input_torch)

    input_paddle = paddle.to_tensor(x, dtype='int64')
    output_paddle = model_paddle(input_paddle, return_dict=False)

    print(out_torch)

    print(output_paddle)


# test_forward(config)

epochs = 1

warmup_proportion = 0.01

weight_decay = 0.01

model = SequenceClassificationModel(config)
checkpoint = paddle.load(paddle_model_path)
model.set_state_dict(checkpoint)

num_training_steps = len(train_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(lr, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=list(model.parameters()),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
)
metric = paddle.metric.Accuracy()

global_step = 0
for epoch in range(1, epochs + 1):
    correct = 0
    data_num = 0
    batch_id = 0
    model.train()
    adv_modules = hook_sift_layer(model,
                                  hidden_size=model.config.hidden_size,
                                  learning_rate=1e-4,
                                  init_perturbation=1e-2)

    adv = AdversarialLearner(model, adv_modules)
    for data in tqdm(train_loader, desc=f'epoch {epoch} training'):
        input_ids = paddle.to_tensor(data[0])
        token_type_ids = paddle.to_tensor(data[1])
        attention_mask = paddle.to_tensor(data[2])
        position_ids = paddle.to_tensor(data[3])
        labels = paddle.to_tensor(data[4])

        input_data = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'attention_mask': attention_mask,
                      'labels': labels,
                      'position_ids': position_ids}

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
        loss = loss / (1 + adv_weight)


        probs = F.softmax(logits, axis=-1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()
        # probs = F.softmax(logits, axis=1)
        # correct += np.sum((probs.argmax(axis=1) == labels).numpy() != 0)
        # data_num += labels.shape[0]
        # acc = correct / data_num

        global_step += 1
        if global_step % 10 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (
                global_step, epoch, batch_id, loss, acc))

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # for p in model.parameters():
        #     print(p.grad)

        optimizer.clear_grad()

    with paddle.no_grad():
        model.eval()
        for data in tqdm(dev_loader, desc=f'epoch {epoch} dev'):
            input_ids = paddle.to_tensor(data[0])
            token_type_ids = paddle.to_tensor(data[1])
            attention_mask = paddle.to_tensor(data[2])
            position_ids = paddle.to_tensor(data[3])
            labels = paddle.to_tensor(data[4])
            outputs = model(input_ids=input_ids,
                            type_ids=token_type_ids,
                            input_mask=attention_mask,
                            position_ids=position_ids)

            logits = outputs['logits']

            probs = F.softmax(logits, axis=-1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

        print('dev accuracy %d'.format(acc))
