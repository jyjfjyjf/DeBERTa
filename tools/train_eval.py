import numpy as np
from my_datasets import MNLIDatasetPD
from collections.abc import Sequence
from config import vocab_path, batch_size, max_length, model_dir, lr, adv_weight, seed, train_data_path, \
    valid_data_path, valid_batch_size, paddle_model_path
import paddle
from tqdm import tqdm
import paddle.nn.functional as F
from paddlenlp.transformers import LinearDecayWithWarmup
import random
from paddle_deberta.paddlenlp.transformers.deberta.deberta_config import DebertaConfig
from paddle_deberta.paddlenlp.transformers.deberta.tokenization_deberta import DebertaTokenizer
from paddle_deberta.paddlenlp.transformers.deberta.modeling_deberta import DebertaForSequenceClassification
from logger import logger


def setup_paddle_seed(sps_seed):
    paddle.seed(sps_seed)
    np.random.seed(sps_seed)
    random.seed(sps_seed)


setup_paddle_seed(seed)


def padding(indice, max_len, pad_idx=0):
    """
        补齐方法
    """
    pad_indice = [item + [pad_idx] * max(0, max_len - len(item)) for item in indice]
    return paddle.to_tensor(pad_indice)


def mnli_collate_fn(batch):
    mcf_input_ids = [mcf_d['input_ids'] for mcf_d in batch]
    mcf_token_type_ids = [mcf_d['type_ids'] for mcf_d in batch]
    mcf_labels = [mcf_d['labels'] for mcf_d in batch]

    mcf_max_length = max(len(t) for t in mcf_input_ids)
    input_ids_padded = padding(mcf_input_ids, mcf_max_length)
    token_type_ids_padded = padding(mcf_token_type_ids, mcf_max_length)

    return input_ids_padded, token_type_ids_padded, \
        paddle.to_tensor(mcf_labels)


vocab_path = vocab_path

tokenizer = DebertaTokenizer.from_pretrained(model_dir)

config = DebertaConfig.from_pretrain(model_dir)
# config = DebertaV2Config.from_pretrain(model_dir)
config.use_return_dict = False
config.output_attentions = False
config.output_hidden_states = False
config.num_labels = 3

model = DebertaForSequenceClassification.from_pretrained(model_dir, config)

# tokenizer = GPTTokenizer.from_pretrained(model_dir)
# train_data, dev_data = paddlenlp.datasets.load_dataset('glue', 'mnli', splits=("train", "dev_mismatched"))

train_dataset = MNLIDatasetPD(tokenizer=tokenizer,
                              data_path=train_data_path,
                              max_length=max_length)

dev_dataset = MNLIDatasetPD(tokenizer=tokenizer,
                            data_path=valid_data_path,
                            max_length=max_length)

train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mnli_collate_fn)
dev_loader = paddle.io.DataLoader(dev_dataset, batch_size=valid_batch_size, shuffle=False, collate_fn=mnli_collate_fn)

epochs = 3

warmup_proportion = 0.01

weight_decay = 0.01

# model = SequenceClassificationModel(config)

# model = torch_DebertaV2ForSequenceClassification(config)

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

# optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warmup_proportion * num_training_steps,
#     num_training_steps=(1 - warmup_proportion) * num_training_steps
# )

metric = paddle.metric.Accuracy()

# model.to('cuda')

global_step = 0
for epoch in range(1, epochs + 1):
    correct = 0
    data_num = 0
    batch_id = 0
    model.train()
    # adv_modules = hook_sift_layer(model,
    #                               hidden_size=model.config.hidden_size,
    #                               learning_rate=1e-4,
    #                               init_perturbation=1e-2)
    #
    # adv = AdversarialLearner(model, adv_modules)

    for data in tqdm(train_loader, desc=f'epoch {epoch} training'):
        input_ids = paddle.to_tensor(data[0])
        token_type_ids = paddle.to_tensor(data[1])
        labels = paddle.to_tensor(data[2])

        # input_ids = torch.tensor(data[0].numpy()).to('cuda')
        # token_type_ids = torch.tensor(data[1].numpy()).to('cuda')
        # labels = torch.tensor(data[2].numpy()).to('cuda')

        input_data = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
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

        # logits = outputs['logits']
        # loss = outputs['loss']
        logits = outputs[1]
        loss = outputs[0]
        # loss += adv.loss(logits, pert_logits_fn,
        #                  loss_fn="symmetric-kl", **input_data) * adv_weight
        # loss = loss / 2


        probs = F.softmax(logits, axis=-1)

        # probs = torch.softmax(logits, dim=-1)

        correct = metric.compute(paddle.to_tensor(probs.cpu().detach().numpy()),
                                 paddle.to_tensor(labels.cpu().detach().numpy()))
        metric.update(correct)
        acc = metric.accumulate()
        # probs = F.softmax(logits, axis=1)
        # correct += np.sum((probs.argmax(axis=1) == labels).numpy() != 0)
        # data_num += labels.shape[0]
        # acc = correct / data_num

        global_step += 1
        if global_step % 100 == 0:
            logger.info("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (
                global_step, epoch, batch_id, loss, acc))

        loss.backward()

        # optimizer.step()
        # scheduler.step()
        # optimizer.zero_grad()

        optimizer.step()
        lr_scheduler.step()

        optimizer.clear_grad()

    with paddle.no_grad():
        model.eval()
        for data in tqdm(dev_loader, desc=f'epoch {epoch} dev'):
            input_ids = paddle.to_tensor(data[0])
            token_type_ids = paddle.to_tensor(data[1])
            labels = paddle.to_tensor(data[2])

            # input_ids = torch.tensor(data[0].numpy()).to('cuda')
            # token_type_ids = torch.tensor(data[1].numpy()).to('cuda')
            # labels = torch.tensor(data[2].numpy()).to('cuda')

            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids)

            # logits = outputs['logits']
            logits = outputs[0]

            probs = F.softmax(logits, axis=-1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

        logger.info('dev accuracy ', acc)
    state_dict = model.state_dict()
    paddle.save(state_dict, f"epoch_{epoch}_model.pdparams")
