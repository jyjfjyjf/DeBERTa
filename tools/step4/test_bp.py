import numpy as np
import paddle
import torch

from config import model_dir, model_path, paddle_model_path
from tools.deberta_config import DebertaV2Config
from tools.sequence_classification import SequenceClassificationModel
from reprod_log import ReprodLogger
from transformers import AdamW
from transformers import DebertaV2ForSequenceClassification


def pd_train_some_iters(model,
                        criterion,
                        optimizer,
                        fake_data,
                        fake_label,
                        max_iter=2):
    config = DebertaV2Config.from_pretrain(model_dir)
    config.use_return_dict = False
    config.output_attentions = False
    config.output_hidden_states = False

    model = SequenceClassificationModel(config)
    classifier_weights = paddle.load(paddle_model_path)
    model.load_dict(classifier_weights)
    model.eval()
    criterion = paddle.nn.CrossEntropy()
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=3e-5,
        parameters=model.parameters(),
        weight_decay=1e-2,
        epsilon=1e-6,
        apply_decay_param_fun=lambda x: x in decay_params, )
    loss_list = []
    for idx in range(max_iter):
        input_ids = paddle.to_tensor(fake_data)
        labels = paddle.to_tensor(fake_label)

        output = model(input_ids)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss)
    return loss_list


def hf_train_some_iters(fake_data, fake_label, max_iter=2):
    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_dir, num_labels=3)
    classifier_weights = torch.load(
        model_path)
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

    loss_list = []
    for idx in range(max_iter):
        input_ids = torch.from_numpy(fake_data).to(torch.long)
        labels = torch.from_numpy(fake_label).to(torch.long)

        output = model(input_ids)[0]
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss)
    return loss_list


if __name__ == "__main__":
    print("Start training")
    paddle.set_device("cpu")
    fake_data = np.array([[4, 1, 4, 7, 3, 9, 9, 1, 1, 5],
                          [8, 6, 5, 8, 2, 7, 3, 1, 1, 6]])
    fake_label = np.array([2, 1])
    hf_reprod_logger = ReprodLogger()
    hf_loss_list = hf_train_some_iters(fake_data, fake_label, 10)
    for idx, loss in enumerate(hf_loss_list):
        hf_reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    hf_reprod_logger.save("bp_align_torch.npy")

    pd_reprod_logger = ReprodLogger()
    pd_loss_list = hf_train_some_iters(fake_data, fake_label, 10)
    for idx, loss in enumerate(pd_loss_list):
        pd_reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    pd_reprod_logger.save("bp_align_paddle.npy")
