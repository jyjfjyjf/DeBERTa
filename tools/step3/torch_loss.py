from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from reprod_log import ReprodLogger
from transformers import DebertaV2ForSequenceClassification, DebertaV2Config

from config import model_dir, model_path, adv_weight, lr
from tools.sift_torch import hook_sift_layer, AdversarialLearner

if __name__ == "__main__":

    # def logger
    reprod_logger = ReprodLogger()

    criterion = nn.CrossEntropyLoss()

    config = DebertaV2Config.from_pretrained(model_dir, num_labels=3)
    model = DebertaV2ForSequenceClassification.from_pretrained(model_dir, config=config)
    classifier_weights = torch.load(model_path)
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()
    adv_modules = hook_sift_layer(model,
                                  hidden_size=model.config.hidden_size,
                                  learning_rate=lr,
                                  init_perturbation=1e-2)

    adv = AdversarialLearner(model, adv_modules)
    # read or gen fake data
    x = np.array([[4, 1, 4, 7, 3, 9, 9, 1, 1, 5],
                  [8, 6, 5, 8, 2, 7, 3, 1, 1, 6]])
    y = np.array([2, 1])
    fake_data = torch.from_numpy(x).to(torch.long)
    fake_label = torch.from_numpy(y).to(torch.long)

    def pert_logits_fn(plf_model,
                       **plf_input_data):
        o = plf_model(**plf_input_data)
        plf_logits = o['logits']

        reprod_logger.add("loss", plf_logits.cpu().detach().numpy())
        reprod_logger.save("loss_torch.npy")

        if isinstance(plf_logits, Sequence):
            plf_logits = plf_logits[-1]
        return plf_logits

    # forward
    out = model(fake_data)[0]

    input_data = {'input_ids': fake_data}

    loss = criterion(out, fake_label)
    loss += adv.loss(out, pert_logits_fn,
                     loss_fn="symmetric-kl", **input_data) * adv_weight
    loss = loss / (1 + adv_weight)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")
