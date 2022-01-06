import numpy as np
import torch
import torch.nn as nn
from reprod_log import ReprodLogger
from transformers import DebertaV2ForSequenceClassification, DebertaV2Config

from config import model_dir, model_path

if __name__ == "__main__":

    # def logger
    reprod_logger = ReprodLogger()

    criterion = nn.CrossEntropyLoss()

    config = DebertaV2Config.from_pretrained(model_dir, num_labels=3)
    model = DebertaV2ForSequenceClassification.from_pretrained(model_dir, config=config)
    classifier_weights = torch.load(model_path)
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()
    # read or gen fake data
    x = np.array([[4, 1, 4, 7, 3, 9, 9, 1, 1, 5],
                  [8, 6, 5, 8, 2, 7, 3, 1, 1, 6]])
    y = np.array([2, 1])
    fake_data = torch.from_numpy(x).to(torch.long)
    fake_label = torch.from_numpy(y).to(torch.long)

    # forward
    out = model(fake_data)[0]

    loss = criterion(out, fake_label)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")
