import numpy as np
import torch
from reprod_log import ReprodLogger
from transformers.models.deberta_v2 import DebertaV2ForSequenceClassification
from config import model_path, model_dir

if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    model = DebertaV2ForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    classifier_weights = torch.load(model_path)
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()

    # read or gen fake data
    x = np.array([[4, 1, 4, 7, 3, 9, 9, 1, 1, 5],
                  [8, 6, 5, 8, 2, 7, 3, 1, 1, 6]])
    input_torch = torch.tensor(x, dtype=torch.long)
    # forward
    out = model(input_torch)[0]
    #
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_torch.npy")
