import numpy as np
import paddle
import paddle.nn as nn
from scripts.deberta_config import DebertaV2Config
from config import paddle_model_path, model_dir
from scripts.sequence_classification import SequenceClassificationModel
from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("cpu")

    # def logger
    reprod_logger = ReprodLogger()
    config = DebertaV2Config.from_pretrain(model_dir)
    config.use_return_dict = False
    config.output_attentions = False
    config.output_hidden_states = False

    model = SequenceClassificationModel(config)
    classifier_weights = paddle.load(paddle_model_path)
    model.load_dict(classifier_weights)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # read or gen fake data
    x = np.array([[4, 1, 4, 7, 3, 9, 9, 1, 1, 5],
                  [8, 6, 5, 8, 2, 7, 3, 1, 1, 6]])
    y = np.array([2, 1])
    fake_data = paddle.to_tensor(x).astype('int64')
    fake_label = paddle.to_tensor(y).astype('int64')

    # forward
    out = model(fake_data)['logits']

    loss = criterion(out, fake_label)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")
