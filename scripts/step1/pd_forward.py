import numpy as np
import paddle
# 导入模型
from config import model_dir, paddle_model_path
from scripts.deberta_config import DebertaV2Config
from scripts.sequence_classification import SequenceClassificationModel
# 导入reprod_log中的ReprodLogger类
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()
# 组网并初始化
config = DebertaV2Config.from_pretrain(model_dir)
model = SequenceClassificationModel(config)

# 加载分类权重
checkpoint = paddle.load(paddle_model_path)
model.set_state_dict(checkpoint)
model.eval()
# 读入fake data并转换为tensor，这里也可以固定seed在线生成fake data
x = np.array([[4, 1, 4, 7, 3, 9, 9, 1, 1, 5],
              [8, 6, 5, 8, 2, 7, 3, 1, 1, 6]])
input_paddle = paddle.to_tensor(x, dtype='int64')
fake_data = paddle.to_tensor(x)
# 模型前向
out = model(fake_data)
# 保存前向结果，对于不同的任务，需要开发者添加。
reprod_logger.add("logits", out['logits'].detach().numpy())
reprod_logger.save("forward_paddle.npy")
