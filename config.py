import os
root_path = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(root_path, 'lib\\deberta-v2-xlarge-mnli')
vocab_path = os.path.join(model_dir, 'spm.model')
model_path = os.path.join(model_dir, 'pytorch_model.bin')
paddle_model_path = os.path.join(model_dir, 'model_state.pdparams')
batch_size = 1
valid_batch_size = 4
max_length = 128
lr = 4e-6
adv_weight = 5  # 对抗学习权重
seed = 2022369


