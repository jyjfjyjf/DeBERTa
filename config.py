import os
root_path = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(root_path, 'lib\\deberta_large')
vocab_path = os.path.join(model_dir, 'spm.model')
model_path = os.path.join(model_dir, 'pytorch_model.bin')
paddle_model_path = os.path.join(model_dir, 'model_state.pdparams')
batch_size = 4
valid_batch_size = 32
max_length = 128
lr = 1e-5
adv_weight = 5  # 对抗学习权重
seed = 2022369
f16 = 'O2'
train_data_path = os.path.join(root_path, 'data\\MNLI\\train.tsv')
valid_data_path = os.path.join(root_path, 'data\\MNLI\\dev_matched.tsv')

mnli_label2id = {'contradiction': 0, 'entailment': 2, 'neutral': 1}
mnli_id2label = {0: 'contradiction', 2: 'entailment', 1: 'neutral'}
device = 'cuda'
log_path = os.path.join(root_path, 'log\\log.log')

