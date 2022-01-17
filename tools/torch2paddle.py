import torch
import paddle
from config import model_path, paddle_model_path

"""模型torch转paddle"""
torch_dict = torch.load(model_path)

paddle_dict = {}

fc_names = ['self.query', 'self.key', 'self.value', 'dense', 'self.pos_query_proj',
            'self.pos_key_proj', 'classifier.weight', 'classifier.bias']

for key in torch_dict:
    weight = torch_dict[key].cpu().numpy()
    flag = [i in key for i in fc_names]
    if any(flag):
        print('weight {} need to be trans'.format(key))
        weight = weight.transpose()
    paddle_dict[key] = weight.astype('float32')

paddle.save(paddle_dict, paddle_model_path)
