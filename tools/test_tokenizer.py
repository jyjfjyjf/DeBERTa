from paddle import nn
import numpy as np
import paddle

embedding = nn.Embedding(100, 200)

x = np.array([[4, 1, 4, 10000, 3, 9, 900, 1, 1, 5],
              [8, 6, 5, 8000, 2, 7, 3, 1, 1, 6]])
input_paddle = paddle.to_tensor(x, dtype='int64')

output = embedding(input_paddle)
print(output)
