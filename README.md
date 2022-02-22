# DeBERTa
DeBERTa
## 1 简介

**本项目基于PaddlePaddle复现的Deberta，完成情况如下:**

- 在tweetqa和xsum数据集上均达到论文精度
- 我们复现的ByT5是基于paddlenlp
- 我们提供aistudio notebook, 帮助您快速验证模型

**项目参考：**
- [deberta_v2](https://github.com/huggingface/transformers/tree/master/src/transformers/models/deberta_v2)

## 2 复现精度
>#### 在MNLI-m/mm数据集的测试效果如下表。没有达到要求

|      模型       |opt| 数据集  | Acc  | Acc(原论文) |
|:-------------:| :---: |:----:|:----:|:--------:|
| deberta-large |AdamW| MNLI | 33.3 |   91.3/91.1   |

>复现代码训练日志：
[复现代码训练日志](https://github.com/jyjfjyjf/DeBERTa/blob/master/log/pt_log.txt)


## 3 数据集
我们主要复现MNLI-m/mm数据集的精度, 数据集，

tweetqa数据集可以前往此处下载:
[地址](https://gluebenchmark.com/tasks)


## 4环境依赖
运行以下命令即可配置环境(由于nltk在源码中复制，所以可以不安装)
```bash
pip install paddlepaddle-gpu
pip install sentencepiece
```

## 5 快速开始

1. 将转换后的模型放到lib/deberta_large/下面 
>转换之后的模型链接为https://aistudio.baidu.com/aistudio/datasetdetail/125983
2. 调整xsum数据集目录：MNLI数据集较大，下载数据集到指定目录data/MNLI/
3. 微调和验证：
   以下是训练以及验证MNLI的train_eval.py
```
python tools/train_eval.py
```

## 6 主要代码路径
1. tokenizer代码
   byt5tokenizer：paddle_deberta/paddlenlp/tokenization_deberta.py
2. 数据集加载：
   tools/my_datasets.py
3. tools目录中包含微调任务的训练与测试脚本，train_eval.py执行训练验证脚本

