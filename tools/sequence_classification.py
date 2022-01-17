from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle.nn import CrossEntropyLoss
from tools.nnmodule import NNModule
from tools.debera import *
from tools.pooling import PoolConfig, ContextPooler
from paddle.fluid.layers import reshape

__all__ = ['SequenceClassificationModel']


class SequenceClassificationModel(NNModule):
    def __init__(self, config, num_labels=3, drop_out=None, pre_trained=None):
        super().__init__(config)
        self.num_labels = num_labels
        # self._register_load_state_dict_pre_hook(self._pre_load_hook)
        self.deberta = DeBERTa(config, pre_trained=pre_trained)
        if pre_trained is not None:
            self.config = self.deberta.config
        else:
            self.config = config
        pool_config = PoolConfig(self.config)
        output_dim = self.deberta.config.hidden_size
        self.pooler = ContextPooler(pool_config)
        output_dim = self.pooler.output_dim()

        self.classifier = paddle.nn.Linear(output_dim, num_labels)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        # self.dropout = paddle.nn.Dropout(drop_out)
        self.apply(self.init_weights)
        self.deberta.apply_state()

    def forward(self, input_ids, type_ids=None, input_mask=None, labels=None, position_ids=None, **kwargs):
        outputs = self.deberta(input_ids, attention_mask=input_mask, token_type_ids=type_ids,
                               position_ids=position_ids, output_all_encoded_layers=True)
        encoder_layers = outputs['hidden_states']

        pooled_output = self.pooler(encoder_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = 0
        if labels is not None:
            if self.num_labels == 1:
                # regression task
                loss_fn = paddle.nn.MSELoss()
                logits = reshape(logits, (-1)).astype(labels.dtype)
                loss = loss_fn(logits, reshape(labels, (-1)))
            elif labels.dim() == 1 or labels.size(-1) == 1:
                label_index = (labels >= 0).nonzero()
                labels = labels.astype('int64')
                if label_index.shape[0] > 0:
                    # labeled_logits = paddle.gather(logits, 0, label_index.expand(label_index.shape[0], logits.shape[1]))
                    # labels = paddle.gather(labels, 0, reshape(label_index, (-1)))

                    logits = logits.reshape((-1, self.num_labels))
                    labels = labels.reshape([-1])

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                else:
                    # loss = paddle.to_tensor(0).to(logits)
                    loss = paddle.to_tensor(0)
            else:
                log_softmax = paddle.nn.LogSoftmax(-1)
                label_confidence = 1
                loss = -((log_softmax(logits) * labels).sum(-1) * label_confidence).mean()

        return {
            'logits': logits,
            'loss': loss
        }
        # return (logits, loss)

    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
                       missing_keys, unexpected_keys, error_msgs):
        new_state = dict()
        bert_prefix = prefix + 'bert.'
        deberta_prefix = prefix + 'deberta.'
        for k in list(state_dict.keys()):
            if k.startswith(bert_prefix):
                nk = deberta_prefix + k[len(bert_prefix):]
                value = state_dict[k]
                del state_dict[k]
                state_dict[nk] = value
