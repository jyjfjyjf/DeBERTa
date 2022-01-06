"""
Pooling functions
"""

from paddle import nn
import copy
import json
from scripts.bert import ACT2FN
from scripts.ops import StableDropout
from scripts.deberta_config import AbsModelConfig

__all__ = ['PoolConfig', 'ContextPooler']


class PoolConfig(AbsModelConfig):

    def __init__(self, config=None):
        """Constructs PoolConfig.

        Args:
           `config`: the config of the model. The field of pool config will be initalized with the 'pooling' field in model config.
        """

        self.hidden_size = 768
        self.dropout = 0
        self.hidden_act = 'gelu'
        if config:
            pool_config = getattr(config, 'pooling', config)
            if isinstance(pool_config, dict):
                pool_config = AbsModelConfig.from_dict(pool_config)
            self.hidden_size = getattr(pool_config, 'hidden_size', config.hidden_size)
            self.dropout = getattr(pool_config, 'dropout', 0)
            self.hidden_act = getattr(pool_config, 'hidden_act', 'gelu')


class ContextPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = StableDropout(config.dropout)
        # self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, hidden_states, mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)

        pooled_output = ACT2FN[self.config.hidden_act](pooled_output)
        return pooled_output

    def output_dim(self):
        return self.config.hidden_size
