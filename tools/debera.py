import copy
import paddle

from tools.bert import *
from tools.cache_utils import load_model_state

__all__ = ['DeBERTa']


class DeBERTa(paddle.nn.Layer):

    def __init__(self, config=None, pre_trained=None):
        super().__init__()
        state = None
        if pre_trained is not None:
            state, model_config = load_model_state(pre_trained)
            if config is not None and model_config is not None:
                for k in config.__dict__:
                    if k not in ['hidden_size',
                                 'intermediate_size',
                                 'num_attention_heads',
                                 'num_hidden_layers',
                                 'vocab_size',
                                 'max_position_embeddings']:
                        model_config.__dict__[k] = config.__dict__[k]
            config = copy.copy(model_config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.config = config
        self.pre_trained = pre_trained
        self.apply_state(state)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_all_encoded_layers=True,
                position_ids=None, return_att=False):

        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids)

        ebd_output = self.embeddings(input_ids.astype('int64'), token_type_ids.astype('int64'), position_ids,
                                     attention_mask)
        embedding_output = ebd_output['embeddings']

        encoder_output = self.encoder(embedding_output,
                                      attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers, return_att=return_att)

        encoder_output.update(ebd_output)

        return encoder_output

    def apply_state(self, state=None):
        if self.pre_trained is None and state is None:
            return
        if state is None:
            state, config = load_model_state(self.pre_trained)
            self.config = config

        # prefix = ''
        # for k in state:
        #     if 'embeddings.' in k:
        #         if not k.startswith('embeddings.'):
        #             prefix = k[:k.index('embeddings.')]
        #         break
        #
        # missing_keys = []
        # unexpected_keys = []
        # error_msgs = []
        # self._load_from_state_dict(state, prefix=prefix, local_metadata=None, strict=True, missing_keys=missing_keys,
        #                            unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        self.set_state_dict(state)
