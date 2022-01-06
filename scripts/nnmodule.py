import copy
from scripts.deberta_config import DebertaV2Config
import paddle
from scripts.cache_utils import load_model_state

from scripts.logger_utils import get_logger

logger = get_logger()


class NNModule(paddle.nn.Layer):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def init_weights(self, module):

        if isinstance(module, paddle.nn.Linear):
            module.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=module.weight.shape
                )
            )
            if module.bias is not None:
                module.bias.set_value(paddle.zeros_like(module.bias))
        elif isinstance(module, paddle.nn.Embedding):
            module.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=module.weight.shape
                )
            )
            if module._padding_idx is not None:
                module.weight[module._padding_idx].set_value(
                    paddle.zeros_like(module.weight[module._padding_idx])
                )
        elif isinstance(module, paddle.nn.LayerNorm):
            module.bias.set_value(paddle.zeros_like(module.bias))
            module.weight.set_value(paddle.ones_like(module.weight))

    @classmethod
    def load_model(cls, model_path, model_config=None, tag=None, no_cache=False, cache_dir=None, *inputs, **kwargs):

        # Load config
        if model_config:
            config = DebertaV2Config.from_pretrain(model_config)
        else:
            config = None
        model_config = None
        model_state = None
        if (model_path is not None) and (model_path.strip() == '-' or model_path.strip() == ''):
            model_path = None

        try:
            model_state, model_config = load_model_state(model_path, tag=tag, no_cache=no_cache, cache_dir=cache_dir)
        except Exception as exp:
            raise Exception(f'Failed to get model {model_path}. Exception: {exp}')

        if config is not None and model_config is not None:
            for k in config.__dict__:
                if k not in ['hidden_size',
                             'intermediate_size',
                             'num_attention_heads',
                             'num_hidden_layers',
                             'vocab_size',
                             'max_position_embeddings'] or (k not in model_config.__dict__) or (
                        model_config.__dict__[k] < 0):
                    model_config.__dict__[k] = config.__dict__[k]
        if model_config is not None:
            config = copy.copy(model_config)
        vocab_size = config.vocab_size
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if not model_state:
            return model
        # copy state_dict so _load_from_state_dict can modify it
        state_dict = model_state.copy()

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        logger.warning(f'Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}, error_msgs: {error_msgs}')
        return model
