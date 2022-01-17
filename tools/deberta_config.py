import json
import os
import copy


class AbsModelConfig(object):
    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ModelConfig` from a Python dictionary of parameters."""
        config = cls()
        for key, value in json_object.items():
            if isinstance(value, dict):
                value = AbsModelConfig.from_dict(value)
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ModelConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        def _json_default(obj):
            if isinstance(obj, AbsModelConfig):
                return obj.__dict__
        return json.dumps(self.__dict__, indent=2, sort_keys=True, default=_json_default) + "\n"


class DebertaV2Config:

    def __init__(self,
                 vocab_size=128100,
                 hidden_size=1536,
                 num_hidden_layers=24,
                 num_attention_heads=24,
                 intermediate_size=6144,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=0,
                 initializer_range=0.02,
                 layer_norm_eps=1e-7,
                 relative_attention=False,
                 max_relative_positions=-1,
                 pad_token_id=0,
                 position_biased_input=True,
                 pos_att_type=None,
                 pooler_dropout=0,
                 pooler_hidden_act="gelu",
                 **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act

    @classmethod
    def from_pretrain(cls, config_dir):
        if os.path.isdir(config_dir):
            files = os.listdir(config_dir)
            config_path = None
            for file in files:
                if file == 'config.json':
                    config_path = os.path.join(config_dir, file)
            with open(config_path, 'r', encoding='utf-8') as f:
                json_config = json.load(f)
        else:
            with open(config_dir, 'r', encoding='utf-8') as f:
                json_config = json.load(f)
        rp_deberta_config = cls()
        rp_deberta_config.model_type = json_config['model_type']
        rp_deberta_config.attention_probs_dropout_prob = json_config['attention_probs_dropout_prob']
        rp_deberta_config.hidden_act = json_config['hidden_act']
        rp_deberta_config.hidden_dropout_prob = json_config['hidden_dropout_prob']
        rp_deberta_config.hidden_size = json_config['hidden_size']
        rp_deberta_config.initializer_range = json_config['initializer_range']
        rp_deberta_config.intermediate_size = json_config['intermediate_size']
        rp_deberta_config.max_position_embeddings = json_config['max_position_embeddings']
        rp_deberta_config.relative_attention = json_config['relative_attention']
        rp_deberta_config.position_buckets = json_config['position_buckets']
        rp_deberta_config.norm_rel_ebd = json_config['norm_rel_ebd']
        rp_deberta_config.share_att_key = json_config['share_att_key']
        rp_deberta_config.pos_att_type = json_config['pos_att_type']
        rp_deberta_config.layer_norm_eps = json_config['layer_norm_eps']
        rp_deberta_config.conv_kernel_size = json_config['conv_kernel_size']
        rp_deberta_config.conv_act = json_config['conv_act']
        rp_deberta_config.max_relative_positions = json_config['max_relative_positions']
        rp_deberta_config.position_biased_input = json_config['position_biased_input']
        rp_deberta_config.num_attention_heads = json_config['num_attention_heads']
        rp_deberta_config.attention_head_size = json_config['attention_head_size']
        rp_deberta_config.num_hidden_layers = json_config['num_hidden_layers']
        rp_deberta_config.type_vocab_size = json_config['type_vocab_size']
        rp_deberta_config.vocab_size = json_config['vocab_size']
        return rp_deberta_config


# if __name__ == '__main__':
#     deberta_config = DebertaV2Config.from_pretrain('..\\lib\\deberta-v2-xlarge')
#     print(deberta_config.position_buckets)

