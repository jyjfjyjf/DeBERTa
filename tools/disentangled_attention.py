import math

import numpy as np
import paddle
from paddle import nn

from tools.da_utils import build_relative_position
from paddle.fluid.layers import reshape, transpose, clip
from tools.logger_utils import get_logger
from tools.ops import StableDropout, XSoftmax

logger = get_logger()


class DisentangledSelfAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.attention_head_size = getattr(config, 'attention_head_size', _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.share_att_key = getattr(config, 'share_att_key', False)
        self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'c2p').lower().split('|')]  # c2p|p2c
        self.relative_attention = getattr(config, 'relative_attention', False)

        if self.relative_attention:
            self.position_buckets = getattr(config, 'position_buckets', -1)
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
                # For backward compitable

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)
            # self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size)
                if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.softmax = XSoftmax(axis=-1)
        # self.softmax = nn.Softmax()
        self.dropout = StableDropout(config.attention_probs_dropout_prob)
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # self._register_load_state_dict_pre_hook(self._pre_load_hook)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = tuple(x.shape[:-1]) + (attention_heads, -1)
        x = reshape(x, new_x_shape)

        return reshape(transpose(x, (0, 2, 1, 3)), (-1, x.shape[1], x.shape[-1]))

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None,
                rel_embeddings=None):

        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states),
                                                self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        from reprod_log import ReprodLogger
        reprod_logger = ReprodLogger()
        reprod_logger.add("logits", self.query_proj.weight.T.cpu().detach().numpy())
        reprod_logger.save("forward_paddle.npy")

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.shape[-1] * scale_factor)
        attention_scores = paddle.bmm(query_layer, transpose(key_layer, (0, 2, 1))) / scale

        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings,
                                                       scale_factor)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        attention_scores = attention_scores

        # attention_scores = (attention_scores - attention_scores.max(axis=-1, keepdim=True).detach()).astype(
        #     hidden_states.dtype)

        attention_scores = reshape(attention_scores, (-1, self.num_attention_heads, attention_scores.shape[-2],
                                                      attention_scores.shape[-1]))

        # bxhxlxd
        _attention_probs = self.softmax(attention_scores, attention_mask)
        # _attention_probs = self.softmax(attention_scores)

        attention_probs = self.dropout(_attention_probs)

        context_layer = paddle.bmm(reshape(attention_probs, (-1, attention_probs.shape[-2], attention_probs.shape[-1])),
                                   value_layer)

        context_layer = transpose(reshape(context_layer, (-1, self.num_attention_heads, context_layer.shape[-2],
                                                          context_layer.shape[-1])), (0, 2, 1, 3))

        new_context_layer_shape = tuple(context_layer.shape[:-2]) + (-1,)
        context_layer = reshape(context_layer, new_context_layer_shape)

        return {
            'hidden_states': context_layer,
            'attention_probs': attention_probs,
            'attention_logits': attention_scores
        }

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.shape[-2]
            relative_pos = build_relative_position(q, key_layer.shape[-2], bucket_size=self.position_buckets,
                                                   max_position=self.max_relative_positions)

        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.astype('int64')

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(
            0)  # .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)

        if self.share_att_key:
            # pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads) \
            #     .repeat(query_layer.shape[0] // self.num_attention_heads, 1, 1)  # .split(self.all_head_size, dim=-1)

            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)
            pos_query_layer = paddle.tile(pos_query_layer, (query_layer.shape[0] // self.num_attention_heads, 1, 1))

            # pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads) \
            #     .repeat(query_layer.shape[0] // self.num_attention_heads, 1, 1)  # .split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)
            pos_key_layer = paddle.tile(pos_key_layer, (query_layer.shape[0] // self.num_attention_heads, 1, 1))

        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                # pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads) \
                #     .repeat(query_layer.shape[0] // self.num_attention_heads, 1, 1)  # .split(self.all_head_size,
                # # dim=-1)

                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)
                pos_key_layer = paddle.tile(pos_key_layer, (query_layer.shape[0] // self.num_attention_heads, 1, 1))
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                # pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings),
                #                                             self.num_attention_heads) \
                #     .repeat(query_layer.shape[0] // self.num_attention_heads, 1, 1)  # .split(self.all_head_size,
                # # dim=-1)

                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings),
                                                            self.num_attention_heads)
                pos_query_layer = paddle.tile(pos_query_layer, (query_layer.shape[0] // self.num_attention_heads, 1, 1))

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            scale = math.sqrt(pos_key_layer.shape[-1] * scale_factor)
            c2p_att = paddle.bmm(query_layer, transpose(pos_key_layer, (0, 2, 1)).astype(query_layer.dtype))
            c2p_pos = clip(relative_pos + att_span, 0, att_span * 2 - 1)
            # index = c2p_pos.squeeze(0).expand(
            #     [query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]])
            #
            # b, c, _ = tuple(c2p_att.shape)
            #
            # c2p_att = concat(
            #     [
            #         reshape(
            #             concat([
            #                 reshape(gather(c2p_att[i, j, :], index[i, j, :]), [1, -1])
            #                 for j in range(c)], axis=0),
            #             [1, c, -1]) for i in range(b)], axis=0
            # )

            b, c, _ = tuple(c2p_att.shape)

            c2p_att = paddle.index_sample(c2p_att.flatten(0, 1),
                                          c2p_pos.squeeze(0).expand([query_layer.shape[0], query_layer.shape[1],
                                                                     relative_pos.shape[-1]]).flatten(0, 1)).reshape(
                (b, c, -1))

            # c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand(
            #     [query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]]))
            score += c2p_att / scale

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = math.sqrt(pos_query_layer.shape[-1] * scale_factor)
            if key_layer.shape[-2] != query_layer.shape[-2]:
                r_pos = build_relative_position(key_layer.shape[-2], key_layer.shape[-2],
                                                bucket_size=self.position_buckets,
                                                max_position=self.max_relative_positions)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = clip(-r_pos + att_span, 0, att_span * 2 - 1)
            if query_layer.shape[-2] != key_layer.shape[-2]:
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = paddle.bmm(key_layer, transpose(pos_query_layer, (0, 2, 1)).astype(key_layer.dtype))

            # index = p2c_pos.squeeze(0).expand(
            #     [query_layer.shape[0], key_layer.shape[-2], key_layer.shape[-2]])

            b, c, _ = tuple(p2c_att.shape)

            p2c_att = paddle.index_sample(p2c_att.flatten(0, 1),
                                          p2c_pos.squeeze(0).
                                          expand([query_layer.shape[0],
                                                  key_layer.shape[-2],
                                                  key_layer.shape[-2]]).flatten(0, 1)).reshape((b, c, -1))

            p2c_att = transpose(p2c_att, (0, 2, 1))

            # p2c_att = concat(
            #     [
            #         reshape(
            #             concat([
            #                 reshape(gather(p2c_att[i, j, :], index[i, j, :]), [1, -1])
            #                 for j in range(c)], axis=0),
            #             [1, c, -1]) for i in range(b)], axis=0
            # )
            # p2c_att = transpose(p2c_att, (0, 2, 1))

            # p2c_att = paddle.gather(p2c_att, axis=-1, index=p2c_pos.squeeze(0).expand(
            #     [query_layer.shape[0], key_layer.shape[-2], key_layer.shape[-2]])).transpose(-1, -2)
            if query_layer.shape[-2] != key_layer.shape[-2]:
                b, c, _ = tuple(p2c_att.shape)

                p2c_att = paddle.index_sample(p2c_att.flatten(0, 1),
                                              p2c_pos.squeeze(0).expand([query_layer.shape[0], key_layer.shape[-2],
                                                                         key_layer.shape[-2]]).flatten(0, 1)).reshape(
                    (b, c, -1))

                # index = pos_index.expand(
                #     p2c_att.shape[:2] + (pos_index.shape[-2], key_layer.shape[-2]))
                #
                # p2c_att = concat(
                #     [
                #         reshape(
                #             concat([
                #                 reshape(gather(p2c_att[i, j, :], index[i, j, :]), [1, -1])
                #                 for j in range(c)], axis=0),
                #             [1, c, -1]) for i in range(b)], axis=0
                # )

                # p2c_att = paddle.gather(p2c_att, axis=-2, index=pos_index.expand(
                #     p2c_att.shape[:2] + (pos_index.shape[-2], key_layer.shape[-2])))
            score += p2c_att / scale

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:, :, att_span:, :]
            p2p_att = paddle.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.shape[:2] + p2p_att.shape[2:])

            # p2p_att = paddle.gather(p2p_att, axis=-1, index=c2p_pos.expand(
            #     [query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], relative_pos.shape[-1]]))
            b, c, _ = tuple(p2p_att.shape)
            p2p_att = paddle.index_sample(p2p_att.flatten(0, 1),
                                          c2p_pos.squeeze(0).
                                          expand([query_layer.shape[1],
                                                  query_layer.shape[2],
                                                  relative_pos.shape[-1]]).flatten(0, 1)).reshape((b, c, -1))

            p2p_att = transpose(p2p_att, (0, 2, 1))

            score += p2p_att

        return score

    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
                       missing_keys, unexpected_keys, error_msgs):
        self_state = self.state_dict()
        if ((prefix + 'query_proj.weight') not in state_dict) and ((prefix + 'in_proj.weight') in state_dict):
            v1_proj = state_dict[prefix + 'in_proj.weight']
            v1_proj = v1_proj.unsqueeze(0).reshape(self.num_attention_heads, -1, v1_proj.size(-1))
            q, k, v = v1_proj.chunk(3, dim=1)
            state_dict[prefix + 'query_proj.weight'] = q.reshape(-1, v1_proj.size(-1))
            state_dict[prefix + 'key_proj.weight'] = k.reshape(-1, v1_proj.size(-1))
            state_dict[prefix + 'key_proj.bias'] = self_state['key_proj.bias']
            state_dict[prefix + 'value_proj.weight'] = v.reshape(-1, v1_proj.size(-1))
            v1_query_bias = state_dict[prefix + 'q_bias']
            state_dict[prefix + 'query_proj.bias'] = v1_query_bias
            v1_value_bias = state_dict[prefix + 'v_bias']
            state_dict[prefix + 'value_proj.bias'] = v1_value_bias

            v1_pos_key_proj = state_dict[prefix + 'pos_proj.weight']
            state_dict[prefix + 'pos_key_proj.weight'] = v1_pos_key_proj
            v1_pos_query_proj = state_dict[prefix + 'pos_q_proj.weight']
            state_dict[prefix + 'pos_query_proj.weight'] = v1_pos_query_proj
            v1_pos_query_proj_bias = state_dict[prefix + 'pos_q_proj.bias']
            state_dict[prefix + 'pos_query_proj.bias'] = v1_pos_query_proj_bias
            state_dict[prefix + 'pos_key_proj.bias'] = self_state['pos_key_proj.bias']

            del state_dict[prefix + 'in_proj.weight']
            del state_dict[prefix + 'q_bias']
            del state_dict[prefix + 'v_bias']
            del state_dict[prefix + 'pos_proj.weight']
            del state_dict[prefix + 'pos_q_proj.weight']
            del state_dict[prefix + 'pos_q_proj.bias']
