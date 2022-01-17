import math

import torch.nn.functional
from packaging import version
import paddle
from paddle.nn import LayerNorm
from paddle.autograd import PyLayerContext
from paddle.fluid import op


__all__ = ['StableDropout', 'MaskedLayerNorm', 'ACT2FN', 'LayerNorm', 'XSoftmax']


# class DropoutContext(object):
#     def __init__(self):
#         self.dropout = 0
#         self.mask = None
#         self.scale = 1
#         self.reuse_mask = True


class XSoftmax(paddle.nn.Layer):

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def forward(self, inputs, mask):

        stop_gradient = inputs.stop_gradient
        rmask = paddle.logical_not(mask, 'bool')
        # tmp_mask = paddle.full(inputs.shape, float("-inf"), inputs.dtype)
        # output = paddle.where(rmask, tmp_mask, inputs)
        output = paddle.nn.functional.softmax(inputs, self.axis)
        tmp_mask = paddle.full(output.shape, 0.0, output.dtype)
        output = paddle.where(rmask, tmp_mask, output)
        output.stop_gradient = stop_gradient
        return output


def get_mask(input, local_context):
    if not isinstance(local_context, PyLayerContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - paddle.bernoulli(paddle.rand(input.shape))).astype('bool')

    if isinstance(local_context, PyLayerContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            stop_gradient = input.stop_gradient
            # 使用paddle实现torch的masked_fill_
            tmp_mask = paddle.full(input.shape, 0, input.dtype)
            input = paddle.where(mask, tmp_mask, input)
            input.stop_gradient = stop_gradient

            return input * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            mask, = ctx.saved_tensor()

            stop_gradient = grad_output.stop_gradient
            tmp_mask = paddle.full(grad_output.shape, 0, grad_output.dtype)
            grad_output = paddle.where(mask, tmp_mask, grad_output)
            grad_output.stop_gradient = stop_gradient
            grad_output = grad_output * ctx.scale

            return grad_output
        else:
            return grad_output


class StableDropout(paddle.nn.Layer):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):

        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(PyLayerContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


def MaskedLayerNorm(layerNorm, input, mask=None):

    output = layerNorm(input)
    if mask is None:
        return output
    if mask.dim() != input.dim():
        if mask.dim() == 4:
            mask = mask.squeeze(1).squeeze(1)
        mask = mask.unsqueeze(2)
    mask = mask.astype(output.dtype)
    return output * mask


# def gelu(x):
#
#     return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))


# def swish(x):
#     sigmoid = paddle.nn.Sigmoid()
#     return x * sigmoid(x)


def linear_act(x):
    return x


ACT2FN = {"gelu": paddle.nn.functional.gelu, "relu": paddle.nn.functional.relu, "swish": paddle.nn.functional.swish,
          "tanh": paddle.tanh, "linear": linear_act,
          'sigmoid': paddle.nn.functional.sigmoid}
