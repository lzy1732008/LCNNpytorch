# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import utils
import unfold1d

class CNN(nn.Module):
    def __init__(self,args, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=False, bias=False):
        super().__init__()
        self.layers1 = nn.ModuleList([])
        self.layers1.extend([
            LightweightConv1dTBC(input_size, kernel_size=kernel_size, padding_l=padding_l, num_heads=num_heads,
                                 weight_dropout=weight_dropout, weight_softmax=weight_softmax, bias=bias)
            for i in range(args.con1_layers)
        ])

        self.layers2 = nn.ModuleList([])
        self.layers2.extend([
            LightweightConv1dTBC(input_size, kernel_size=kernel_size, padding_l=padding_l, num_heads=num_heads,
                                 weight_dropout=weight_dropout, weight_softmax=weight_softmax, bias=bias)
            for i in range(args.con2_layers)
        ])

        self.fc1 = nn.Linear(args.seq_length_1 + args.seq_length_2, args.output_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(args.output_dim, args.class_num)

    def forward(self, x1, x2):
        x1 = Variable(x1).float()
        x2 = Variable(x2).float()

        for layer in self.layers1:
            x1 = layer(x1)

        # x1 = self.conv1(x1)
        # x1 = self.conv1_2(x1)
        x1 = F.max_pool1d(x1, x1.shape[2])
        x1 = x1.view(-1, x1.shape[1])

        for layer in self.layers2:
            x2 = layer(x2)
        # x2 = self.conv2(x2)
        # x2 = self.conv2(x2)
        x2 = F.max_pool1d(x2, x2.shape[2])
        x2 = x2.view(-1, x2.shape[1])

        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        x = self.dropout(x)
        logit = self.fc2(x)

        return logit

class LightweightConv1dTBC(nn.Module):
    '''Lightweight Convolution assuming the input is TxBxC
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias

    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=False, bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads  #这个就是论文中的H的值
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.reset_parameters()

        self.onnx_trace = False

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, incremental_state=None, unfold=False):
        '''Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
        '''
        unfold = unfold or (incremental_state is not None)

        if unfold:
            output = self._forward_unfolded(x, incremental_state)
        else:
            output = self._forward_expanded(x, incremental_state)

        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)
        return output

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _forward_unfolded(self, x, incremental_state):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H  #这个是间隔
        assert R * H == C == self.input_size

        weight = self.weight.view(H, K)
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :, -self.kernel_size+1:])
            x_unfold = x_unfold.view(T*B*H, R, -1)
        else:
            # unfold the input: T x B x C --> T' x B x C x K
            x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)
            x_unfold = x_unfold.view(T*B*H, R, K)

        if self.weight_softmax:
            weight = utils.softmax(weight, dim=1, onnx_trace=self.onnx_trace).type_as(weight)

        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)

        weight = weight.view(1, H, K).expand(T*B, H, K).contiguous().view(T*B*H, K, 1)

        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        output = torch.bmm(x_unfold, weight) # T*B*H x R x 1
        output = output.view(T, B, C)

        return output

    def _forward_expanded(self, x, incremental_state):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        '''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight.view(H, K)
        if self.weight_softmax:
            weight = utils.softmax(weight, dim=1, onnx_trace=self.onnx_trace).type_as(weight)
        weight = weight.view(1, H, K).expand(T*B, H, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)

        x = x.view(T, B*H, R).transpose(0, 1)
        P = self.padding_l
        if K > T and P == K-1:
            weight = weight.narrow(2, K-T, T)
            K, P = T, T-1
        # turn the convolution filters into band matrices
        weight_expanded = weight.new_zeros(B*H, T, T+K-1, requires_grad=False)
        weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, P, T) #[30,128,128]
        weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training)

        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def extra_repr(self):
        s = '{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, bias={}'.format(
            self.input_size, self.kernel_size, self.padding_l,
            self.num_heads, self.weight_softmax, self.bias is not None
        )
        if self.weight_dropout > 0.:
            s += ', weight_dropout={}'.format(self.weight_dropout)
        return s
