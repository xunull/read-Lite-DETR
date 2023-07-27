# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from ..functions import ms_deform_attn_core_pytorch
from ..functions import ms_deform_attn_core_pytorch_key_aware
import torch.utils.checkpoint as checkpoint



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0

# 这里的代码有修改
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4,
                 # 以下参数是新增的
                 use_pytorch_version=False, value_proj_after=False, key_aware=True,
                 add=True, proj_key=True, deformable_use_checkpoint=False, same_loc=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.use_pytorch_version = use_pytorch_version
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.same_loc = same_loc

        if not same_loc:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)

        if not key_aware:
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        self.value_proj = nn.Linear(d_model, d_model)

        self.proj_key = proj_key
        if proj_key:
            self.key_proj = nn.Linear(d_model, d_model)
        else:
            self.key_proj = None
        # self.key_proj = None
        self.query_proj = nn.Linear(d_model, d_model)

        self.output_proj = nn.Linear(d_model, d_model)

        self.key_aware = key_aware
        self.add = add
        self.deformable_use_checkpoint = deformable_use_checkpoint

        print("use_pytorch_version key_aware, add， same_loc", use_pytorch_version, key_aware, add, same_loc)
        self.value_proj_after = value_proj_after

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.same_loc:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1,
                                                                                                               self.n_points,
                                                                                                               1)
            for i in range(self.n_points):
                grid_init[:, i, :] *= i + 1
        else:
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                                  self.n_levels,
                                                                                                                  self.n_points,
                                                                                                                  1)
            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if not self.key_aware:
            constant_(self.attention_weights.weight.data, 0.)
            constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)

        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
        if self.proj_key:
            xavier_uniform_(self.key_proj.weight.data)
            constant_(self.key_proj.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        参数与原来是相同的

        query是上一层的输出加上了位置编码
        :param query                       (N, Length_{query}, C)
        参考点位的坐标
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        encoder是上一层的输出，decoder使用的是encoder的输出 [bs, all hw, 256]
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        4个特征层的高宽
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        各个特征层的起始index的下标 如: [    0,  8056, 10070, 10583]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        # query 是src+pos，query 经过全连接网络 下面变成了attention_weights
        # input_flatten 是src，input_flatten 对应了V
        N, Len_q, _ = query.shape
        # [bs,all hw,256] input_flatten在encoder和decoder阶段，都是all hw的那个维度大小，因为在decoder阶段，他就是encoder的memory
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # 这块这样写，value也必须在这个地方赋值，也必须走这个判断
        if not self.value_proj_after:
            # 图二中左下的全连接Linear
            value = self.value_proj(input_flatten)

        # if key is None:
        # KDA需要使用key，key和value的来源都是从input_flatten开始
        key = input_flatten

        # 如果key需要经过单独的全连接网络，那么就使用单独的全连接网络
        # 否则就是直接使用value就行，key value是相同的，value就是input_flatten经过全连接网络之后的
        if self.proj_key:
            key = self.key_proj(key)
        else:
            key = value

        # value = input_flatten
        if input_padding_mask is not None:
            # 在mask的地方填充0 [bs, all hw,256]
            value = value.masked_fill(input_padding_mask[..., None], float(0))
            # Deformable DETR 没有处理key，这里需要对key进行同样的处理
            key = key.masked_fill(input_padding_mask[..., None], float(0))

        # key与value进行同样的处理
        # 分成多头，拆分的是最后的256 [bs,all hw,256] -> [bs,all hw, 8, 32]
        key = key.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # 分成多头，拆分的是最后的256 [bs,all hw,256] -> [bs,all hw, 8, 32]
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        if not self.same_loc:
            # sampling_offsets 是一个全连接网络 [bs,x,8,4,4,2]
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points,
                                                                 2)
        else:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_points, 2)
            # 相同的location 这里进行repeat就可以
            sampling_offsets = sampling_offsets[:, :, :, None].repeat(1, 1, 1, self.n_levels, 1, 1)

        attention_weights = None

        # 是否使用 KDA
        # 如果不使用KDA, 这个地方就是正常的Deformable DETR的处理
        if not self.key_aware:
            # 这里就是Deformable DETR的正常处理，不使用key_aware
            attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads,
                                                                   self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels,
                                                                      self.n_points)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            # input_spatial_shapes 换位置，高宽 变成 宽高
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # reference_points  [bs,all hw,4,2] -> [bs,all hw,1,4,1,2]
            # sampling_offsets  [bs,all hw,8,4,4,2]
            # offset_normalizer [4,2] -> [1,1,1,4,1,2]
            # like (bs, hw,8,4,4,2)
            # 采样点加上偏移量 sampling_offsets / offset_normalizer 表示相对的偏移量
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:

            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        # 上面有个判断是 not self.key_aware 这两个判断正好是相反的
        # 使用KDA
        if self.key_aware:

            if not self.deformable_use_checkpoint:
                # ms_deform_attn_core_pytorch_key_aware Lite DETR新增的方法
                output = ms_deform_attn_core_pytorch_key_aware(
                    query, value, key,
                    # 这个参数实际没有被使用
                    input_padding_mask,
                    input_spatial_shapes, sampling_locations,
                    self.key_proj,self.value_proj, self.query_proj,
                    # 下面这两个参数实际没有被使用
                    attention_weights,
                    self.add
                )
            else:

                output = checkpoint.checkpoint(ms_deform_attn_core_pytorch_key_aware, query, value, key,
                                               input_padding_mask,
                                               input_spatial_shapes, sampling_locations, self.key_proj, self.value_proj,
                                               self.query_proj, attention_weights, self.add)

        elif self.use_pytorch_version:

            output = ms_deform_attn_core_pytorch(
                value, input_spatial_shapes, sampling_locations, attention_weights,
            )
        else:
            # Deformable DETR 正常的方法
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
                self.im2col_step)

        if self.value_proj_after:
            output = self.value_proj(output)

        # 图二中右下的全连接Linear
        output = self.output_proj(output)
        return output
