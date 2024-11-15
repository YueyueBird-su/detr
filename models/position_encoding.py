# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor  # 从 util.misc 导入 NestedTensor 类


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()  # 调用父类的初始化方法
        self.num_pos_feats = num_pos_feats  # 设置位置特征的数量
        self.temperature = temperature  # 设置温度参数
        self.normalize = normalize  # 设置是否归一化
        if scale is not None and normalize is False:
            raise ValueError(
                "normalize should be True if scale is passed"
            )  # 如果传递了 scale 参数但未归一化，抛出异常
        if scale is None:
            scale = 2 * math.pi  # 如果未传递 scale 参数，设置默认值为 2π
        self.scale = scale  # 设置 scale 参数

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # 获取张量
        mask = tensor_list.mask  # 获取掩码
        assert mask is not None  # 确保掩码不为空
        not_mask = ~mask  # 取反掩码
        # 沿 x、y 轴累加，结果依旧是[b, h, w]
        """
        not_mask = [[[1, 1, 1, 1],          y_embed = [[[1, 1, 1, 1],
                    [1, 1, 1, 1],     ->                [2, 2, 2, 2],
                    [1, 1, 1, 1]]]                      [3, 3, 3, 3]]]
                                            x_embed = [[[1, 2, 3, 4],
                                      ->                [1, 2, 3, 4],
                                                        [1, 2, 3, 4]]]
        """
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 沿 x 轴累加
        if self.normalize:
            eps = 1e-6  # 设置一个小值以避免除零错误
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 归一化 y 轴嵌入
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 归一化 x 轴嵌入

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )  # 创建一个包含位置特征数量的张量
        dim_t = self.temperature ** (
            2 * (dim_t // 2) / self.num_pos_feats
        )  # 计算位置编码的温度参数

        pos_x = x_embed[:, :, :, None] / dim_t  # 计算 x 轴位置编码 [b, h, w] -> [b, h, w, feat_dim]
        pos_y = y_embed[:, :, :, None] / dim_t  # 计算 y 轴位置编码
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(
            3
        )  # 将 x 轴位置编码的正弦和余弦部分堆叠并展平 [b, c, h, w//2, 2] -> [b, c, h, w]
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(
            3
        )  # 将 y 轴位置编码的正弦和余弦部分堆叠并展平 [b, h, w, c//2, 2] -> [b, h, w, c]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(
            0, 3, 1, 2
        )  # 将 x 和 y 轴位置编码拼接并调整维度顺序 [b, h, w, 2c] -> [b, 2c, h, 2]
        return pos  # 返回位置编码


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()  # 调用父类的初始化方法
        self.row_embed = nn.Embedding(50, num_pos_feats)  # 创建行嵌入
        self.col_embed = nn.Embedding(50, num_pos_feats)  # 创建列嵌入
        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)  # 初始化行嵌入的权重
        nn.init.uniform_(self.col_embed.weight)  # 初始化列嵌入的权重

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # 获取张量
        h, w = x.shape[-2:]  # 获取张量的高度和宽度
        i = torch.arange(w, device=x.device)  # 创建一个包含宽度范围的张量
        j = torch.arange(h, device=x.device)  # 创建一个包含高度范围的张量
        x_emb = self.col_embed(i)  # 获取列嵌入
        y_emb = self.row_embed(j)  # 获取行嵌入
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),  # 将列嵌入扩展并重复
                    y_emb.unsqueeze(1).repeat(1, w, 1),  # 将行嵌入扩展并重复
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )  # 拼接嵌入并调整维度顺序
        return pos  # 返回位置编码


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2  # 计算位置编码的维度大小
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(
            N_steps, normalize=True
        )  # 创建正弦位置编码
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)  # 创建学习位置编码
    else:
        raise ValueError(
            f"not supported {args.position_embedding}"
        )  # 如果位置编码类型不支持，抛出异常

    return position_embedding  # 返回位置编码
