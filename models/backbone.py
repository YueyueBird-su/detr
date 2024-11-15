# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0] # 将mask插值到和x相同的大小
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # 得到backbone的输出
        out: List[NestedTensor] = []  # 初始化NestedTensor列表
        pos = []
        for name, x in xs.items():
            # 对每一个backbone的输出进行位置编码
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)  #
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks  # 根据参数 masks 判断是否返回中间层特征
    backbone = Backbone(
        args.backbone, train_backbone, return_interm_layers, args.dilation
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DETR training and evaluation script")
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--lr_backbone", default=1e-5, type=float, help="Learning rate for the backbone"
    )
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Whether the model generates masks (via segmentation) or not",
    )

    args = parser.parse_args()

    args.hidden_dim = 256
    args.position_embedding = "sine"
    args.masks = True
    model = build_backbone(args)
    print(model)

    # test
    input_tensor = torch.randn(1, 3, 1024, 1024)
    input_mask = torch.ones(1, 1024, 1024).bool()
    out = model(NestedTensor(input_tensor, input_mask))
    # size:
    # if args.masks:
    #   layer1: torch.Size([1, 256, 256, 256])   pos: torch.Size([1, 256, 256, 256])
    #   layer2: torch.Size([1, 512, 128, 128])   pos: torch.Size([1, 256, 128, 128])
    #   layer3: torch.Size([1, 1024, 64, 64])    pos: torch.Size([1, 256, 64, 64])
    #   layer4: torch.Size([1, 2048, 32, 32])    pos: torch.Size([1, 256, 32, 32])
    # else:
    #   layer4: torch.Size([1, 2048, 32, 32])    pos: torch.Size([1, 256, 32, 32])

    print(out.keys())
