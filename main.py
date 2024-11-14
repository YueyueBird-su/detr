# 导入必要的库和模块
import argparse
import datetime
import json
import random
import time
from pathlib import Path  # Path模块用于处理文件路径

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


# 定义一个函数来解析命令行参数
def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)  # 学习率
    parser.add_argument("--lr_backbone", default=1e-5, type=float)  # 主干网络的学习率
    parser.add_argument("--batch_size", default=2, type=int)  # 批处理大小
    parser.add_argument("--weight_decay", default=1e-4, type=float)  # 权重衰减
    parser.add_argument("--epochs", default=300, type=int)  # 训练的总轮数
    parser.add_argument("--lr_drop", default=200, type=int)  # 学习率下降的轮数
    # 梯度裁剪的最大范数，在训练过程中，这个参数会在 train_one_epoch 函数中被使用，用于防止梯度爆炸。
    # 具体来说，它会限制梯度的最大范数，从而确保训练过程的稳定性。
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # 模型参数
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )  # 预训练模型的路径，如果设置了这个参数，只有 mask head 会被训练。
    # 主干网络
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )  # 主干网络的名称
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )  # 是否使用膨胀卷积，如果设置了这个参数，最后一个卷积块的 stride 会被 dilation 替代。
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )  # 位置嵌入的类型
    ## sine 和 learned 是两种不同的位置编码方式：
    # Sine（正弦位置编码）：

    # 使用固定的正弦和余弦函数生成位置编码。
    # 位置编码是预定义的，不会在训练过程中更新。
    # 这种方法不依赖于数据，适用于任何长度的序列。
    # Learned（学习位置编码）：

    # 位置编码是可训练的参数，会在训练过程中更新。
    # 这种方法允许模型根据数据学习最优的位置编码。
    # 可能更适合特定任务，但需要更多的训练数据和时间。

    # Transformer参数
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )  # 编码层数
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )  # 解码层数
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )  # 前馈网络的中间层大小
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )  # 嵌入的大小，设置Transformer 模型中隐藏层的维度。这个参数决定了模型中各层之间传递的向量的维度大小。
    # 较大的 hidden_dim 可以增加模型的表达能力，但也会增加计算和存储的开销。
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )  # Dropout率
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )  # 注意力头的数量
    parser.add_argument(
        "--num_queries", default=100, type=int, help="Number of query slots"
    )  # 查询查询槽的数量，每个查询向量会与编码器的输出进行交互，以生成最终的预测结果。查询向量的数量通常与模型需要检测的目标数量相关。
    parser.add_argument(
        "--pre_norm", action="store_true"
    )  # 是否使用预归一化，当设置该参数时，模型会在每个子层的输入上应用 Layer Normalization，而不是在子层的输出上。这可以帮助稳定训练过程，特别是在深层网络中。

    # 分割参数
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )  # 是否训练分割头

    # 损失参数
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )  # 是否禁用辅助解码损失

    # 匹配器参数
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )  # 类别匹配成本系数
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )  # L1框匹配成本系数
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )  # giou框匹配成本系数
    # 损失系数
    parser.add_argument("--mask_loss_coef", default=1, type=float)  # 掩码损失系数
    parser.add_argument("--dice_loss_coef", default=1, type=float)  # Dice损失系数
    parser.add_argument("--bbox_loss_coef", default=5, type=float)  # 边框损失系数
    parser.add_argument("--giou_loss_coef", default=2, type=float)  # giou损失系数
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )  # 无对象类的相对分类权重

    # 数据集参数
    parser.add_argument("--dataset_file", default="coco")  # 数据集文件
    parser.add_argument("--coco_path", type=str)  # COCO数据集路径
    parser.add_argument("--coco_panoptic_path", type=str)  # COCO全景数据集路径
    parser.add_argument("--remove_difficult", action="store_true")  # 是否移除困难样本

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )  # 输出目录
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )  # 设备类型
    parser.add_argument("--seed", default=42, type=int)  # 随机种子
    parser.add_argument(
        "--resume", default="", help="resume from checkpoint"
    )  # 从检查点恢复
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )  # 起始轮数
    parser.add_argument("--eval", action="store_true")  # 是否进行评估
    parser.add_argument("--num_workers", default=2, type=int)  # 工作线程数

    # 分布式训练参数
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )  # 分布式进程数
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )  # 分布式训练的URL
    return parser


# 主函数
def main(args):
    utils.init_distributed_mode(args)  # 初始化分布式模式
    print("git:\n  {}\n".format(utils.get_sha()))  # 打印git版本信息

    if args.frozen_weights is not None:
        assert (
            args.masks
        ), "Frozen training is meant for segmentation only"  # 冻结训练仅用于分割
    print(args)

    device = torch.device(args.device)  # 设置设备

    # 固定随机种子以保证可重复性
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)  # 构建模型
    model.to(device)  # 将模型移动到设备上

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )  # 分布式数据并行
        model_without_ddp = model.module
    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )  # 计算模型参数数量
    print("number of params:", n_parameters)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },  # 主干网络之外的参数
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],  # 主干网络的参数
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )  # 优化器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop
    )  # 学习率调度器

    dataset_train = build_dataset(image_set="train", args=args)  # 构建训练数据集
    dataset_val = build_dataset(image_set="val", args=args)  # 构建验证数据集

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)  # 分布式采样器
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # 随机采样器
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # 顺序采样器

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )  # 批采样器

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )  # 训练数据加载器
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )  # 验证数据加载器

    if args.dataset_file == "coco_panoptic":
        # 在全景训练期间，我们还在原始COCO数据集上评估AP
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")  # 加载冻结权重
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )  # 从URL加载检查点
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")  # 从本地加载检查点
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
        )  # 评估模型
        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )  # 保存评估结果
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)  # 设置当前轮数
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )  # 训练一个轮次
        lr_scheduler.step()  # 更新学习率
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # 在学习率下降前和每100轮保存一次检查点
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
        )  # 评估模型

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")  # 保存日志

            # 保存评估日志
            if coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name,
                        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )  # 创建参数解析器
    args = parser.parse_args()  # 解析参数
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 创建输出目录
    main(args)  # 调用主函数
