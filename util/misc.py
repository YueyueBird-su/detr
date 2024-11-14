# 版权所有 (c) Facebook, Inc. 及其子公司。保留所有权利。
"""
杂项功能，包括分布式助手。

大部分代码来自 torchvision 的参考实现。
"""
import os  # 导入操作系统模块，用于与操作系统交互
import subprocess  # 导入子进程模块，用于执行系统命令
import time  # 导入时间模块，用于时间相关操作
from collections import defaultdict, deque  # 导入集合模块中的 defaultdict 和 deque
import datetime  # 导入日期时间模块
import pickle  # 导入 pickle 模块，用于序列化和反序列化对象
from packaging import version  # 导入版本模块，用于版本比较
from typing import Optional, List  # 导入类型提示模块中的 Optional 和 List

import torch  # 导入 PyTorch
import torch.distributed as dist  # 导入 PyTorch 的分布式模块
from torch import Tensor  # 从 PyTorch 导入 Tensor

# 由于 pytorch 和 torchvision 0.5 中的空张量 bug，需要以下导入
import torchvision  # 导入 torchvision

if version.parse(torchvision.__version__) < version.parse(
    "0.7"
):  # 如果 torchvision 版本小于 0.7
    from torchvision.ops import _new_empty_tensor  # 导入 _new_empty_tensor
    from torchvision.ops.misc import _output_size  # 导入 _output_size


class SmoothedValue(object):
    """跟踪一系列值，并提供窗口或全局系列平均值的平滑值访问。"""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(
            maxlen=window_size
        )  # 初始化一个双端队列，最大长度为 window_size
        self.total = 0.0  # 初始化总值
        self.count = 0  # 初始化计数
        self.fmt = fmt  # 初始化格式

    def update(self, value, n=1):
        self.deque.append(value)  # 将值添加到双端队列
        self.count += n  # 增加计数
        self.total += value * n  # 增加总值

    def synchronize_between_processes(self):
        """
        警告：不会同步 deque！
        """
        if not is_dist_avail_and_initialized():  # 如果分布式不可用或未初始化
            return
        t = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device="cuda"
        )  # 创建一个张量，包含计数和总值
        dist.barrier()  # 同步所有进程
        dist.all_reduce(t)  # 对所有进程的张量进行求和
        t = t.tolist()  # 将张量转换为列表
        self.count = int(t[0])  # 更新计数
        self.total = t[1]  # 更新总值

    @property
    def median(self):
        d = torch.tensor(list(self.deque))  # 将双端队列转换为张量
        return d.median().item()  # 返回中位数

    @property
    def avg(self):
        d = torch.tensor(
            list(self.deque), dtype=torch.float32
        )  # 将双端队列转换为浮点型张量
        return d.mean().item()  # 返回平均值

    @property
    def global_avg(self):
        return self.total / self.count  # 返回全局平均值

    @property
    def max(self):
        return max(self.deque)  # 返回最大值

    @property
    def value(self):
        return self.deque[-1]  # 返回最新值

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    在任意可序列化的数据（不一定是张量）上运行 all_gather
    参数：
        data: 任意可序列化的对象
    返回：
        list[data]: 从每个 rank 收集的数据列表
    """
    world_size = get_world_size()  # 获取全局进程数
    if world_size == 1:  # 如果只有一个进程
        return [data]

    # 序列化为张量
    buffer = pickle.dumps(data)  # 将数据序列化为字节流
    storage = torch.ByteStorage.from_buffer(buffer)  # 创建字节存储
    tensor = torch.ByteTensor(storage).to("cuda")  # 创建字节张量并移动到 GPU

    # 获取每个 rank 的张量大小
    local_size = torch.tensor([tensor.numel()], device="cuda")  # 获取本地张量的大小
    size_list = [
        torch.tensor([0], device="cuda") for _ in range(world_size)
    ]  # 初始化大小列表
    dist.all_gather(size_list, local_size)  # 收集所有进程的张量大小
    size_list = [int(size.item()) for size in size_list]  # 将大小列表转换为整数列表
    max_size = max(size_list)  # 获取最大张量大小

    # 从所有 rank 接收张量
    # 我们填充张量，因为 torch all_gather 不支持收集不同形状的张量
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size,), dtype=torch.uint8, device="cuda")
        )  # 初始化张量列表
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )  # 创建填充张量
        tensor = torch.cat((tensor, padding), dim=0)  # 填充张量
    dist.all_gather(tensor_list, tensor)  # 收集所有进程的张量

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]  # 将张量转换为字节流
        data_list.append(pickle.loads(buffer))  # 反序列化字节流

    return data_list


def reduce_dict(input_dict, average=True):
    """
    参数：
        input_dict (dict): 所有值将被减少
        average (bool): 是否进行平均或求和
    将所有进程中的字典值减少，以便所有进程都有平均结果。返回一个与 input_dict 具有相同字段的字典，经过减少后。
    """
    world_size = get_world_size()  # 获取全局进程数
    if world_size < 2:  # 如果只有一个进程
        return input_dict
    with torch.no_grad():  # 禁用梯度计算
        names = []
        values = []
        # 对键进行排序，以便在所有进程中一致
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)  # 将值堆叠成张量
        dist.all_reduce(values)  # 对所有进程的张量进行求和
        if average:
            values /= world_size  # 计算平均值
        reduced_dict = {k: v for k, v in zip(names, values)}  # 创建减少后的字典
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(
            SmoothedValue
        )  # 初始化一个默认字典，值为 SmoothedValue 对象
        self.delimiter = delimiter  # 初始化分隔符

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):  # 如果值是张量
                v = v.item()  # 将张量转换为标量
            assert isinstance(v, (float, int))  # 确保值是浮点数或整数
            self.meters[k].update(v)  # 更新字典中的值

    def __getattr__(self, attr):
        if attr in self.meters:  # 如果属性在字典中
            return self.meters[attr]  # 返回字典中的值
        if attr in self.__dict__:  # 如果属性在实例字典中
            return self.__dict__[attr]  # 返回实例字典中的值
        raise AttributeError(
            "'{}' 对象没有属性 '{}'".format(type(self).__name__, attr)
        )  # 引发属性错误

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)  # 返回格式化的字符串

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()  # 同步所有进程的值

    def add_meter(self, name, meter):
        self.meters[name] = meter  # 添加新的计量器

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()  # 记录开始时间
        end = time.time()  # 记录结束时间
        iter_time = SmoothedValue(fmt="{avg:.4f}")  # 初始化迭代时间计量器
        data_time = SmoothedValue(fmt="{avg:.4f}")  # 初始化数据时间计量器
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"  # 格式化字符串
        if torch.cuda.is_available():  # 如果 CUDA 可用
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0  # 定义 MB 单位
        for obj in iterable:
            data_time.update(time.time() - end)  # 更新数据时间
            yield obj
            iter_time.update(time.time() - end)  # 更新迭代时间
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (
                    len(iterable) - i
                )  # 计算预计剩余时间
                eta_string = str(
                    datetime.timedelta(seconds=int(eta_seconds))
                )  # 格式化预计剩余时间
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )  # 打印日志
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )  # 打印日志
            i += 1
            end = time.time()  # 更新结束时间
        total_time = time.time() - start_time  # 计算总时间
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # 格式化总时间
        print(
            "{} 总时间: {} ({:.4f} 秒 / 次)".format(
                header, total_time_str, total时间 / len(iterable)
            )
        )  # 打印总时间


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))  # 获取当前工作目录

    def _run(command):
        return (
            subprocess.check_output(command, cwd=cwd).decode("ascii").strip()
        )  # 运行命令并返回输出

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])  # 获取当前 git 提交的 SHA
        subprocess.check_output(["git", "diff"], cwd=cwd)  # 检查是否有未提交的更改
        diff = _run(["git", "diff-index", "HEAD"])  # 获取未提交的更改
        diff = "有未提交的更改" if diff else "干净"  # 设置 diff 状态
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])  # 获取当前分支
    except Exception:
        pass
    message = f"sha: {sha}, 状态: {diff}, 分支: {branch}"  # 格式化消息
    return message


def collate_fn(batch):
    batch = list(zip(*batch))  # 转置批次
    batch[0] = nested_tensor_from_tensor_list(batch[0])  # 将张量列表转换为嵌套张量
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]  # 初始化最大值列表
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)  # 更新最大值
    return maxes


class NestedTensor(object):
    """
    嵌套张量是一个包含张量和掩码的元组，用于处理不同大小的图像。
    """

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors  # 初始化张量
        self.mask = mask  # 初始化掩码

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)  # 将张量移动到设备
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)  # 将掩码移动到设备
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)  # 返回嵌套张量

    def decompose(self):
        return self.tensors, self.mask  # 返回张量和掩码

    def __repr__(self):
        return str(self.tensors)  # 返回张量的字符串表示


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO 使其更通用
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() 不适合导出到 ONNX
            # 调用 _onnx_nested_tensor_from_tensor_list() 代替
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO 使其支持不同大小的图像
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])  # 获取最大尺寸
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size  # 构建批次形状
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype  # 获取数据类型
        device = tensor_list[0].device  # 获取设备
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)  # 创建零张量
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)  # 创建掩码
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)  # 填充图像
            m[: img.shape[1], : img.shape[2]] = False  # 更新掩码
    else:
        raise ValueError("不支持")
    return NestedTensor(tensor, mask)  # 返回嵌套张量


# _onnx_nested_tensor_from_tensor_list() 是 nested_tensor_from_tensor_list() 的实现，支持 ONNX 跟踪。
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(
            torch.int64
        )  # 获取最大尺寸
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # 解决 pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # 这在 onnx 中尚不支持
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]  # 计算填充
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )  # 填充图像
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)  # 创建掩码
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )  # 填充掩码
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)  # 堆叠图像
    mask = torch.stack(padded_masks)  # 堆叠掩码

    return NestedTensor(tensor, mask=mask)  # 返回嵌套张量


def setup_for_distributed(is_master):
    """
    当不是主进程时禁用打印
    """
    import builtins as __builtin__  # 导入内置模块

    builtin_print = __builtin__.print  # 获取内置打印函数

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)  # 获取 force 参数
        if is_master or force:  # 如果是主进程或强制打印
            builtin_print(*args, **kwargs)  # 调用内置打印函数

    __builtin__.print = print  # 覆盖内置打印函数


def is_dist_avail_and_initialized():
    if not dist.is_available():  # 检查分布式是否可用
        return False  # 如果不可用，返回 False
    if not dist.is_initialized():  # 检查分布式是否已初始化
        return False  # 如果未初始化，返回 False
    return True  # 如果可用且已初始化，返回 True


def get_world_size():
    if not is_dist_avail_and_initialized():  # 检查分布式是否可用且已初始化
        return 1  # 如果不可用或未初始化，返回 1
    return dist.get_world_size()  # 返回全局进程数


def get_rank():
    if not is_dist_avail_and_initialized():  # 检查分布式是否可用且已初始化
        return 0  # 如果不可用或未初始化，返回 0
    return dist.get_rank()  # 返回当前进程的 rank


def is_main_process():
    return get_rank() == 0  # 检查当前进程是否为主进程


def save_on_master(*args, **kwargs):
    if is_main_process():  # 如果是主进程
        torch.save(*args, **kwargs)  # 保存模型


def init_distributed_mode(args):
    if (
        "RANK" in os.environ and "WORLD_SIZE" in os.environ
    ):  # 检查环境变量中是否有 RANK 和 WORLD_SIZE
        args.rank = int(os.environ["RANK"])  # 设置 rank
        args.world_size = int(os.environ["WORLD_SIZE"])  # 设置全局进程数
        args.gpu = int(os.environ["LOCAL_RANK"])  # 设置本地 GPU
    elif "SLURM_PROCID" in os.environ:  # 检查环境变量中是否有 SLURM_PROCID
        args.rank = int(os.environ["SLURM_PROCID"])  # 设置 rank
        args.gpu = args.rank % torch.cuda.device_count()  # 设置本地 GPU
    else:
        print("Not using distributed mode")  # 打印不使用分布式模式
        args.distributed = False  # 设置分布式为 False
        return

    args.distributed = True  # 设置分布式为 True

    torch.cuda.set_device(args.gpu)  # 设置当前 GPU 设备
    args.dist_backend = "nccl"  # 设置分布式后端为 NCCL
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )  # 打印分布式初始化信息
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )  # 初始化分布式进程组
    torch.distributed.barrier()  # 同步所有进程
    setup_for_distributed(args.rank == 0)  # 设置分布式环境


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """计算指定 k 值的精度@k"""
    if target.numel() == 0:  # 如果目标张量为空
        return [torch.zeros([], device=output.device)]  # 返回一个空张量
    maxk = max(topk)  # 获取 topk 中的最大值
    batch_size = target.size(0)  # 获取批次大小

    _, pred = output.topk(maxk, 1, True, True)  # 获取 topk 预测结果
    pred = pred.t()  # 转置预测结果
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 计算预测结果是否正确

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)  # 计算 topk 正确预测的数量
        res.append(correct_k.mul_(100.0 / batch_size))  # 计算精度并添加到结果列表
    return res  # 返回精度列表


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    等效于 nn.functional.interpolate，但支持空批次大小。
    这将最终由 PyTorch 原生支持，这个类可以消失。
    """
    if version.parse(torchvision.__version__) < version.parse(
        "0.7"
    ):  # 如果 torchvision 版本小于 0.7
        if input.numel() > 0:  # 如果输入张量不为空
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )  # 调用 PyTorch 的插值函数

        output_shape = _output_size(2, input, size, scale_factor)  # 计算输出形状
        output_shape = list(input.shape[:-2]) + list(output_shape)  # 构建输出形状
        return _new_empty_tensor(input, output_shape)  # 返回一个新的空张量
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )  # 调用 torchvision 的插值函数
