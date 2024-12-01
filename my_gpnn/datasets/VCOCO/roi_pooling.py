import torch
import torch.nn as nn
import torch.autograd as ag
import math
from torch.autograd.function import Function

"""这个文件发生了较大的改动，需要注意！！！"""
class AdaptiveMaxPool2d(Function):
    @staticmethod
    def forward(ctx, input, out_w, out_h):
        """
        前向传播方法，使用静态方法定义。
        """
        # 使用 PyTorch 内置的功能进行池化
        output = torch.nn.functional.adaptive_max_pool2d(input, (out_w, out_h))
        
        # 保存输入和其他需要在反向传播中使用的变量
        ctx.save_for_backward(input)
        ctx.out_w = out_w
        ctx.out_h = out_h
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播方法，使用静态方法定义。
        """
        input, = ctx.saved_tensors
        # 将梯度调整为输入尺寸大小
        grad_input = torch.nn.functional.adaptive_max_pool2d(grad_output, input.shape[2:])
        return grad_input, None, None  # None对应的参数是forward中的out_w和out_h，它们不需要梯度



def adaptive_max_pool(input, size):
    """
    使用自定义的自适应最大池化函数
    """
    return AdaptiveMaxPool2d.apply(input, size[0], size[1])


def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):
    """
    实现区域池化（RoI Pooling）
    """
    assert rois.dim() == 2
    assert rois.size(1) == 5

    output = []
    rois = rois.float()  # 转换为浮点类型
    num_rois = rois.size(0)

    # 缩放 RoIs 坐标
    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()  # 转换为长整型

    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        output.append(adaptive_max_pool(im, size))

    return torch.cat(output, 0)


if __name__ == '__main__':
    # 示例输入
    input = torch.rand(1, 1, 10, 10, requires_grad=True)  # 不再使用 ag.Variable，直接使用 Tensor
    rois = torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8]])  # 直接使用 LongTensor，PyTorch 3.9 已经支持了

    # 使用自适应最大池化
    out = adaptive_max_pool(input, (3, 3))
    out.backward(out.data.clone().uniform_())

    # 使用RoI池化
    out = roi_pooling(input, rois, size=(3, 3))
    out.backward(out.data.clone().uniform_())
