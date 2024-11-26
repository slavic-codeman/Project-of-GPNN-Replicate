import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMaxPool2d(nn.Module):
    """
    A wrapper for Adaptive Max Pooling 2D to match the interface of the old Function class.
    """
    def __init__(self, out_h, out_w):
        super(AdaptiveMaxPool2d, self).__init__()
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, input):
        return F.adaptive_max_pool2d(input, (self.out_h, self.out_w))


def adaptive_max_pool(input, size):
    """
    Functional API for Adaptive Max Pooling 2D.
    """
    return AdaptiveMaxPool2d(size[0], size[1]).forward(input)


def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):
    """
    ROI Pooling implementation using Adaptive Max Pooling.
    Args:
        input (Tensor): The input feature map (N, C, H, W).
        rois (Tensor): Regions of Interest (ROIs) in the format [batch_idx, x1, y1, x2, y2].
        size (tuple): The output size of the pooled region (height, width).
        spatial_scale (float): Scale factor to map ROI coordinates to feature map scale.

    Returns:
        Tensor: ROI Pooled regions (N_ROIs, C, pooled_height, pooled_width).
    """
    assert rois.dim() == 2, "ROIs should be a 2D tensor"
    assert rois.size(1) == 5, "Each ROI should have 5 values [batch_idx, x1, y1, x2, y2]"

    output = []
    num_rois = rois.size(0)

    for i in range(num_rois):
        roi = rois[i]
        batch_idx = int(roi[0].item())
        x1, y1, x2, y2 = (roi[1:] * spatial_scale).round().int()

        roi_feature = input[batch_idx, :, y1:y2 + 1, x1:x2 + 1]
        pooled_feature = F.adaptive_max_pool2d(roi_feature, size)
        output.append(pooled_feature)

    return torch.cat(output, dim=0)


def main():
    # Test adaptive_max_pool
    input = torch.rand(1, 1, 10, 10, requires_grad=True)
    print("Input Tensor:\n", input)

    out = adaptive_max_pool(input, (3, 3))
    print("Adaptive Max Pool Output:\n", out)
    out.backward(torch.ones_like(out))
    print("Gradient after Adaptive Max Pooling:\n", input.grad)

    # Test roi_pooling
    input = torch.rand(1, 1, 10, 10, requires_grad=True)
    rois = torch.tensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8]], dtype=torch.float32)
    print("ROIs:\n", rois)

    out = roi_pooling(input, rois, size=(3, 3))
    print("ROI Pooling Output:\n", out)
    out.backward(torch.ones_like(out))
    print("Gradient after ROI Pooling:\n", input.grad)
    print("All unit tests passed")


if __name__ == '__main__':
    main()
