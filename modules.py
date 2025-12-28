import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalizer(nn.Module):
    def __init__(self, num_features, norm_type="bn", momentum=0.1):
        """
        PyTorch 的 momentum 定义与 TF 的 decay 不同：
        PyTorch momentum = 1 - TF decay (即 TF 0.99 对应 PyTorch 0.01)
        """
        super(Normalizer, self).__init__()
        self.norm_type = norm_type
        if norm_type == "bn":
            # 自动适配 1D (Batch, C, L) 或 2D (Batch, C, H, W)
            self.norm = nn.BatchNorm2d(num_features, momentum=momentum)
        elif norm_type == "ln":
            # LayerNorm 通常在最后几个维度进行
            self.norm = nn.InstanceNorm2d(num_features, affine=True) # 简化逻辑
        elif norm_type == "in":
            self.norm = nn.InstanceNorm2d(num_features, affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # PyTorch 默认输入是 (N, C, H, W)，而 TF 通常是 (N, H, W, C)
        # 如果你的输入是从之前的 TF 代码迁移的，可能需要先 permute
        return self.norm(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rate=1, 
                 padding="SAME", use_bias=False, norm_type=None, activation=None):
        super(ConvBlock, self).__init__()
        
        # 处理 Padding
        if padding.upper() == "SAME":
            pad_val = (kernel_size - 1) * rate // 2
        elif padding.upper() == "VALID":
            pad_val = 0
        else:
            pad_val = 0 # 默认

        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=pad_val, 
            dilation=rate, 
            bias=use_bias
        )
        
        # 归一化层
        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "ln":
            # 注意：PyTorch LayerNorm 需要指定输入 shape
            self.norm = nn.GroupNorm(1, out_channels) # GroupNorm(1,...) 等效于 LayerNorm
        elif norm_type == "in":
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.Identity()
            
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        
        # 为了保证 x 可以和 f(x) 相加，输入输出通道必须一致
        # 同时 padding 必须设为 (kernel_size - 1) // 2 以保持尺寸不变
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # 保存原始输入
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 关键：这就是残差连接 (Shortcut)
        out = self.relu(out)  # 最后再过一次激活函数
        return out