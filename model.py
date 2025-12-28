import torch
import torch.nn as nn
from modules import ConvBlock, ResBlock # 确保导入了 ResBlock

class SudokuNet(nn.Module):
    def __init__(self, hp):
        super(SudokuNet, self).__init__()
        self.hp = hp
        
        # 1. 初始层：将 1 通道映射到高维 (例如 512)
        # 这一步很重要，因为 ResBlock 要求输入输出通道数一致
        self.first_layer = ConvBlock(
            in_channels=1,
            out_channels=hp.num_filters,
            kernel_size=hp.filter_size,
            padding="SAME",
            norm_type="bn",
            activation=nn.ReLU()
        )
        
        # 2. 残差骨干网：堆叠 hp.num_blocks 个 ResBlock
        # 每个 ResBlock 包含两层卷积和一个 Shortcut
        res_layers = []
        for i in range(hp.num_blocks):
            res_layers.append(
                ResBlock(channels=hp.num_filters, kernel_size=hp.filter_size, norm_type="bn")
            )
        self.res_backbone = nn.Sequential(*res_layers)
        
        # 3. 输出层
        self.logits_layer = nn.Conv2d(hp.num_filters, 9, kernel_size=1)

    def forward(self, x):
        # 转换形状: (Batch, 1, 9, 9)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
            
        # 第一步：提升维度
        x = self.first_layer(x)
        
        # 第二步：通过残差块进行深层逻辑推理
        x = self.res_backbone(x)
        
        # 第三步：1x1 卷积得到 logits (Batch, 9, 9, 9)
        logits = self.logits_layer(x)
        
        return logits