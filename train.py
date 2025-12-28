import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 导入自定义模块
from model import SudokuNet
from data_load import SudokuDataset
from hyperparams import Hyperparams as hp

def train():
    # 自动选择设备（RTX 3090 将自动使用 CUDA）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 准备数据
    train_dataset = SudokuDataset(data_type="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hp.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    # 2. 实例化模型
    # 注意：请确保 SudokuNet 的最后一层 Conv2d 输出通道数为 9
    model = SudokuNet(hp).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, 
                                          steps_per_epoch=len(train_loader), 
                                          epochs=hp.num_epochs)
    
    # 损失函数：使用 reduction='none' 以便后续应用掩码
    criterion = nn.CrossEntropyLoss(reduction='none') 

    # 3. 训练循环
    model.train()
    global_step = 0
    
    print("Start Training...")
    for epoch in range(1, hp.num_epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=90)
        
        for x, y in pbar:
            # x: (N, 9, 9), y: (N, 9, 9) 其中 y 包含数字 1-9
            x, y = x.to(device), y.to(device).long()
            
            # 前向传播
            # logits 形状预期为 (N, 9, 9, 9) -> (批次, 类别, 高, 宽)
            logits = model(x) 
            
            # 计算掩码 (istarget): 只有原题为 0 的空白格才计算损失
            istarget = (x == 0).float() 
            
            # 【关键修改 1】：标签减 1
            # PyTorch CrossEntropy 期望标签为 0-8，所以 y(1-9) 需要减 1
            loss_map = criterion(logits, y - 1) 
            
            # 应用掩码并计算平均损失
            loss = (loss_map * istarget).sum() / (istarget.sum() + 1e-8)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 【关键修改 2】：计算准确率时还原预测值
            # argmax 得到的是 0-8，加 1 还原回 1-9
            preds = torch.argmax(logits, dim=1) + 1 
            
            # 只统计空白格的正确数
            hits = (preds == y).float() * istarget
            acc = hits.sum() / (istarget.sum() + 1e-8)
            
            global_step += 1
            
            # 每 10 步更新一次进度条显示
            if global_step % 10 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}", 
                    'Acc': f"{acc.item():.4f}"
                })

        # 4. 保存 Checkpoint
        if not os.path.exists(hp.logdir):
            os.makedirs(hp.logdir)
        
        # 保存最新模型，方便 test.py 加载
        latest_path = os.path.join(hp.logdir, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)
        
        # 也可以按 epoch 备份
        epoch_path = os.path.join(hp.logdir, f"model_epoch_{epoch:02d}.pth")
        torch.save(model.state_dict(), epoch_path)
        
        print(f"Epoch {epoch} finished. Model saved.")

if __name__ == "__main__":
    train()