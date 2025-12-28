import torch
import torch.nn as nn
import numpy as np
import os
import copy
from hyperparams import Hyperparams as hp

from model import SudokuNet 

def is_perfect_sudoku(grid):
    """
    检查一个 9x9 的 numpy 矩阵是否为合法的数独解。
    """
    # 1. 检查是否还有空格 (0)
    if np.any(grid == 0):
        return False
    
    # 2. 检查行和列
    for i in range(9):
        # 行校验
        if len(set(grid[i, :])) != 9: return False
        # 列校验
        if len(set(grid[:, i])) != 9: return False
        
    # 3. 检查 3x3 宫
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            block = grid[r:r+3, c:c+3].flatten()
            if len(set(block)) != 9: return False
            
    return True

def write_to_file(x, y, preds, fout_path):
    '''将结果写入文件并计算准确率及完美率(PR)'''
    with open(fout_path, 'w', encoding='utf-8') as fout:
        total_hits, total_blanks = 0, 0
        perfect_count = 0  # 新增：记录完全正确的数量
        total_samples = len(x)
        
        for i in range(total_samples):
            xx, yy, pp = x[i], y[i], preds[i]
            
            qz_str = "".join(str(int(num)) if num != 0 else "_" for num in xx.flatten())
            sn_str = "".join(str(int(num)) for num in yy.flatten())
            pd_str = "".join(str(int(num)) for num in pp.flatten())
            
            fout.write(f"qz: {qz_str}\n")
            fout.write(f"sn: {sn_str}\n")
            fout.write(f"pd: {pd_str}\n")

            # 传统 Acc 计算 (与答案对比)
            mask = (xx == 0)
            num_hits = np.equal(yy[mask], pp[mask]).sum()
            num_blanks = mask.sum()
            acc = (num_hits / num_blanks) if num_blanks > 0 else 1.0
            
            # 完美率检测 (逻辑校验)
            is_perfect = is_perfect_sudoku(pp)
            if is_perfect:
                perfect_count += 1
            
            fout.write(f"accuracy = {num_hits}/{num_blanks} = {acc:.2f} | Perfect: {is_perfect}\n\n")

            total_hits += num_hits
            total_blanks += num_blanks
            
        total_acc = total_hits / total_blanks if total_blanks > 0 else 0
        perfect_rate = perfect_count / total_samples if total_samples > 0 else 0
        
        fout.write("-" * 30 + "\n")
        fout.write(f"Total Token-level Accuracy = {total_acc:.4f}\n")
        fout.write(f"Perfect Rate (PR) = {perfect_count}/{total_samples} = {perfect_rate:.4f}\n")
        print(f"Test Done. Perfect Rate: {perfect_rate:.2%}")

# def test():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     
#     # 1. 使用你已有的 get_dataloader 获取数据
#     from data_load import get_dataloader
#     test_loader, _ = get_dataloader(data_type="test", batch_size=hp.batch_size)
#     
#     # 2. 加载模型
#     model = SudokuNet(hp).to(device)
#     checkpoint_path = os.path.join(hp.logdir, "latest_model.pth")
#     if os.path.exists(checkpoint_path):
#         model.load_state_dict(torch.load(checkpoint_path))
#     model.eval()
# 
#     all_x, all_y, all_preds = [], [], []
# 
#     # ... 前面部分相同 ...
#     print("Testing with iterative inference...")
#     with torch.no_grad():
#         for x_batch, y_batch in test_loader:
#             x_np = x_batch.numpy()
#             y_np = y_batch.numpy()
#             curr_x = copy.deepcopy(x_np)
#             
#             # 迭代推理开始
#             while True:
#                 inputs = torch.from_numpy(curr_x).float().to(device)
#                 logits = model(inputs) # 形状 (N, 9, 9, 9)
#                 probs = torch.softmax(logits, dim=1)
#                 
#                 # 【修改点 1】: 不要切片 [:, 1:]。直接在 9 个通道找最大
#                 # val_preds 此时是 0-8
#                 val_probs, val_preds = torch.max(probs, dim=1) 
#                 
#                 # 【修改点 2】: 还原为数字 1-9
#                 v_probs = val_probs.cpu().numpy()
#                 v_preds = (val_preds + 1).cpu().numpy() # 0-8 -> 1-9
#                 
#                 mask = (curr_x == 0)
#                 masked_probs = v_probs * mask
#                 
#                 # 找到每张图中最有把握的格
#                 m_probs_flat = masked_probs.reshape(len(curr_x), -1)
#                 if np.all(np.max(m_probs_flat, axis=1) == 0): 
#                     break # 填满了或者没把握了，退出
#                 
#                 max_idx = np.argmax(m_probs_flat, axis=1)
#                 max_p = np.max(m_probs_flat, axis=1)
#                 
#                 curr_x_flat = curr_x.reshape(len(curr_x), -1)
#                 v_preds_flat = v_preds.reshape(len(curr_x), -1)
#                 
#                 for i in range(len(curr_x)):
#                     if max_p[i] > 0:
#                         # 填入最有信心的那个数字
#                         curr_x_flat[i, max_idx[i]] = v_preds_flat[i, max_idx[i]]
#                 
#                 curr_x = curr_x_flat.reshape(-1, 9, 9)
# 
#             all_x.append(x_np)
#             all_y.append(y_np)
#             all_preds.append(curr_x)
# 
#     # 将所有 Batch 的结果合并
#     x_final = np.concatenate(all_x, axis=0)
#     y_final = np.concatenate(all_y, axis=0)
#     p_final = np.concatenate(all_preds, axis=0)
# 
#     write_to_file(x_final.astype(np.int32), y_final, p_final.astype(np.int32), hp.result_fpath)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from data_load import get_dataloader
    # 建议测试时 batch_size 不要太大，128 或 256 即可，因为迭代次数较多
    test_loader, _ = get_dataloader(data_type="test", batch_size=hp.batch_size)
    
    model = SudokuNet(hp).to(device)
    checkpoint_path = os.path.join(hp.logdir, "latest_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_x, all_y, all_preds = [], [], []

    print(f"Testing with prioritized iterative inference on {device}...")
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_np = x_batch.numpy()
            y_np = y_batch.numpy()
            curr_x = copy.deepcopy(x_np).astype(np.float32)
            
            # --- 迭代推理开始 ---
            # 最多迭代 81 次（实际取决于题目留白数量）
            for step in range(81):
                # 1. 检查是否所有样本都填满了
                mask = (curr_x == 0)
                if not np.any(mask):
                    break
                
                # 2. 模型预测
                inputs = torch.from_numpy(curr_x).to(device)
                logits = model(inputs) 
                probs = torch.softmax(logits, dim=1) # (N, 9, 9, 9)
                
                # 3. 获取每个格子的最大概率及对应的数字
                val_probs, val_preds = torch.max(probs, dim=1)
                v_probs = val_probs.cpu().numpy()
                v_preds = (val_preds + 1).cpu().numpy()
                
                # 4. 只关注原本为空白的位置
                masked_probs = v_probs * mask
                
                # 5. 找到每个样本中最有把握（概率最大）的那个空格
                m_probs_flat = masked_probs.reshape(len(curr_x), -1)
                max_indices = np.argmax(m_probs_flat, axis=1) # 每个样本最大概率的索引
                max_confidences = np.max(m_probs_flat, axis=1) # 每个样本的最大概率值
                
                # 如果所有样本都没有可以填的空了（虽然极少发生），则退出
                if np.all(max_confidences == 0):
                    break
                
                # 6. 执行填入动作
                curr_x_flat = curr_x.reshape(len(curr_x), -1)
                v_preds_flat = v_preds.reshape(len(curr_x), -1)
                
                updated = False
                for i in range(len(curr_x)):
                    # 只有当该样本还有空格且有预测概率时才填充
                    if max_confidences[i] > 0:
                        curr_x_flat[i, max_indices[i]] = v_preds_flat[i, max_indices[i]]
                        updated = True
                
                curr_x = curr_x_flat.reshape(-1, 9, 9)
                
                # 如果这一轮没有任何更新，提前结束
                if not updated:
                    break
            # --- 迭代推理结束 ---

            all_x.append(x_np)
            all_y.append(y_np)
            all_preds.append(curr_x)

    # 合并结果并写入文件
    x_final = np.concatenate(all_x, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    p_final = np.concatenate(all_preds, axis=0)

    write_to_file(x_final.astype(np.int32), y_final, p_final.astype(np.int32), hp.result_fpath)

if __name__ == '__main__':
    test()