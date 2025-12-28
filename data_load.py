import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from hyperparams import Hyperparams as hp

class SudokuDataset(Dataset):
    def __init__(self, data_type="train"):
        """
        初始化数据集，一次性将数据加载进内存
        """
        fpath = hp.train_fpath if data_type == "train" else hp.test_fpath
        print(f"Loading {data_type} data from: {fpath}")
        
        # Python 3 中 open 默认支持 utf-8
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()[1:]  # 跳过表头
        
        self.samples = []
        for line in lines:
            quiz, solution = line.split(",")
            # 将字符串转换为数字列表，再转为 numpy 数组
            q_arr = np.array([int(c) for c in quiz], dtype=np.float32).reshape(9, 9)
            s_arr = np.array([int(c) for c in solution], dtype=np.int64).reshape(9, 9)
            
            # 可以在这里做归一化 (例如 q_arr / 9.0)
            self.samples.append((q_arr, s_arr))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引返回一对 (题目, 答案)
        PyTorch 会自动将 numpy 转换为 Tensor
        """
        quiz, solution = self.samples[idx]
        return torch.from_numpy(quiz), torch.from_numpy(solution)

def get_dataloader(data_type="train", batch_size=None):
    """
    返回 DataLoader 和 总批次数
    """
    if batch_size is None:
        batch_size = hp.batch_size
        
    dataset = SudokuDataset(data_type=data_type)
    
    # DataLoader 替代了原来的 tf.train.shuffle_batch
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(data_type == "train"), # 训练集打乱，测试集通常不打乱
        num_workers=4,                  # 多线程读取
        pin_memory=True                 # 如果使用 GPU，建议开启
    )
    
    num_batches = len(loader)
    return loader, num_batches

if __name__ == "__main__":
    # 1. 测试 Dataset
    print("=== Testing Dataset ===")
    dataset = SudokuDataset(data_type="train")
    if len(dataset) > 0:
        q, s = dataset[0]
        print(f"Dataset length: {len(dataset)}")
        print(f"Quiz shape: {q.shape}, dtype: {q.dtype}")
        print(f"Solution shape: {s.shape}, dtype: {s.dtype}")
        print(f"Quiz sample (first row): {q[0]}")
        print(f"Solution sample (first row): {s[0]}")
        print(f"Solution max value: {s.max()}, min value: {s.min()}")

    print("\n" + "="*30 + "\n")

    # 2. 测试 DataLoader
    print("=== Testing DataLoader ===")
    loader, num_batches = get_dataloader(data_type="train", batch_size=4)
    print(f"Total batches: {num_batches}")

    # 取出一个 batch
    for batch_idx, (quizzes, solutions) in enumerate(loader):
        print(f"Batch {batch_idx + 1} shapes:")
        print(f"  Quizzes: {quizzes.shape}")      # 预期 [4, 9, 9]
        print(f"  Solutions: {solutions.shape}")  # 预期 [4, 9, 9]
        
        # 验证数值范围
        # 因为你现在训练用 y - 1，所以这里打印原始 y 看看是不是 1-9
        print(f"  First solution in batch (first row): {solutions[0][0]}")
        
        # 只检查第一个 batch 即可
        break