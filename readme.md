## Todolist

- 使用命令行参数指导

## 整体介绍

我们搜索了网络已有项目，选择 `https://github.com/Kyubyong/sudoku` 的代码作为重要参考。

在我们看来，该项目缺点有以下几个：

- 项目过于久远，甚至是 python2 + tensorflow 1.1.0，运行起来遇到众多麻烦；
- 没有对于使用 CNN 解决数独问题的思路与原理详细描述；
- 没有对超参数与架构的调整进行探讨；
- 训练数据集随机生成，没有经过筛选；
- 测试仅以准确率 (Acc) 作为参考标准，但没有考虑多解情况。

首先我们大幅调整了代码以适配 python3+pytorch，并在改动后使用原本的数据集 `sudoku.csv` 和测试集 `test.csv` 进行了训练与测试，得到了 `results/output_origin.txt`，大致与原模型的 Acc 率相当。

并在下面对尝试对训练数据集、架构、超参数、测试等进行改进。

我们在 `test.py` 中添加了完美率 (Perfect Rate, PR) 的测试，使用独立代码逻辑检测模型的输出是否符合数独逻辑，并在我们组织的新测试集上运行测试，得到了 `output_orogin_model_with_new_test.txt`，Acc, PR 分别为 0.6790, 0.2504。

目录下的 `data` 和 `logdir` 未上传，`data` 用于存放训练与测试数据，`logdir` 用于存放训练好的模型。

`data` 内部的文件如下：

- `full_test_set_10000.csv`
- `full_train_set_2990000.csv`
- `sudoku.csv`
- `test.csv`

前二者是我们自行搜集整理的数据集，后二者是原项目的数据集。

在这之后，我们对模型训练流程作如下调整：

- 使用了从可靠网站上下载并自行筛选整理的非随机高质量数据集 `full_train_set_2990000.csv` 用于训练；
- 根据经验，我们认为添加 ResNet 设计会显著提高性能，我们考虑添加 `ResBlock` 后将层数从 10 层增加到 20 层；
- 为适配现代显卡性能，将 `batch_size` 调整为 `1024`。


## 环境配置

标准测试环境为 Ubuntu 22.04 LTS, RTX 4090, cuda 12.8, pytorch 2.9.1。

需要安装 `tqdm + pytorch`:

```bash
# 如可以，最好新建一个单独环境来运行代码
# conda create -n sudoku python=3.12
# conda activate sudoku

# 此外，可能需要检查本地的 cuda 版本来安装对应的 pytorch
# 此处需要自行参考 pytorch 的官网文档。标准测试环境是 RTX 4090
pip3 install torch torchvision
pip3 install tqdm
```

## 运行指令

调整 `hyperparams.py` 指定训练集和测试集，依次执行 `train.py` 和 `test.py`。

## 运行结果

Acc 率和完美率分别达到 70%，30%。