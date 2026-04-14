# EuroSAT RGB Three-Layer MLP From Scratch

本项目使用纯 `NumPy` 从零实现三层多层感知机（MLP），在 `EuroSAT_RGB` 遥感图像数据集上完成：

- 数据加载与预处理
- 三层 MLP 前向与反向传播
- SGD 训练与学习率衰减
- L2 正则化
- 验证集最优模型保存
- 测试集评估与混淆矩阵
- 超参数搜索
- 训练曲线、第一层权重、错例分析可视化

## 目录结构

```text
hw1/
├── code/
├── results/
│   ├── curves/
│   ├── weights/
│   ├── errors/
│   ├── search/
│   └── checkpoints/
├── EuroSAT_RGB/
├── requirements.txt
└── README.md
```

## 依赖环境

- Python 3.10 或更高
- `numpy`
- `matplotlib`
- `Pillow`

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集假设

- `EuroSAT_RGB/` 已经放在项目根目录下
- 数据按类别子文件夹组织，例如 `EuroSAT_RGB/Forest/*.jpg`
- 代码会自动读取类别名并映射标签

## 训练命令

默认训练：

```bash
python code/train.py
```

自定义超参数训练：

```bash
python code/train.py \
  --image-size 32 \
  --batch-size 128 \
  --epochs 30 \
  --hidden-dim1 256 \
  --hidden-dim2 128 \
  --activation relu \
  --learning-rate 0.01 \
  --lr-decay 0.95 \
  --weight-decay 1e-4 \
  --run-name eurosat_relu_run
```

训练输出：

- 最优模型权重：`results/checkpoints/`
- 训练历史 JSON：`results/curves/`
- loss / accuracy 曲线：`results/curves/`
- 第一层权重可视化：`results/weights/`

## 一键运行

### 方式 1：Shell 脚本

直接完整执行安装依赖、粗搜索、自动读取粗搜索最优组合、细搜索、自动读取细搜索最优组合、正式训练、测试、重绘可视化：

```bash
bash run_all.sh
```

也可以通过环境变量覆盖默认参数：

```bash
RUN_NAME=exp1 EPOCHS=20 IMAGE_SIZE=32 bash run_all.sh
```

说明：

- `RUN_NAME=full_run_v2 bash run_all.sh` 表示只对这一次脚本运行，把实验名设为 `full_run_v2`
- 对应输出会保存为 `results/checkpoints/full_run_v2_best.npz`、`results/curves/full_run_v2_history.json` 等文件
- `run_all.sh` 现在会把完整终端输出保存到 `results/logs/<RUN_NAME>.log`
- 默认会自动清理同名旧结果；如果你想保留旧结果，可以设置 `AUTO_CLEAN=0`
- 默认流程是：先跑一轮粗搜索，再自动读取粗搜索最优组合生成细搜索空间，接着跑细搜索，最后自动读取细搜索最优组合进行正式训练
- 默认粗搜索配置为：`learning rate = {0.03, 0.01, 0.003}`，`hidden dim = {128, 256, 512}`，`weight decay = {0, 1e-4}`，`activation = {relu, tanh}`，共 `36` 组
- 默认细搜索空间会根据粗搜索最优组合自动生成
- 如果你不想让正式训练自动采用细搜索最优组合，可以设置 `TRAIN_FROM_FINE_SEARCH_BEST=0`
- 细搜索默认会围绕粗搜索最优组合自动生成更小的搜索空间：
  - learning rate 约为 `0.5x / 1x / 2x`
  - hidden dim 约为 `0.5x / 1x / 1.5x`
  - weight decay 约为 `0.5x / 1x / 5x`
  - activation 默认固定为粗搜索最优激活函数

示例：

```bash
RUN_NAME=full_run_v2 AUTO_CLEAN=1 bash run_all.sh
RUN_NAME=ablation_relu AUTO_CLEAN=0 SEARCH_NAME=ablation_relu_search bash run_all.sh
RUN_NAME=manual_train TRAIN_FROM_FINE_SEARCH_BEST=0 HIDDEN_DIM1=256 HIDDEN_DIM2=128 LEARNING_RATE=0.01 ACTIVATION=relu bash run_all.sh
RUN_NAME=custom_fine FINE_LEARNING_RATES=0.02,0.01,0.005 FINE_HIDDEN_DIMS=256,384,512 FINE_WEIGHT_DECAYS=1e-5,1e-4,5e-4 FINE_ACTIVATIONS=relu bash run_all.sh
```

### 方式 2：Makefile

完整流水线：

```bash
make all
```

分步骤执行：

```bash
make install
make search RUN_NAME=exp2
make train RUN_NAME=exp2 EPOCHS=20
make test RUN_NAME=exp2
make visualize RUN_NAME=exp2
```

## 超参数搜索

网格搜索：

```bash
python code/search.py --mode grid --epochs 15
```

随机搜索：

```bash
python code/search.py --mode random --epochs 15 --max-trials 12
```

自定义更充分的网格搜索：

```bash
python code/search.py \
  --mode grid \
  --epochs 10 \
  --search-name coarse_grid \
  --learning-rates 0.03,0.01,0.003 \
  --hidden-dims 128,256,512 \
  --weight-decays 0,1e-4 \
  --activations relu,tanh,sigmoid \
  --top-k 10
```

细粒度搜索示例：

```bash
python code/search.py \
  --mode grid \
  --epochs 15 \
  --search-name fine_grid \
  --learning-rates 0.02,0.01,0.005 \
  --hidden-dims 256,384,512 \
  --weight-decays 5e-5,1e-4,5e-4 \
  --activations relu \
  --top-k 10
```

搜索输出：

- 每组实验结果 CSV：`results/search/*_results.csv`
- Top-k 结果 CSV：`results/search/*_top*.csv`
- 搜索摘要 JSON：`results/search/*_summary.json`
- Trial 排名柱状图：`results/search/*_best_val_accuracy.png`
- 按 learning rate / hidden dim / weight decay / activation 分组统计的 CSV 和图
- 学习率-隐藏层、正则-隐藏层 heatmap 图

搜索参数覆盖：

- learning rate
- hidden dimension
- regularization strength
- activation function

建议的课程作业搜索方案：

- 粗搜索：`3 x 3 x 2 x 2 = 36` 组，`10` epoch
- 细搜索：围绕粗搜索前几名再做 `9-12` 组，`15` epoch
- 最终模型：用最优组合再完整训练 `30` epoch

## 测试命令

使用最佳模型在测试集评估：

```bash
python code/test.py --checkpoint results/checkpoints/your_best_model.npz
```

测试输出：

- 测试准确率
- 混淆矩阵 JSON
- 混淆矩阵图片
- 错分样本可视化图

保存位置：

- `results/errors/*_summary.json`
- `results/errors/*_confusion_matrix.png`
- `results/errors/*_misclassified.png`

## 单独生成可视化

如果已经有训练历史或 checkpoint，可以重新画图：

```bash
python code/visualize.py \
  --history results/curves/your_run_history.json \
  --checkpoint results/checkpoints/your_best_model.npz \
  --test-summary results/errors/your_test_summary.json \
  --run-name redraw
```

## 实现说明

### 1. 数据预处理

- 自动遍历 `EuroSAT_RGB/` 下的 10 个类别子文件夹
- `EuroSAT_RGB` 原始图像尺寸为 `64x64`
- 训练前支持图像 resize，当前默认实验使用 `32x32` 作为 MLP 输入尺寸
- 将 RGB 图像归一化到 `[0, 1]`
- 将图像展平为 MLP 输入向量
- 按固定随机种子做分层 `train / val / test` 划分
- 使用训练集均值与标准差对各划分做标准化

### 2. 模型结构

MLP 结构如下：

```text
input -> hidden1 -> activation -> hidden2 -> activation -> output
```

- 可切换激活函数：`relu` / `sigmoid` / `tanh`
- 系统化超参数搜索可直接比较这三种激活函数的验证集表现
- 手工实现反向传播
- 不依赖 PyTorch、TensorFlow、JAX 或任何自动微分框架

### 3. 损失与优化

- softmax + cross entropy
- L2 regularization
- SGD optimizer
- learning rate decay

### 4. 报告支持内容

本项目已实现并保存以下报告所需结果：

- 训练集 loss 曲线
- 验证集 loss 曲线
- 验证集 accuracy 曲线
- 第一层权重可视化
- 测试错例图
- confusion matrix
- 超参数搜索结果表

## 模型保存与加载

checkpoint 采用 `.npz` 保存，内容包括：

- 模型权重与偏置
- 类别名
- 训练时使用的输入图像尺寸
- 训练配置
- 输入维度
- 标准化相关元信息

加载方式：

```bash
python code/test.py --checkpoint results/checkpoints/your_best_model.npz
```

## 合理假设

- 默认 `EuroSAT_RGB` 中图片可全部读入内存；对 EuroSAT 规模通常可行
- 原始 EuroSAT RGB 图像为 `64x64`，默认下采样到 `32x32` 以平衡训练速度与分类效果
- 超参数搜索默认每个 trial 使用较少 epoch，以控制总运行时间

## 建议提交内容

建议最终提交时附上：

- `code/` 全部源码
- `results/` 中最优模型与主要图表
- 实验报告中引用 `results/` 下生成的图片和 JSON/CSV 结果
