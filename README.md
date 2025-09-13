# WeatherGen: LiDAR Weather Generation with Diffusion Models

基于扩散模型的 LiDAR 点云天气生成项目，支持雾、雪、雨等天气条件的模拟。

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境
conda env create -f environment.yml
conda activate weathergen

# 检查环境设置
python check_setup.py
```

### 2. 数据准备

**KITTI-360 数据集:**
- 下载地址: https://www.cvlibs.net/datasets/kitti-360/download.php
- 只需要 `KITTI-360--data_3d_raw` 部分
- 解压到 `data/kitti_360/` 目录

**Seeing Through Fog 数据集:**
- 下载地址: https://light.princeton.edu/datasets/automated_driving_dataset  
- 解压到 `data/SeeingThroughFog/` 目录

### 3. 预训练权重 (微调模式需要)

下载预训练权重:
- 链接: https://pan.baidu.com/s/17Z7ZgmTDuJ5thlxIn97PiQ
- 密码: 7788
- 放置到: `logs/diffusion/kitti_360/spherical-1024/20250910T125905/models/`

## 🎯 训练

### 便捷训练脚本

```bash
# 微调模式 (推荐)
python run_training.py --mode finetune

# 从头训练
python run_training.py --mode train

# 启用 LFA (Latent Feature Alignment)
python run_training.py --mode train --lfa

# 调试模式 (更频繁保存)
python run_training.py --mode finetune --debug

# 自定义参数
python run_training.py --mode finetune --lr 5e-5 --steps 50000 --batch-size 4 --gpu 0
```

### 手动训练

```bash
# 使用 accelerate 启动训练
accelerate launch --config_file accelerate_config.yaml train.py \
    --training.train_model finetune \
    --model.lfa false \
    --training.lr 1e-4
```

## 🎨 生成

```bash
# 生成天气点云
python generate.py
```

## 📊 核心功能

### 训练模式
- **train**: 从头开始训练完整模型
- **finetune**: 基于预训练权重进行微调

### 关键技术
- **MDP (Masked Diffusion Process)**: 通过可学习掩码进行天气模拟
- **CLC (Contrastive Learning Control)**: 基于对比学习的天气条件控制
- **LFA (Latent Feature Alignment)**: 潜在特征对齐，使用 VAE 对齐生成数据和真实天气数据

### 支持的天气类型
- 正常天气 (normal)
- 雾天 (fog)  
- 雪天 (snow)
- 雨天 (rain)

## 📁 项目结构

```
├── train.py                 # 主训练脚本
├── generate.py              # 生成脚本
├── run_training.py          # 便捷训练启动脚本
├── check_setup.py           # 环境检查脚本
├── accelerate_config.yaml   # Accelerate 配置
├── environment.yml          # Conda 环境配置
├── models/                  # 模型定义
│   ├── diffusion/          # 扩散模型
│   ├── efficient_unet.py   # EfficientUNet + Mamba
│   └── CLIP/               # CLIP 模型
├── utils/                   # 工具函数
│   ├── weather.py          # 天气处理
│   ├── option.py           # 配置选项
│   └── stf_dataset.py      # STF 数据集
├── data/                    # 数据目录
│   ├── kitti_360/         # KITTI-360 数据
│   └── SeeingThroughFog/   # STF 数据
└── logs/                    # 训练日志和模型权重
```

## 🔧 配置说明

主要配置在 `utils/option.py` 中:

```python
# 训练配置
train_model: "train" | "finetune"  # 训练模式
lr: 4e-4 | 1e-4                   # 学习率
num_steps: 300000 | 100000        # 训练步数
batch_size_train: 8               # 批次大小

# 模型配置  
lfa: bool                         # 是否启用 LFA
architecture: "efficient_unet"    # 模型架构

# 数据配置
dataset: "kitti_360"              # 数据集
projection: "spherical-1024"      # 投影方式
resolution: (64, 1024)           # 分辨率
```

## 📈 生成数据

生成的数据和标签可以在以下链接下载:
- 链接: https://pan.baidu.com/s/1_waBH02ZXpSlEKFA-o5_bw  
- 密码: 7878

标注工具: https://github.com/ch-sa/labelCloud

## 🐛 故障排除

1. **CUDA 内存不足**: 减小 `batch_size_train`
2. **数据路径错误**: 检查 `data/` 目录结构
3. **依赖缺失**: 运行 `python check_setup.py` 检查环境
4. **权重加载失败**: 确认预训练权重路径正确

## 📝 更新日志

- 修复了 `models/efficient_unet.py` 中的拼写错误
- 统一了 `utils/weather.py` 中的路径格式
- 添加了便捷的训练和检查脚本
- 完善了项目文档
