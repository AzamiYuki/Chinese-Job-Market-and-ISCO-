# Chinese-Job-Market-and-ISCO-


##实验环境配置与部署指南

### 硬件环境

**实验平台硬件配置：**
- **操作系统**：Windows 11 专业版 (版本 22H2, Build 22621)
- **处理器**：Intel Core i9-13900K @ 3.0GHz (24核32线程)
- **内存**：32GB DDR5 5600MHz (2×16GB 双通道)
- **显卡**：NVIDIA GeForce RTX 4080 SUPER (16GB GDDR6X显存)
- **存储**：Samsung 990 PRO 2TB NVMe SSD (系统盘) + WD Black SN850X 2TB (数据盘)
- **CUDA版本**：CUDA 12.1 (Driver Version: 537.58)
- **cuDNN版本**：cuDNN 8.9.7 for CUDA 12.x

### 软件环境配置

**基础开发环境：**
```
Python: 3.10.11 (推荐使用Anaconda管理)
PyTorch: 2.1.2+cu121
CUDA Toolkit: 12.1
cuDNN: 8.9.7
Git: 2.42.0.windows.2
Visual Studio Code: 1.85.1
```

**Python环境创建与配置：**

1. **创建虚拟环境**
```bash
# 使用Anaconda创建独立环境
conda create -n isco_hierarchical python=3.10.11
conda activate isco_hierarchical
```

2. **安装PyTorch及CUDA支持**
```bash
# 安装PyTorch 2.1.2 with CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

3. **安装Transformers及相关依赖**
```bash
# 核心依赖包
pip install transformers==4.36.2
pip install datasets==2.16.1
pip install accelerate==0.26.1
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3

# 数据处理相关
pip install pandas==2.1.4
pip install numpy==1.24.3
pip install scikit-learn==1.3.2
pip install openpyxl==3.1.2

# 可视化工具
pip install matplotlib==3.8.2
pip install seaborn==0.13.1
pip install plotly==5.18.0

# 进度条和日志
pip install tqdm==4.66.1
pip install tensorboard==2.15.1
pip install wandb==0.16.2

# 其他工具包
pip install pyyaml==6.0.1
pip install jsonlines==4.0.0
pip install python-dotenv==1.0.0
```

## 模型权重下载与配置

**预训练模型下载：**

1. **创建模型存储目录**
```bash
mkdir -p ./models/bert-base-chinese
mkdir -p ./models/chinese-bert-wwm
mkdir -p ./models/chinese-roberta-wwm-ext
```

2. **下载模型权重**（可选择以下方式之一）

**方式一：使用Hugging Face官方源**
```python
from transformers import AutoModel, AutoTokenizer

# Google BERT Base Chinese
model_name = "bert-base-chinese"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained(f"./models/{model_name}")
tokenizer.save_pretrained(f"./models/{model_name}")

# HFL Chinese BERT-wwm
model_name = "hfl/chinese-bert-wwm"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained("./models/chinese-bert-wwm")
tokenizer.save_pretrained("./models/chinese-bert-wwm")

# HFL Chinese RoBERTa-wwm-ext
model_name = "hfl/chinese-roberta-wwm-ext"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained("./models/chinese-roberta-wwm-ext")
tokenizer.save_pretrained("./models/chinese-roberta-wwm-ext")
```

**方式二：使用ModelScope（国内镜像）**
```python
from modelscope import snapshot_download

# 下载模型到本地
model_dir = snapshot_download('damo/nlp_bert_backbone_base_std', cache_dir='./models/')
model_dir = snapshot_download('damo/nlp_structbert_backbone_base_std', cache_dir='./models/')
```

###项目目录结构

```
isco_hierarchical_classification/
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   │   └── isco_dataset.xlsx      # ISCO职业分类数据集
│   ├── processed/                 # 处理后数据
│   │   ├── train.json            # 训练集
│   │   ├── valid.json            # 验证集
│   │   └── test.json             # 测试集
│   └── cache/                    # 缓存文件
│
├── models/                       # 模型权重目录
│   ├── bert-base-chinese/        # Google BERT
│   ├── chinese-bert-wwm/         # HFL BERT-wwm
│   └── chinese-roberta-wwm-ext/  # HFL RoBERTa
│
├── src/                          # 源代码
│   ├── data_processing.py        # 数据预处理
│   ├── model.py                  # 模型定义
│   ├── train.py                  # 训练脚本
│   ├── evaluate.py               # 评估脚本
│   ├── losses.py                 # 损失函数实现
│   └── utils.py                  # 工具函数
│
├── configs/                      # 配置文件
│   ├── baseline.yaml             # Baseline配置
│   ├── hierarchical.yaml         # Hierarchical配置
│   └── multitask.yaml            # Multitask配置
│
├── experiments/                  # 实验结果
│   ├── logs/                     # 训练日志
│   ├── checkpoints/              # 模型检查点
│   └── results/                  # 实验结果
│
├── notebooks/                    # Jupyter notebooks
│   ├── data_analysis.ipynb       # 数据分析
│   └── results_visualization.ipynb # 结果可视化
│
├── requirements.txt              # 依赖列表
├── README.md                     # 项目说明
└── run_experiments.py            # 主运行脚本
```

### GPU环境验证与优化

**1. 验证CUDA安装**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

**2. GPU内存优化配置**
```python
# 训练配置优化
training_args = {
    "per_device_train_batch_size": 16,  # RTX 4080S 16GB可支持
    "per_device_eval_batch_size": 32,    # 评估时可增大batch
    "gradient_accumulation_steps": 2,     # 梯度累积
    "fp16": True,                        # 混合精度训练
    "dataloader_num_workers": 4,         # 数据加载线程数
    "gradient_checkpointing": True,       # 梯度检查点（节省显存）
}

# 设置环境变量优化性能
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # RTX 4080S架构
```

### 运行环境测试

**创建测试脚本 `test_environment.py`：**
```python
import sys
import torch
import transformers
import numpy as np
import pandas as pd
from packaging import version

def check_environment():
    print("="*50)
    print("实验环境检测报告")
    print("="*50)
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # 核心库版本
    print(f"\n核心库版本:")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    
    # GPU信息
    print(f"\nGPU信息:")
    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 测试GPU计算
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)
        print(f"GPU计算测试: 通过")
    else:
        print(f"CUDA可用: 否")
    
    # 测试模型加载
    print(f"\n模型加载测试:")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        print(f"Tokenizer加载: 通过")
    except Exception as e:
        print(f"Tokenizer加载: 失败 - {e}")
    
    print("="*50)

if __name__ == "__main__":
    check_environment()
```

###  常见问题与解决方案

**1. CUDA版本不匹配**
```bash
# 检查CUDA版本
nvidia-smi
nvcc --version

# 如果版本不匹配，重新安装对应版本的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

**2. 显存不足错误 (OOM)**
```python
# 减小batch size
training_args["per_device_train_batch_size"] = 8

# 启用梯度累积
training_args["gradient_accumulation_steps"] = 4

# 使用DeepSpeed优化
training_args["deepspeed"] = "./configs/ds_config.json"
```

**3. Windows长路径问题**
```python
# 在代码开始处添加
import os
os.system("fsutil file setshortname")

# 或在注册表启用长路径支持
# Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
# LongPathsEnabled = 1
```

### 性能基准测试

**运行性能测试脚本：**
```python
def benchmark_training_speed():
    """测试不同batch size下的训练速度"""
    import time
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained("bert-base-chinese").cuda()
    
    for batch_size in [8, 16, 32]:
        try:
            # 创建随机输入
            input_ids = torch.randint(0, 21128, (batch_size, 256)).cuda()
            
            # 预热
            for _ in range(3):
                _ = model(input_ids)
            
            # 计时
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(10):
                outputs = model(input_ids)
                loss = outputs.last_hidden_state.mean()
                loss.backward()
            
            torch.cuda.synchronize()
            end = time.time()
            
            print(f"Batch Size: {batch_size}, "
                  f"平均耗时: {(end-start)/10:.3f}秒/批次, "
                  f"吞吐量: {batch_size*10/(end-start):.1f}样本/秒")
                  
        except RuntimeError as e:
            print(f"Batch Size: {batch_size} - 显存不足")
```

