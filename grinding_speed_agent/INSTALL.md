# 安装指南

本文档提供详细的安装步骤和环境配置说明。

## 系统要求

### 硬件要求

**最低配置**:
- CPU: 4核心以上
- RAM: 8GB (CPU模式)
- GPU: 无要求 (可选)
- 硬盘: 20GB 可用空间

**推荐配置**:
- CPU: 8核心以上
- RAM: 16GB
- GPU: NVIDIA GPU, 12GB+ 显存 (如 RTX 3060, RTX 4070等)
- 硬盘: 50GB 可用空间 (SSD推荐)

### 软件要求

- **操作系统**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8 - 3.11 (推荐 3.10)
- **CUDA**: 11.8+ (如果使用GPU)
- **Git**: 用于克隆项目 (可选)

## 安装步骤

### 1. 安装Python

#### Windows
1. 访问 https://www.python.org/downloads/
2. 下载Python 3.10安装包
3. 安装时勾选 "Add Python to PATH"
4. 验证安装: `python --version`

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv
```

#### macOS
```bash
# 使用Homebrew
brew install python@3.10
```

### 2. 创建虚拟环境 (推荐)

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 3. 下载项目

```bash
# 如果有Git
git clone <repository-url>
cd grinding_speed_agent

# 或者直接解压下载的zip文件
```

### 4. 安装依赖包

#### GPU模式 (推荐)

如果你有NVIDIA GPU:

```bash
# 安装CUDA版PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

#### CPU模式

如果没有GPU或想使用CPU:

```bash
# 安装CPU版PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

### 5. 下载大模型 (可选)

大模型会在首次使用时自动下载，但你也可以提前下载：

```python
from transformers import AutoModel, AutoTokenizer

# 下载Qwen模型 (约14GB)
model_name = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# 或下载ChatGLM模型 (约12GB)
model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
```

### 6. 配置文件

编辑 `grinding_speed_agent/config/config.yaml`:

**GPU模式配置**:
```yaml
llm:
  model_name: "Qwen/Qwen-7B-Chat"
  device: "cuda"
  quantization:
    enabled: true  # 启用量化节省显存
    bits: 4
```

**CPU模式配置**:
```yaml
llm:
  model_name: "Qwen/Qwen-7B-Chat"
  device: "cpu"
  quantization:
    enabled: false  # CPU模式不需要量化
```

### 7. 验证安装

```bash
# 测试导入
python -c "import torch; print(torch.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
python -c "import xgboost; print(xgboost.__version__)"

# 检查CUDA是否可用 (GPU模式)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 常见安装问题

### 问题 1: pip安装速度慢

**解决方案**: 使用国内镜像源

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 2: transformers安装失败

**解决方案**:
```bash
pip install --upgrade pip setuptools wheel
pip install transformers
```

### 问题 3: CUDA版本不匹配

**解决方案**: 检查CUDA版本并安装对应的PyTorch

```bash
# 检查CUDA版本
nvidia-smi

# 访问 https://pytorch.org/ 选择对应版本
```

### 问题 4: 显存不足

**解决方案**:
1. 启用4-bit量化
2. 减小batch size
3. 使用更小的模型 (如Qwen-1.8B)
4. 使用CPU模式

### 问题 5: Windows上找不到命令

**解决方案**:
```bash
# 确保Python在PATH中
python --version

# 使用完整路径
C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe main.py
```

## 卸载

```bash
# 停用虚拟环境
deactivate

# 删除虚拟环境
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows

# 删除项目文件夹
```

## 更新

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows

# 更新依赖
pip install --upgrade -r requirements.txt

# 或更新特定包
pip install --upgrade transformers
```

## 离线安装

如果需要在无网络环境安装:

1. 在有网络的机器上下载所有依赖:
```bash
pip download -r requirements.txt -d packages/
```

2. 复制 `packages/` 文件夹到离线机器

3. 离线安装:
```bash
pip install --no-index --find-links=packages/ -r requirements.txt
```

## Docker安装 (高级)

```dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py", "--mode", "ui"]
```

构建和运行:
```bash
docker build -t grinding-speed-agent .
docker run -p 8501:8501 grinding-speed-agent
```

## 性能优化建议

1. **使用SSD**: 模型加载速度更快
2. **启用量化**: 减少70%显存占用
3. **使用GPU**: 推理速度提升10-50倍
4. **预下载模型**: 避免首次使用时长时间等待

## 获取帮助

如果遇到安装问题:
1. 查看错误信息
2. 搜索相关错误解决方案
3. 提交Issue并附上详细错误日志
4. 检查Python和依赖包版本

---

安装完成后，请查看 [README.md](README.md) 了解使用方法。
