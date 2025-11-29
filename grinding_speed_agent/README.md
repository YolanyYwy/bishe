# 研磨速度预测 AI Agent

一个集成轻量级大模型与传统机器学习模型的智能预测系统，用于研磨速度的自动化预测分析。

## 项目简介

本项目实现了一个智能Agent框架，结合以下技术：

- **轻量级大模型**: Qwen-7B / ChatGLM3-6B (本地部署)
- **传统ML模型**: RandomForest, XGBoost, LightGBM, GradientBoosting, SVR
- **Web UI**: Streamlit交互式界面
- **自动报告**: Markdown格式的专业分析报告

## 核心特性

### 🤖 智能Agent
- 自然语言理解用户意图
- 自动协调多个预测模型
- 智能选择最优模型



### 📊 多模型集成
- 自动训练5种主流机器学习模型
- 交叉验证评估模型性能
- 自动选择最佳模型

### 🎨 友好UI界面
- 基于Streamlit的Web界面
- 数据上传与预览
- 实时可视化分析
- 交互式对话功能

### 📄 自动报告生成
- Markdown格式专业报告
- 包含模型性能对比
- 特征重要性分析
- 改进建议

## 项目结构

```
grinding_speed_agent/
├── agent/                      # Agent核心模块
│   ├── __init__.py
│   └── llm_agent.py           # Agent主逻辑
├── models/                     # ML模型模块
│   ├── __init__.py
│   └── ml_models.py           # 传统ML模型管理
├── llm/                        # 大模型模块
│   ├── __init__.py
│   └── local_llm.py           # 本地大模型封装
├── ui/                         # UI界面
│   ├── __init__.py
│   └── streamlit_app.py       # Streamlit应用
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── data_processor.py      # 数据处理
│   └── report_generator.py    # 报告生成
├── config/                     # 配置文件
│   └── config.yaml            # 主配置
├── data/                       # 数据目录
├── models_saved/               # 保存的模型
├── reports/                    # 生成的报告
├── requirements.txt            # 依赖包
├── main.py                     # 主入口
└── README.md                   # 本文件
```

## 快速开始

### 1. 环境准备

**系统要求**:
- Python 3.8+
- CUDA 11.8+ (推荐，用于GPU加速)
- 至少 8GB RAM (CPU模式) 或 12GB GPU显存 (GPU模式)

### 2. 安装依赖

```bash
# 克隆或下载项目后，进入项目目录
cd grinding_speed_agent

# 安装依赖
pip install -r requirements.txt
```

**注意**: 如果使用CPU模式，可以安装CPU版本的PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 配置模型

编辑 `config/config.yaml` 文件：

```yaml
llm:
  model_name: "Qwen/Qwen-7B-Chat"  # 或 "THUDM/chatglm3-6b"
  device: "cuda"  # 或 "cpu"
  quantization:
    enabled: true  # 4-bit量化，节省显存
    bits: 4
```

### 4. 运行方式

#### 方式1: Streamlit UI (推荐)

```bash
# 启动Web界面
python -m grinding_speed_agent.main --mode ui

# 或直接运行
streamlit run grinding_speed_agent/ui/streamlit_app.py
```

然后在浏览器中打开 http://localhost:8501

#### 方式2: 命令行模式

```bash
# 完整流程（数据分析 + 模型训练 + 报告生成）
python main.py --mode pipeline --data path/to/your/data.csv

# 仅训练模型
python main.py --mode train --data path/to/your/data.csv

# 仅预测
python main.py --mode predict --data path/to/your/data.csv

# 生成报告
python main.py --mode report
```

## 使用指南

### 数据格式要求

输入数据应为CSV或Excel格式，包含：
- 特征列：研磨相关的各种参数
- 目标列：研磨速度（通常为最后一列）

示例数据格式：

| 参数1 | 参数2 | 参数3 | ... | 研磨速度 |
|-------|-------|-------|-----|----------|
| 10.5  | 20.3  | 15.7  | ... | 50.2     |
| 12.1  | 18.9  | 16.2  | ... | 48.5     |

### Streamlit UI 使用流程

1. **初始化Agent**: 点击侧边栏的"初始化Agent"按钮
2. **数据分析**: 上传数据文件，查看数据质量和统计信息
3. **模型训练**: 选择目标列，开始训练多个模型
4. **查看结果**: 查看模型性能对比和特征重要性
5. **数据预测**: 上传新数据进行预测
6. **生成报告**: 生成完整的Markdown分析报告

### 命令行使用示例

```bash
# 示例1: 训练模型
python main.py --mode train --data data/grinding_data.csv

# 示例2: 使用训练好的模型进行预测
python main.py --mode predict --data data/new_data.csv

# 示例3: 完整流程
python main.py --mode pipeline --data data/grinding_data.csv
```

## 模型说明

### 支持的ML模型

1. **RandomForest**: 随机森林，适合处理非线性关系
2. **XGBoost**: 梯度提升树，高性能
3. **LightGBM**: 轻量级梯度提升，训练速度快
4. **GradientBoosting**: sklearn梯度提升
5. **SVR**: 支持向量回归，适合小数据集

系统会自动训练所有模型并选择性能最优的模型。

### 评估指标

- **R² Score**: 决定系数，越接近1越好
- **RMSE**: 均方根误差，越小越好
- **MAE**: 平均绝对误差，越小越好
- **交叉验证**: 5折交叉验证评估模型稳定性

## 配置说明

### 模型配置 (config/config.yaml)

```yaml
# 大模型配置
llm:
  model_name: "Qwen/Qwen-7B-Chat"
  device: "cuda"
  quantization:
    enabled: true
    bits: 4

# 机器学习模型配置
ml_models:
  algorithms:
    - RandomForest
    - XGBoost
    - LightGBM
  hyperparameters:
    RandomForest:
      n_estimators: 100
      max_depth: 10

# 数据处理配置
data:
  test_size: 0.2
  validation_size: 0.1
  feature_engineering:
    enabled: true
    interaction_features: true

# 报告配置
report:
  include_visualizations: true
  include_feature_importance: true
```

## 输出说明

### 模型文件

训练好的模型保存在 `grinding_speed_agent/models_saved/` 目录:
- `{ModelName}.pkl`: 各个模型文件
- `scaler.pkl`: 数据标准化器
- `metadata.pkl`: 模型元数据

### 报告文件

报告保存在 `grinding_speed_agent/reports/` 目录:
- `grinding_speed_prediction_report_{timestamp}.md`: Markdown报告
- `model_comparison.png`: 模型性能对比图
- `feature_importance.png`: 特征重要性图
- `predictions.csv`: 预测结果

## 常见问题

### Q1: 显存不足怎么办？

**A**: 有几种解决方案：
1. 启用4-bit量化（在config.yaml中设置）
2. 使用CPU模式（设置device: "cpu"）
3. 使用更小的模型（如Qwen-1.8B）

### Q2: 模型训练很慢？

**A**:
- 确保使用GPU模式
- 减少数据量或特征数量
- 在config中减少模型数量
- 调整超参数（如n_estimators）

### Q3: 如何只使用传统ML模型，不加载大模型？

**A**:
- 使用命令行模式，大模型只在需要时才加载
- 或者在UI中避免使用"智能对话"功能

### Q4: 预测精度不理想？

**A**:
1. 检查数据质量（缺失值、异常值）
2. 增加训练数据量
3. 启用特征工程
4. 调整模型超参数
5. 查看报告中的改进建议

## 技术栈

- **深度学习框架**: PyTorch, Transformers
- **机器学习**: scikit-learn, XGBoost, LightGBM
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn, plotly
- **Web框架**: Streamlit
- **配置管理**: PyYAML

## 性能优化建议

1. **GPU加速**: 使用NVIDIA GPU可显著提升大模型性能
2. **量化**: 4-bit量化可减少70%的显存占用
3. **数据预处理**: 提前清洗数据可加快训练速度
4. **模型选择**: 对于小数据集，可以只使用部分模型

## 扩展开发

### 添加新的ML模型

在 `models/ml_models.py` 中的 `get_model` 方法添加新模型：

```python
model_map = {
    'RandomForest': RandomForestRegressor,
    'YourNewModel': YourNewModelClass,  # 添加这里
    ...
}
```

### 自定义报告模板

修改 `utils/report_generator.py` 中的 `_build_report` 方法。

### 更换大模型

在 `config/config.yaml` 中修改 `model_name`:
```yaml
llm:
  model_name: "THUDM/chatglm3-6b"  # 或其他兼容模型
```

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue。

---

**Made with ❤️ for Grinding Speed Prediction**
