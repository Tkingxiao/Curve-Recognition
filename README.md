# 曲线检测与生成工具

这个项目包含曲线生成、模型训练和曲线检测等功能，可用于生成各种类型的曲线图像、训练曲线检测模型以及对图像中的曲线进行识别。

## 功能说明

1. **曲线生成**：可生成基础函数曲线、特殊函数曲线、物理/工程曲线、经济/金融曲线、分形曲线、组合曲线等多种类型曲线，并将其保存为图像
2. **模型训练**：使用CNN模型训练曲线检测器，可识别图像中是否包含曲线
3. **曲线检测**：对测试图像进行预测，判断是否包含曲线并分类保存

## 环境依赖

- Python 3.x
- 所需Python库：
  - numpy
  - matplotlib
  - opencv-python
  - tensorflow/keras
  - scipy
  - argparse
  - shutil
  - datetime

可通过pip安装所需依赖：
```bash
pip install numpy matplotlib opencv-python tensorflow scipy argparse shutil datetime
```

## 使用方法

### 1. 曲线生成

曲线生成功能位于`curve_seed.py`中，可通过调用相关函数生成不同类型的曲线。

示例代码：
```python
from curve_seed import CurveGenerator  # 假设存在此类封装

# 创建生成器实例
generator = CurveGenerator(output_dir="generated_curves")

# 生成随机类型曲线
x, y, curve_type, _, _ = generator.generate_curve()

# 绘制并保存曲线
file_path = generator.plot_curve(x, y, curve_type)
print(f"曲线已保存至: {file_path}")
```

可生成的曲线类型包括：
- 基础函数（正弦、余弦、正切、指数、对数等）
- 特殊函数（贝塞尔函数、误差函数、伽马函数等）
- 物理/工程曲线（阻尼振动、热传导、流体流动等）
- 经济/金融曲线
- 分形曲线
- 组合曲线

### 2. 模型训练

使用`train.py`脚本训练曲线检测模型：

```bash
# 基本用法
python train.py

# 自定义参数
python train.py --train_dir "path/to/training/images" --epochs 100 --batch_size 32 --model_save_path "model/custom_model.h5"
```

参数说明：
- `--train_dir`：训练图像所在目录，默认为"train/"
- `--epochs`：训练轮数，默认为50
- `--batch_size`：批次大小，默认为16
- `--model_save_path`：模型保存路径，默认为"model/curve_detector.h5"

训练完成后，会生成模型文件和训练历史图像。

### 3. 曲线检测预测

使用`predict.py`脚本对图像进行曲线检测：

```bash
# 基本用法
python predict.py

# 自定义参数
python predict.py --test_dir "path/to/test/images" --model_path "model/custom_model.h5" --confidence_threshold 0.7
```

参数说明：
- `--test_dir`：测试图像所在目录，默认为"test/"
- `--model_path`：预训练模型路径，默认为"model/curve_detector.h5"
- `--confidence_threshold`：置信度阈值，用于判断是否为曲线，默认为0.5

预测结果：
- 图像会被分类保存到"goal/true/"（含曲线）和"goal/false/"（不含曲线）目录
- 生成CSV格式的置信度报告，保存到"log/"目录
- 在控制台输出预测分析报告

### 4. 图片格式转换（附加功能）

使用`convert_images.py`进行图片格式转换：

```bash
python convert_images.py
```

运行后会打开一个GUI界面，选择文件夹后可将其中的图片转换为PNG格式。支持的输入格式包括：JPG、JPEG、WEBP、PNG、BMP、GIF、TIFF。

## 项目结构

- `curve_seed.py`：曲线生成核心代码
- `model.py`：CNN模型定义与相关功能
- `train.py`：模型训练脚本
- `predict.py`：曲线检测与预测脚本
- `convert_images.py`：图片格式转换工具
- `model/`：存放训练好的模型
- `train/`：存放训练图像（需自行创建）
- `test/`：存放测试图像（需自行创建）
- `goal/`：存放预测结果图像
- `log/`：存放预测报告

## 注意事项

- 训练模型前，请确保`train/`目录中包含足够的曲线图像
- 预测前，请确保`test/`目录中包含待检测的图像
- 模型性能可能受训练数据质量和数量的影响，建议使用多样化的曲线图像进行训练
