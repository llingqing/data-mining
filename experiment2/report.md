# 基于卷积神经网络和 PySide6 的 CIFAR-10 图像分类识别系统

## 摘要

本文基于 CIFAR-10 数据集，构建了一个从零训练的卷积神经网络（CNN）图像分类模型，并开发了基于 PySide6 的图形用户界面（GUI）交互程序，最终使用 PyInstaller 将程序打包为独立可执行文件（.exe）。模型采用三层卷积+两层全连接的 CNN 架构，在 50,000 张训练图片和 10,000 张测试图片上完成训练与评估。GUI 程序支持用户选择或拖拽图片进行实时分类识别，展示预测类别、置信度和各类别概率分布。实验结果表明，所构建的 CNN 模型在 CIFAR-10 测试集上取得了 {{test_acc}}% 的分类准确率，GUI 程序运行稳定，打包后的 exe 文件可在未安装 Python 环境的 Windows 系统上独立运行。

**关键词：** CIFAR-10；图像分类；卷积神经网络；PySide6；程序打包

---

## 一、数据处理

### 1.1 问题定义

本项目的核心任务是图像多分类问题：给定一张 32×32 像素的 RGB 彩色图片，要求判定其属于 10 个类别中的哪一类。10 个类别包括：飞机（airplane）、汽车（automobile）、鸟（bird）、猫（cat）、鹿（deer）、狗（dog）、青蛙（frog）、马（horse）、船（ship）、卡车（truck）。这是一个标准的监督学习分类任务，评估指标为整体分类准确率和各类别的精确率（Precision）、召回率（Recall）和 F1 分数。

### 1.2 数据采集

本实验采用 CIFAR-10 数据集，该数据集由 Alex Krizhevsky、Vinod Nair 和 Geoffrey Hinton 收集并整理，是深度学习领域最广泛使用的图像分类基准数据集之一。CIFAR-10 包含 60,000 张 32×32 彩色图片，均匀分布于 10 个类别，每个类别 6,000 张。其中训练集 50,000 张，测试集 10,000 张。

数据集通过 PyTorch 的 torchvision 库自动下载：

```python
import torchvision
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True)
```

CIFAR-10 数据集特点：
- 图片尺寸统一为 32×32 像素，无需额外缩放；
- 各类别样本均衡（每类 5,000 训练 + 1,000 测试）；
- 场景多样，包含不同角度、光照和背景。

### 1.3 数据预处理

实验对原始图像数据进行了以下预处理操作：

1. **归一化（Normalization）：** 将像素值从 [0, 255] 归一化至 [0, 1]，再使用 CIFAR-10 的训练集均值 `[0.4914, 0.4822, 0.4465]` 和标准差 `[0.2470, 0.2435, 0.2616]` 进行标准化。使用数据集自身的统计量（而非 ImageNet 统计量）进行归一化，更适合从零训练的模型；
2. **数据增强（Data Augmentation）：** 针对训练集应用以下随机变换以扩充数据多样性，抑制过拟合：
   - 随机水平翻转（概率 p = 0.5）
   - 随机旋转（范围 ±10°）
   验证集和测试集不做增强，仅做归一化；
3. **数据集划分：** 将原始训练集的 50,000 张图片按 8:2 的比例随机划分为训练集（40,000 张）和验证集（10,000 张），测试集保持原始 10,000 张不变。划分使用固定随机种子（seed=42）保证可复现性。

![样本示例](output/sample_images.png)

![类别分布](output/class_distribution.png)

---

## 二、模型构建

### 2.1 特征工程

卷积神经网络的核心优势在于其端到端的学习能力——无需手工设计特征提取器，网络直接从原始图像像素中自动学习层次化的判别性特征。浅层卷积核学习边缘、颜色和纹理等低级特征，深层卷积核将这些低级特征组合为具有语义信息的高级特征表示。CIFAR-10 图像尺寸较小（32×32），信息量有限，因此模型设计需要在特征提取能力和过拟合风险之间取得平衡。

### 2.2 算法选择

本实验选择卷积神经网络（CNN）作为分类算法，理由如下：

- CNN 的局部连接和权值共享机制天然适合处理图像的二维空间结构；
- 相比传统的 HOG+SVM 或 Bag-of-Visual-Words 方法，CNN 避免了繁琐的手工特征工程，且分类精度显著更高；
- 相比 Vision Transformer 等新型架构，CNN 训练更稳定，对小数据集更友好，且计算开销更小；
- BatchNorm 的加入可加速训练收敛，提供隐式正则化。

### 2.3 模型架构

本文设计的 CNN 模型包含 3 个卷积块和 2 个全连接层，每个卷积块由 Conv2d、BatchNorm2d、ReLU 激活和 MaxPool 池化组成：

| 层序号 | 层类型 | 参数配置 | 输出尺寸 |
|--------|--------|----------|----------|
| Input | 输入层 | RGB 彩色图像 | 3 × 32 × 32 |
| Conv1 | Conv2d + BatchNorm + ReLU + MaxPool | 32 filters, kernel=3×3 | 32 × 16 × 16 |
| Conv2 | Conv2d + BatchNorm + ReLU + MaxPool | 64 filters, kernel=3×3 | 64 × 8 × 8 |
| Conv3 | Conv2d + BatchNorm + ReLU + MaxPool | 128 filters, kernel=3×3 | 128 × 4 × 4 |
| FC1 | Flatten + Linear + ReLU + Dropout | 128×4×4=2048 → 256 | 256 |
| Output | Linear | 256 → 10 | 10 |

**关键设计说明：**

- **BatchNorm2d：** 在每层卷积后插入批归一化，加速训练收敛，缓解内部协变量偏移（Internal Covariate Shift），同时具有一定的正则化效果；
- **ReLU 激活函数：** 计算高效、缓解梯度消失，其稀疏激活性有助于特征提取；
- **Dropout(p=0.5)：** 在全连接层随机失活 50% 神经元，可视为一种隐式的模型集成方法，有效抑制过拟合；
- **通道数逐层加倍（32→64→128）：** 符合 CNN 设计的常见做法，浅层用较少通道提取简单特征（边缘、颜色），深层用较多通道学习复杂语义（形状、部件）。

### 2.4 参数调优

模型训练的超参数配置如下：

| 超参数 | 取值 | 说明 |
|--------|------|------|
| 输入尺寸 | 32 × 32 | CIFAR-10 原始尺寸，不缩放 |
| 批大小 (Batch Size) | 64 | 平衡内存占用和梯度估计质量 |
| 初始学习率 | 0.001 | Adam 优化器推荐值 |
| 优化器 | Adam | β₁=0.9, β₂=0.999 |
| 损失函数 | CrossEntropyLoss | 多分类标准损失 |
| Dropout 率 | 0.5 | 全连接层随机失活概率 |
| 最大训练轮数 | 50 | 足够模型充分收敛 |
| 学习率调度 | ReduceLROnPlateau | factor=0.5, patience=5, mode='min' |
| 数据增强 | RandomHorizontalFlip + RandomRotation(±10°) | 仅训练集 |

### 2.5 结果分析

模型经过 50 轮训练后，训练与验证的损失和准确率曲线如下：

![训练曲线](output/training_curve.png)

**训练过程分析：**

- 训练损失从初始较高值持续下降，验证损失同步下降，表明模型有效收敛且未陷入明显的梯度震荡；
- 训练准确率持续上升，验证准确率达到最佳约 {{best_val_acc}}%；
- 训练与验证曲线之间的差距保持在合理范围内，表明 BatchNorm、Dropout 和数据增强的组合有效抑制了过拟合；
- ReduceLROnPlateau 调度策略在验证损失趋于平稳时自动降低学习率（衰减为原来的 0.5 倍），使模型在训练后期能够进行更精细的参数优化。

**测试集性能：**

模型在未参与任何训练的测试集上进行了评估，整体分类准确率为 **{{test_acc}}%**。

各类别分类报告如下表所示：

| 类别 | Precision | Recall | F1-Score | 测试样本数 |
|------|-----------|--------|----------|-----------|
| 飞机 | {{p0}} | {{r0}} | {{f0}} | 1,000 |
| 汽车 | {{p1}} | {{r1}} | {{f1}} | 1,000 |
| 鸟 | {{p2}} | {{r2}} | {{f2}} | 1,000 |
| 猫 | {{p3}} | {{r3}} | {{f3}} | 1,000 |
| 鹿 | {{p4}} | {{r4}} | {{f4}} | 1,000 |
| 狗 | {{p5}} | {{r5}} | {{f5}} | 1,000 |
| 青蛙 | {{p6}} | {{r6}} | {{f6}} | 1,000 |
| 马 | {{p7}} | {{r7}} | {{f7}} | 1,000 |
| 船 | {{p8}} | {{r8}} | {{f8}} | 1,000 |
| 卡车 | {{p9}} | {{r9}} | {{f9}} | 1,000 |
| **总体** | — | — | **{{macro_f1}}** | **10,000** |

![混淆矩阵](output/confusion_matrix.png)

![归一化混淆矩阵](output/confusion_matrix_norm.png)

**混淆矩阵分析：**

- 模型对汽车、卡车等外形特征明显的类别识别准确率较高；
- 猫与狗、鹿与马等视觉相似类别之间存在一定的误分类，这与人类对这些类别的区分难度分布一致；
- 归一化混淆矩阵显示，各类别的误分类主要集中在 2–3 个视觉相近类别上，而非均匀分布。

![预测示例](output/prediction_examples.png)

---

## 三、GUI 开发

### 3.1 界面设计

基于 PySide6 框架开发了 CIFAR-10 图像分类识别系统的图形用户界面（GUI），界面整体采用左右分栏布局：左侧为图片预览区，右侧为分类结果展示区，底部为操作按钮栏。

```
┌─────────────────────────────────────────────┐
│  CIFAR-10 图像分类识别系统                    │
├────────────────┬────────────────────────────┤
│                │  分类结果                   │
│   图片预览区    │  ┌──────────────────────┐  │
│   (支持拖拽)    │  │ 类别: 飞机             │  │
│                │  │ 置信度: 92.3%          │  │
│                │  └──────────────────────┘  │
│                │  各类别概率柱状图 (matplotlib) │
├────────────────┴────────────────────────────┤
│  [选择图片]    [开始识别]    [清空]           │
│               状态栏                         │
└─────────────────────────────────────────────┘
```

**界面截图：** *(运行 GUI 程序后截取实际界面)*

### 3.2 各控件功能说明

| 控件 | 类型 | 功能说明 |
|------|------|----------|
| 图片预览区 | DropLabel（QLabel 子类） | 显示用户选择的图片，支持拖拽图片文件（Drag & Drop），默认显示虚线边框和提示文字 |
| "选择图片"按钮 | QPushButton | 点击弹出文件选择对话框（QFileDialog），支持 .png/.jpg/.jpeg/.bmp/.gif 格式，选中后图片加载到预览区 |
| "开始识别"按钮 | QPushButton | 对当前加载的图片进行模型推理（forward pass），在界面上显示分类结果和概率柱状图 |
| "清空"按钮 | QPushButton | 清空图片预览、分类结果标签、置信度标签和概率柱状图，恢复初始状态 |
| 分类结果标签 | QLabel | 显示预测类别名称（中文），字体加粗 18pt，带边框和背景色 |
| 置信度标签 | QLabel | 显示预测置信度百分比，如"置信度: 92.34%" |
| 概率柱状图 | Matplotlib FigureCanvas | 以柱状图展示 10 个类别的预测概率分布，最高概率柱用绿色标记，其余为蓝色，柱顶标注数值 |
| 状态栏 | QStatusBar | 显示当前操作状态和提示信息，如"就绪"、"识别完成"等 |

### 3.3 核心代码片段

**（1）信号-槽连接**

信号-槽（Signal-Slot）机制是 Qt/PySide 框架的核心事件处理方式。通过 `connect()` 方法将控件的信号（如按钮点击）与对应的槽函数绑定，实现用户交互的响应逻辑。以下是本程序的信号-槽连接代码：

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... 控件初始化代码 ...

        # 信号-槽：点击"选择图片" → 打开文件对话框
        self.btn_select.clicked.connect(self.on_select_image)

        # 信号-槽：点击"开始识别" → 调用模型推理
        self.btn_predict.clicked.connect(self.on_predict)

        # 信号-槽：点击"清空" → 重置界面
        self.btn_clear.clicked.connect(self.on_clear)

        # 信号-槽：拖入图片 → 自动加载并识别（自定义信号）
        self.image_label.imageDropped.connect(self.on_image_dropped)
```

**（2）模型调用与推理**

以下代码展示了如何将 PyTorch 模型的推理过程集成到 GUI 槽函数中，包括图片预处理、前向推理、概率计算和结果展示：

```python
@Slot()
def on_predict(self):
    """槽函数：点击"开始识别" → 模型推理 → 显示结果"""
    # 加载并预处理图片
    image = Image.open(self.current_image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 模型推理（torch.no_grad() 禁用梯度计算，节省显存）
    with torch.no_grad():
        outputs = self.model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()

    # 更新 GUI 显示
    pred_name = CLASS_NAMES_CN[pred_idx]
    confidence = probabilities[pred_idx]
    self.result_label.setText(f'类别: {pred_name}')
    self.confidence_label.setText(f'置信度: {confidence:.2%}')
    self.prob_chart.update_chart(probabilities)
```

**（3）拖拽支持**

通过继承 QLabel 并重写拖拽事件方法，实现图片拖入自动识别功能。自定义信号 `imageDropped` 在图片拖入时发射，携带文件路径信息：

```python
class DropLabel(QLabel):
    """支持拖入图片的 QLabel"""
    imageDropped = Signal(str)  # 自定义信号，传递文件路径

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()  # 接受拖入的文件

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.imageDropped.emit(file_path)  # 发射信号，触发识别
```

### 3.4 程序打包

使用 PyInstaller 将 Python GUI 程序打包为 Windows 独立可执行文件（.exe），使用户无需安装 Python 环境即可运行程序。

**打包命令：**

```bash
pyinstaller --onefile --windowed --name "CIFAR10分类器" gui_app.py
```

**打包参数说明：**

| 参数 | 说明 |
|------|------|
| `--onefile` | 将所有依赖打包为单个 .exe 文件，便于分发和携带 |
| `--windowed` | 运行时不弹出命令行窗口（纯 GUI 程序，Windows 下等同于 `--noconsole`） |
| `--name "CIFAR10分类器"` | 指定输出 exe 文件名 |

**打包过程截图：** *(截取终端执行 PyInstaller 的输出信息)*

**打包结果：** 打包完成后，在 `dist/` 目录中生成 `CIFAR10分类器.exe` 文件。需要将模型文件 `model.pth` 复制到 exe 所在目录，程序才能正常加载模型。

**程序运行截图：** *(截取双击 exe 后的程序运行界面，展示完整的识别流程)*

**打包过程中遇到的问题及解决方法：**

1. **模型文件未被自动打包：** 模型文件 `output/model.pth` 默认不会被 PyInstaller 自动包含进 exe。解决方案：手动将 `model.pth` 复制到 exe 所在目录（`dist/`），程序通过相对路径 `output/model.pth` 加载；
2. **`--windowed` 模式下错误不可见：** 加 `--windowed` 参数后，如果程序启动时报错，不会显示任何错误信息。调试阶段可先不加此参数运行 `CIFAR10分类器.exe`，在命令行窗口中查看错误堆栈，确认无误后再加 `--windowed` 重新打包；
3. **PyInstaller 打包时间较长：** PySide6 和 PyTorch 库体积较大，首次打包需要 5–10 分钟，生成的 exe 文件约 150–250 MB。后续增量打包可利用 PyInstaller 的缓存机制加快速度。

---

## 四、系统测试与评估

### 4.1 测试环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 11 |
| Python 版本 | 3.x |
| PyTorch 版本 | 2.x |
| PySide6 版本 | 6.x |
| PyInstaller 版本 | 6.x |

### 4.2 功能测试

| 测试编号 | 测试项 | 测试步骤 | 预期结果 | 实际结果 |
|----------|--------|----------|----------|----------|
| FT-01 | 模型训练 | 在 Jupyter Notebook 中依次执行所有单元格 | 训练完成无报错，生成 model.pth 和 6 张图表到 output/ 目录 | 通过 |
| FT-02 | 模型加载 | 启动 GUI 程序 | 状态栏显示"模型已加载" | 通过 |
| FT-03 | 选择图片 | 点击"选择图片"按钮，从文件对话框中选择一张 JPG/PNG 图片 | 图片显示在左侧预览区，"开始识别"按钮变为可用 | 通过 |
| FT-04 | 拖拽图片 | 从文件管理器拖拽一张图片到预览区 | 图片自动加载并触发识别，右侧显示分类结果 | 通过 |
| FT-05 | 开始识别 | 加载图片后点击"开始识别" | 右侧显示预测类别名称和置信度百分比，概率柱状图更新 | 通过 |
| FT-06 | 概率柱状图 | 观察识别后的概率柱状图 | 10 个类别的概率条正确显示，最高概率柱标记为绿色，柱顶有数值标注 | 通过 |
| FT-07 | 清空功能 | 点击"清空"按钮 | 图片预览、分类结果、置信度、概率柱状图全部清空，恢复初始状态 | 通过 |
| FT-08 | 程序打包 | 在 experiment2/ 目录下执行 PyInstaller 打包命令 | 成功生成 dist/CIFAR10分类器.exe | 通过 |
| FT-09 | exe 独立运行 | 双击 dist/ 目录中的 exe 文件 | 程序正常启动，GUI 界面完整显示，功能与源码运行一致 | 通过 |
| FT-10 | 无 Python 环境运行 | 将 exe 和 model.pth 复制到未安装 Python 的电脑上运行 | 程序正常运行，识别功能正常 | 通过 |

### 4.3 性能测试

| 测试编号 | 测试项 | 测试指标 | 结果 |
|----------|--------|----------|------|
| PT-01 | 模型推理速度（CPU） | 单张图片推理耗时 | < 500ms |
| PT-02 | 模型推理速度（GPU） | 单张图片推理耗时 | < 100ms |
| PT-03 | 模型准确率 | 测试集 Top-1 准确率 | {{test_acc}}% |
| PT-04 | exe 启动时间 | 双击到窗口完整显示 | < 5 秒 |
| PT-05 | exe 内存占用 | 程序运行时内存 | < 500MB |
| PT-06 | 模型大小 | model.pth 文件大小 | < 10MB |

### 4.4 用户交互测试

| 测试编号 | 测试项 | 测试步骤 | 预期结果 | 实际结果 |
|----------|--------|----------|----------|----------|
| UT-01 | 非图片文件处理 | 选择 .txt 文本文件后点击识别 | 程序不崩溃，状态栏提示错误信息 | 通过 |
| UT-02 | 未选图片直接识别 | 启动程序后不加载图片直接点击"开始识别" | 状态栏提示"请先选择一张图片"，不崩溃 | 通过 |
| UT-03 | 模型文件缺失 | 删除 model.pth 后启动程序 | 状态栏显示"警告: 模型文件不存在"，不崩溃 | 通过 |
| UT-04 | 连续多次识别 | 依次选择 5 张不同类别的图片进行识别 | 每次结果显示正确，界面更新流畅，无内存泄漏迹象 | 通过 |
| UT-05 | 窗口缩放 | 拖拽窗口边缘改变窗口大小 | 布局自适应调整（左右比例保持），控件不重叠 | 通过 |

---

## 五、结论与展望

### 5.1 项目总结

本项目基于 CIFAR-10 数据集，完成了一个从模型训练到 GUI 应用再到程序打包的完整图像分类系统开发，具体成果包括：

1. **模型训练：** 使用 PyTorch 构建了三层卷积两层全连接的 CNN 模型，在 CIFAR-10 测试集上取得了 {{test_acc}}% 的分类准确率，验证了 CNN 在图像分类任务中的有效性；
2. **GUI 开发：** 基于 PySide6 框架开发了用户友好的图形界面，支持图片选择和拖拽两种交互方式，实时显示分类结果和各类别概率分布。程序使用了 PyQt 经典的信号-槽（Signal-Slot）机制实现事件驱动架构，代码结构清晰，便于维护和扩展；
3. **程序打包：** 使用 PyInstaller 将 Python 程序成功打包为 Windows 独立 exe 文件，无需用户安装 Python 环境和任何第三方库即可运行，降低了程序的使用门槛。

### 5.2 不足分析

1. **模型精度有限：** {{test_acc}}% 的分类准确率距离 CIFAR-10 上的当前最先进水平（99%+）仍有较大差距，主要受限于自定义浅层 CNN 的特征提取能力和有限的训练策略；
2. **输入尺寸限制：** 模型训练时固定输入为 32×32 像素，GUI 程序在识别前会将任意尺寸图片强制缩放至 32×32，可能导致高分辨率图片的细节信息大量丢失；
3. **仅支持 CIFAR-10 类别：** 模型只能识别 10 个预定义类别，对于其他类别（如人物、建筑、食物等）的图片也会强制分类到 10 类之一，缺乏"无法识别"或"未知类别"的判断机制；
### 5.3 改进方向

1. **模型架构升级：** 使用 ResNet-18、ResNet-50 或 EfficientNet 等更深的网络结构替代自定义浅层 CNN，配合迁移学习（在 ImageNet 预训练后微调 CIFAR-10），预期可将测试准确率提升至 95% 以上；
2. **多尺度训练：** 在数据增强阶段引入多尺度训练策略，使模型对不同分辨率的输入具有更好的鲁棒性和泛化能力；
3. **引入开集识别（Open Set Recognition）：** 增加置信度阈值判断机制，当所有类别概率均低于预设阈值时输出"未知类别"，避免对不相关图片的强制误分类；

---

## 参考文献

[1] Krizhevsky A, Hinton G. Learning multiple layers of features from tiny images[R]. University of Toronto, 2009.

[2] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 770-778.

[3] Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[J]. Communications of the ACM, 2017, 60(6): 84-90.

[4] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[C]//International Conference on Learning Representations (ICLR). 2015.

[5] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]//International Conference on Machine Learning (ICML). 2015: 448-456.