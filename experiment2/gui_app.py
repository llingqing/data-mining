"""
CIFAR-10 图像分类识别系统 — PySide6 GUI 程序
支持选择图片 / 拖拽图片 → 模型推理 → 显示分类结果
"""
import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar, QGroupBox, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ========== CNN 模型定义（与训练代码一致）==========
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ========== 常量 ==========
CLASS_NAMES_CN = ['飞机', '汽车', '鸟', '猫', '鹿',
                  '狗', '青蛙', '马', '船', '卡车']
CLASS_NAMES_EN = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# 图像预处理（与训练时测试集一致）
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 支持拖放的图片预览标签 ==========
class DropLabel(QLabel):
    """支持拖入图片的 QLabel"""
    imageDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText('拖拽图片到此处\n或点击"选择图片"')
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #4caf50;
                border-radius: 8px;
                background-color: #ffffff;
                color: #2e7d32;
                font-size: 14px;
                min-width: 300px;
                min-height: 250px;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.imageDropped.emit(file_path)

# ========== Matplotlib 柱状图画布 ==========
class ProbabilityCanvas(FigureCanvas):
    """显示各类别概率的柱状图"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.init_chart()

    def init_chart(self):
        self.ax.clear()
        self.ax.bar(range(10), [0]*10, color='#ddd')
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels(CLASS_NAMES_CN, rotation=45, ha='right', fontsize=8, color='#2e7d32')
        self.ax.set_ylabel('概率', fontsize=10, color='#2e7d32')
        self.ax.set_ylim(0, 1.05)
        self.ax.tick_params(axis='y', colors='#2e7d32')
        self.fig.tight_layout()
        self.draw()

    def update_chart(self, probabilities):
        self.ax.clear()
        colors = ['#4CAF50' if i == np.argmax(probabilities) else '#81c784'
                  for i in range(10)]
        self.ax.bar(range(10), probabilities, color=colors)
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels(CLASS_NAMES_CN, rotation=45, ha='right', fontsize=8, color='#2e7d32')
        self.ax.set_ylabel('概率', fontsize=10, color='#2e7d32')
        self.ax.set_ylim(0, 1.05)
        self.ax.tick_params(axis='y', colors='#2e7d32')
        # 数值标注
        for i, p in enumerate(probabilities):
            self.ax.text(i, p + 0.02, f'{p:.2f}', ha='center', fontsize=7, color='#2e7d32')
        self.fig.tight_layout()
        self.draw()

# ========== 主窗口 ==========
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.model = None
        self.init_ui()
        self.load_model()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('CIFAR-10 图像分类识别系统')
        self.setMinimumSize(700, 500)

        # ===== 左侧：图片预览 =====
        left_group = QGroupBox('图片预览')
        left_layout = QVBoxLayout(left_group)

        self.image_label = DropLabel()
        # 信号-槽：拖入图片 → 加载并识别
        self.image_label.imageDropped.connect(self.on_image_dropped)

        left_layout.addWidget(self.image_label)

        # ===== 右侧：分类结果 =====
        right_group = QGroupBox('分类结果')
        right_layout = QVBoxLayout(right_group)

        # 结果文本
        self.result_label = QLabel('请选择一张图片进行识别')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border: 1px solid #4caf50;
                border-radius: 6px;
                background-color: #ffffff;
                color: #2e7d32;
            }
        """)
        right_layout.addWidget(self.result_label)

        # 置信度
        self.confidence_label = QLabel('')
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet('font-size: 14px; color: #2e7d32;')
        right_layout.addWidget(self.confidence_label)

        # 概率柱状图
        self.prob_chart = ProbabilityCanvas()
        right_layout.addWidget(self.prob_chart)

        # ===== 左右并排布局（不要设置 parent，后面会 add 到主容器）=====
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_group, 2)
        main_layout.addWidget(right_group, 3)

        # ===== 底部按钮栏 =====
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)

        self.btn_select = QPushButton('选择图片')
        self.btn_select.clicked.connect(self.on_select_image)
        btn_layout.addWidget(self.btn_select)

        self.btn_predict = QPushButton('开始识别')
        self.btn_predict.clicked.connect(self.on_predict)
        self.btn_predict.setEnabled(False)
        btn_layout.addWidget(self.btn_predict)

        self.btn_clear = QPushButton('清空')
        self.btn_clear.clicked.connect(self.on_clear)
        btn_layout.addWidget(self.btn_clear)

        # ===== 组装整体布局（只设一次 setCentralWidget）=====
        main_container = QVBoxLayout()
        main_container.addLayout(main_layout, 1)
        main_container.addWidget(btn_widget)

        central = QWidget()
        central.setLayout(main_container)
        self.setCentralWidget(central)

        # 全局样式：只对特定控件设置，避免破坏 QPushButton 原生渲染
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #4caf50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #2e7d32;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('就绪 — 模型已加载')

    def load_model(self):
        """加载训练好的模型权重"""
        model_path = Path(__file__).parent / 'output' / 'model.pth'
        self.model = CIFAR10CNN(num_classes=10).to(device)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            self.statusBar.showMessage(f'模型已加载: {model_path}')
        else:
            self.statusBar.showMessage(f'警告: 模型文件不存在 {model_path}')

    @Slot()
    def on_select_image(self):
        """槽函数：点击"选择图片"按钮 → 打开文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '',
            '图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*)')
        if file_path:
            self.load_image(file_path)

    @Slot(str)
    def on_image_dropped(self, file_path):
        """槽函数：拖入图片 → 自动加载并识别"""
        self.load_image(file_path)
        self.on_predict()

    @Slot()
    def on_predict(self):
        """槽函数：点击"开始识别" → 模型推理 → 显示结果"""
        if self.current_image_path is None:
            self.statusBar.showMessage('请先选择一张图片')
            return

        if self.model is None:
            self.statusBar.showMessage('模型未加载')
            return

        try:
            # 加载并预处理图片
            image = Image.open(self.current_image_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                _, predicted = torch.max(outputs, 1)
                pred_idx = predicted.item()

            # 更新结果显示
            pred_name = CLASS_NAMES_CN[pred_idx]
            confidence = probabilities[pred_idx]
            self.result_label.setText(f'类别: {pred_name}')
            self.confidence_label.setText(f'置信度: {confidence:.2%}')

            # 更新柱状图
            self.prob_chart.update_chart(probabilities)

            self.statusBar.showMessage(f'识别完成: {pred_name} ({confidence:.2%})')
        except Exception as e:
            self.statusBar.showMessage(f'识别失败: {str(e)}')

    @Slot()
    def on_clear(self):
        """槽函数：点击"清空" → 重置界面"""
        self.current_image_path = None
        self.image_label.clear()
        self.image_label.setText('拖拽图片到此处\n或点击"选择图片"')
        self.result_label.setText('请选择一张图片进行识别')
        self.confidence_label.setText('')
        self.prob_chart.init_chart()
        self.btn_predict.setEnabled(False)
        self.statusBar.showMessage('已清空')

    def load_image(self, file_path):
        """加载图片到预览区"""
        self.current_image_path = file_path
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(300, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            self.btn_predict.setEnabled(True)
            self.statusBar.showMessage(f'已加载: {file_path}')

# ========== 程序入口 ==========
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
