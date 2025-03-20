from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QSlider, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from core.video_processor import VideoProcessor

class ControlPanel(QWidget):
    """控制面板组件，提供视频对齐的控制功能和参数调整"""
    
    # 自定义信号
    parameters_changed = pyqtSignal(dict)  # 发送参数变更
    control_action = pyqtSignal(str, object)  # 发送控制动作
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.init_connections()
        
    def setup_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        
        # 特征提取参数组
        feature_group = QGroupBox("特征提取参数")
        feature_layout = QVBoxLayout(feature_group)
        
        # ORB特征点数量
        orb_layout = QHBoxLayout()
        orb_label = QLabel("特征点数量:")
        self.orb_points = QSpinBox()
        self.orb_points.setRange(100, 5000)
        self.orb_points.setValue(1000)
        self.orb_points.setSingleStep(100)
        orb_layout.addWidget(orb_label)
        orb_layout.addWidget(self.orb_points)
        feature_layout.addLayout(orb_layout)
        
        # 多尺度检测
        self.multi_scale = QCheckBox("启用多尺度检测")
        self.multi_scale.setChecked(True)
        feature_layout.addWidget(self.multi_scale)
        
        main_layout.addWidget(feature_group)
        
        # 时间对齐参数组
        temporal_group = QGroupBox("时间对齐参数")
        temporal_layout = QVBoxLayout(temporal_group)
        
        # 滑动窗口大小
        window_layout = QHBoxLayout()
        window_label = QLabel("窗口大小:")
        self.window_size = QSpinBox()
        self.window_size.setRange(10, 100)
        self.window_size.setValue(30)
        window_layout.addWidget(window_label)
        window_layout.addWidget(self.window_size)
        temporal_layout.addLayout(window_layout)
        
        # 对齐算法选择
        algorithm_layout = QHBoxLayout()
        algorithm_label = QLabel("对齐算法:")
        self.align_algorithm = QComboBox()
        self.align_algorithm.addItems(["动态规划", "互相关", "特征序列"])
        algorithm_layout.addWidget(algorithm_label)
        algorithm_layout.addWidget(self.align_algorithm)
        temporal_layout.addLayout(algorithm_layout)
        
        main_layout.addWidget(temporal_group)
        
        # 空间对齐参数组
        spatial_group = QGroupBox("空间对齐参数")
        spatial_layout = QVBoxLayout(spatial_group)
        
        # RANSAC阈值
        ransac_layout = QHBoxLayout()
        ransac_label = QLabel("RANSAC阈值:")
        self.ransac_threshold = QDoubleSpinBox()
        self.ransac_threshold.setRange(0.1, 10.0)
        self.ransac_threshold.setValue(3.0)
        self.ransac_threshold.setSingleStep(0.1)
        ransac_layout.addWidget(ransac_label)
        ransac_layout.addWidget(self.ransac_threshold)
        spatial_layout.addLayout(ransac_layout)
        
        # 最小匹配点数
        matches_layout = QHBoxLayout()
        matches_label = QLabel("最小匹配点:")
        self.min_matches = QSpinBox()
        self.min_matches.setRange(4, 100)
        self.min_matches.setValue(10)
        matches_layout.addWidget(matches_label)
        matches_layout.addWidget(self.min_matches)
        spatial_layout.addLayout(matches_layout)
        
        main_layout.addWidget(spatial_group)
        
        # 视频融合参数组
        blend_group = QGroupBox("视频融合参数")
        blend_layout = QVBoxLayout(blend_group)
        
        # 混合比例滑块
        ratio_layout = QHBoxLayout()
        blend_label = QLabel("混合比例:")
        self.blend_ratio = QSlider(Qt.Orientation.Horizontal)
        self.blend_ratio.setRange(0, 100)
        self.blend_ratio.setValue(50)
        self.blend_value = QLabel("50%")
        ratio_layout.addWidget(blend_label)
        ratio_layout.addWidget(self.blend_ratio)
        ratio_layout.addWidget(self.blend_value)
        blend_layout.addLayout(ratio_layout)
        
        # 显示设置
        self.show_features = QCheckBox("显示特征点")
        self.show_matches = QCheckBox("显示匹配线")
        blend_layout.addWidget(self.show_features)
        blend_layout.addWidget(self.show_matches)
        
        main_layout.addWidget(blend_group)
        
        # 控制按钮组
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout(control_group)
        
        # 重置按钮
        self.reset_btn = QPushButton("重置参数")
        control_layout.addWidget(self.reset_btn)
        
        # 应用按钮
        self.apply_btn = QPushButton("应用更改")
        control_layout.addWidget(self.apply_btn)
        
        main_layout.addWidget(control_group)
        
        # 添加弹性空间
        main_layout.addStretch()
        
    def init_connections(self):
        """初始化信号连接"""
        # 参数变更信号
        self.orb_points.valueChanged.connect(self._params_changed)
        self.multi_scale.stateChanged.connect(self._params_changed)
        self.window_size.valueChanged.connect(self._params_changed)
        self.align_algorithm.currentTextChanged.connect(self._params_changed)
        self.ransac_threshold.valueChanged.connect(self._params_changed)
        self.min_matches.valueChanged.connect(self._params_changed)
        self.blend_ratio.valueChanged.connect(self._blend_changed)
        self.show_features.stateChanged.connect(self._display_changed)
        self.show_matches.stateChanged.connect(self._display_changed)
        
        # 控制按钮信号
        self.reset_btn.clicked.connect(self._reset_parameters)
        self.apply_btn.clicked.connect(self._apply_parameters)
        
    def _params_changed(self):
        """参数变更处理"""
        params = {
            'orb_points': self.orb_points.value(),
            'multi_scale': self.multi_scale.isChecked(),
            'window_size': self.window_size.value(),
            'align_algorithm': self.align_algorithm.currentText(),
            'ransac_threshold': self.ransac_threshold.value(),
            'min_matches': self.min_matches.value(),
            'blend_ratio': self.blend_ratio.value() / 100.0,
            'show_features': self.show_features.isChecked(),
            'show_matches': self.show_matches.isChecked()
        }
        self.parameters_changed.emit(params)
        
    def _blend_changed(self, value):
        """混合比例变更处理"""
        self.blend_value.setText(f"{value}%")
        self._params_changed()
        
    def _display_changed(self):
        """显示设置变更处理"""
        self.control_action.emit('display_changed', {
            'show_features': self.show_features.isChecked(),
            'show_matches': self.show_matches.isChecked()
        })
        
    def _reset_parameters(self):
        """重置参数"""
        self.orb_points.setValue(1000)
        self.multi_scale.setChecked(True)
        self.window_size.setValue(30)
        self.align_algorithm.setCurrentText("动态规划")
        self.ransac_threshold.setValue(3.0)
        self.min_matches.setValue(10)
        self.blend_ratio.setValue(50)
        self.show_features.setChecked(False)
        self.show_matches.setChecked(False)
        self.control_action.emit('reset', None)
        
    def _apply_parameters(self):
        """应用参数"""
        self._params_changed()
        self.control_action.emit('apply', None)
        
    def get_parameters(self):
        """获取当前参数"""
        return {
            'orb_points': self.orb_points.value(),
            'multi_scale': self.multi_scale.isChecked(),
            'window_size': self.window_size.value(),
            'align_algorithm': self.align_algorithm.currentText(),
            'ransac_threshold': self.ransac_threshold.value(),
            'min_matches': self.min_matches.value(),
            'blend_ratio': self.blend_ratio.value() / 100.0,
            'show_features': self.show_features.isChecked(),
            'show_matches': self.show_matches.isChecked()
        }
