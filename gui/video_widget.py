from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2
import numpy as np

class VideoWidget(QWidget):
    """视频显示组件，提供视频预览和交互功能"""
    
    # 自定义信号
    frame_clicked = pyqtSignal(tuple)  # 发送点击位置坐标
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.init_variables()
        
    def setup_ui(self):
        """初始化界面"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label)
        
        # 启用鼠标追踪
        self.setMouseTracking(True)
        self.video_label.setMouseTracking(True)
        
    def init_variables(self):
        """初始化变量"""
        self.current_frame = None
        self.display_frame = None
        self.scale_factor = 1.0
        self.mouse_pos = None
        self.zoom_enabled = False
        self.draw_overlay = False
        self.overlay_info = {}
        
    def display_frame(self, frame):
        """
        显示视频帧
        :param frame: OpenCV图像(BGR格式)
        """
        if frame is None:
            return
            
        self.current_frame = frame.copy()
        
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            frame_rgb.data,
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # 根据组件大小调整图像
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # 保存缩放因子
        self.scale_factor = min(
            self.video_label.width() / w,
            self.video_label.height() / h
        )
        
        # 在图像上绘制叠加信息
        if self.draw_overlay:
            scaled_pixmap = self._draw_overlay(scaled_pixmap)
            
        self.video_label.setPixmap(scaled_pixmap)
        
    def enable_zoom(self, enabled=True):
        """启用/禁用缩放功能"""
        self.zoom_enabled = enabled
        
    def set_overlay(self, enabled=True, info=None):
        """
        设置叠加显示
        :param enabled: 是否启用叠加
        :param info: 叠加信息字典
        """
        self.draw_overlay = enabled
        if info is not None:
            self.overlay_info = info
            
    def _draw_overlay(self, pixmap):
        """
        在图像上绘制叠加信息
        :param pixmap: 要绘制的QPixmap
        :return: 绘制后的QPixmap
        """
        if not self.overlay_info:
            return pixmap
            
        # 创建可以绘制的副本
        result = QPixmap(pixmap)
        painter = QPainter(result)
        
        # 设置画笔
        pen = QPen(QColor(255, 0, 0))  # 红色
        pen.setWidth(2)
        painter.setPen(pen)
        
        # 绘制特征点
        if 'keypoints' in self.overlay_info:
            for kp in self.overlay_info['keypoints']:
                x = int(kp.pt[0] * self.scale_factor)
                y = int(kp.pt[1] * self.scale_factor)
                size = int(kp.size * self.scale_factor)
                painter.drawEllipse(x - size//2, y - size//2, size, size)
                
        # 绘制匹配线
        if 'matches' in self.overlay_info:
            pen.setColor(QColor(0, 255, 0))  # 绿色
            painter.setPen(pen)
            for match in self.overlay_info['matches']:
                start = (
                    int(match['start'][0] * self.scale_factor),
                    int(match['start'][1] * self.scale_factor)
                )
                end = (
                    int(match['end'][0] * self.scale_factor),
                    int(match['end'][1] * self.scale_factor)
                )
                painter.drawLine(start[0], start[1], end[0], end[1])
                
        # 绘制文本信息
        if 'text' in self.overlay_info:
            pen.setColor(QColor(255, 255, 255))  # 白色
            painter.setPen(pen)
            for text_info in self.overlay_info['text']:
                pos = text_info.get('position', (10, 20))
                text = text_info.get('content', '')
                painter.drawText(
                    int(pos[0] * self.scale_factor),
                    int(pos[1] * self.scale_factor),
                    text
                )
                
        painter.end()
        return result
        
    def mousePressEvent(self, event):
        """鼠标点击事件处理"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            # 转换为原始图像坐标
            if self.current_frame is not None:
                x = int(pos.x() / self.scale_factor)
                y = int(pos.y() / self.scale_factor)
                self.frame_clicked.emit((x, y))
                
    def mouseMoveEvent(self, event):
        """鼠标移动事件处理"""
        self.mouse_pos = event.position()
        if self.zoom_enabled and self.current_frame is not None:
            # 实现放大镜效果
            self._update_zoom_view()
            
    def _update_zoom_view(self):
        """更新放大视图"""
        if self.mouse_pos is None or self.current_frame is None:
            return
            
        # 获取鼠标位置对应的原始图像坐标
        x = int(self.mouse_pos.x() / self.scale_factor)
        y = int(self.mouse_pos.y() / self.scale_factor)
        
        # 定义放大区域
        zoom_size = 150
        zoom_scale = 2.0
        
        # 提取放大区域
        x1 = max(0, x - zoom_size//2)
        y1 = max(0, y - zoom_size//2)
        x2 = min(self.current_frame.shape[1], x1 + zoom_size)
        y2 = min(self.current_frame.shape[0], y1 + zoom_size)
        
        if x1 < x2 and y1 < y2:
            zoom_region = self.current_frame[y1:y2, x1:x2]
            
            # 放大区域
            zoom_region = cv2.resize(
                zoom_region,
                None,
                fx=zoom_scale,
                fy=zoom_scale,
                interpolation=cv2.INTER_LINEAR
            )
            
            # 转换为RGB并创建QImage
            zoom_rgb = cv2.cvtColor(zoom_region, cv2.COLOR_BGR2RGB)
            h, w, ch = zoom_rgb.shape
            bytes_per_line = ch * w
            zoom_image = QImage(
                zoom_rgb.data,
                w, h,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )
