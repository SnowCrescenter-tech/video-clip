from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, QTimer
import cv2
import numpy as np
from core.video_processor import VideoProcessor
from utils.helpers import load_video, get_video_info
from gui.video_widget import VideoWidget
from gui.control_panel import ControlPanel

class MainWindow(QMainWindow):
    """主窗口类，提供视频对齐的图形界面"""

    def __init__(self):
        super().__init__()
        self.video_processor = VideoProcessor()
        self.setup_ui()
        self.setup_timer()
        self.setup_connections()

    def setup_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("视频对齐系统")
        self.setMinimumSize(1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧视频显示区域
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumSize(800, 600)
        video_layout.addWidget(self.video_widget)
        
        # 视频信息标签
        self.video_info = QLabel()
        self.video_info.setAlignment(Qt.AlignmentFlag.AlignLeft)
        video_layout.addWidget(self.video_info)
        
        splitter.addWidget(video_container)
        
        # 右侧控制面板
        self.control_panel = ControlPanel()
        splitter.addWidget(self.control_panel)
        
        # 设置分割器比例
        splitter.setStretchFactor(0, 7)  # 视频区域
        splitter.setStretchFactor(1, 3)  # 控制面板
        
        main_layout.addWidget(splitter)

        # 底部控制区域
        bottom_layout = QHBoxLayout()
        
        # 文件选择按钮
        file_group = QHBoxLayout()
        self.video1_btn = QPushButton("选择视频1")
        self.video1_btn.clicked.connect(self.load_video1)
        file_group.addWidget(self.video1_btn)
        
        self.video2_btn = QPushButton("选择视频2")
        self.video2_btn.clicked.connect(self.load_video2)
        file_group.addWidget(self.video2_btn)
        bottom_layout.addLayout(file_group)
        
        # 控制按钮
        control_group = QHBoxLayout()
        self.start_btn = QPushButton("开始对齐")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.toggle_processing)
        control_group.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        control_group.addWidget(self.pause_btn)
        bottom_layout.addLayout(control_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        bottom_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(bottom_layout)
        
        # 状态标签
        self.status_label = QLabel("请选择要对齐的视频文件")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # 初始化视频路径
        self.video1_path = None
        self.video2_path = None

    def setup_timer(self):
        """设置定时器用于更新视频帧"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(33)  # 约30fps

    def setup_connections(self):
        """设置信号连接"""
        # 控制面板参数变更
        self.control_panel.parameters_changed.connect(self.on_parameters_changed)
        self.control_panel.control_action.connect(self.on_control_action)

    def load_video1(self):
        """加载第一个视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择第一个视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)"
        )
        if file_path:
            success, result = load_video(file_path)
            if success:
                self.video1_path = file_path
                self.video1_btn.setText("视频1已选择")
                info = get_video_info(result)
                self.update_video_info(1, info)
                result.release()
                self.check_videos_loaded()
            else:
                QMessageBox.critical(self, "错误", f"无法加载视频1: {result}")

    def load_video2(self):
        """加载第二个视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择第二个视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)"
        )
        if file_path:
            success, result = load_video(file_path)
            if success:
                self.video2_path = file_path
                self.video2_btn.setText("视频2已选择")
                info = get_video_info(result)
                self.update_video_info(2, info)
                result.release()
                self.check_videos_loaded()
            else:
                QMessageBox.critical(self, "错误", f"无法加载视频2: {result}")

    def update_video_info(self, video_num, info):
        """更新视频信息显示"""
        text = f"视频{video_num}: {info['width']}x{info['height']} "
        text += f"@ {info['fps']:.2f}fps, "
        text += f"总帧数: {info['frame_count']}, "
        text += f"时长: {info['duration']}秒"
        if video_num == 1:
            self.video_info.setText(text)
        else:
            current = self.video_info.text()
            self.video_info.setText(current + "\n" + text)

    def check_videos_loaded(self):
        """检查是否两个视频都已加载"""
        if self.video1_path and self.video2_path:
            self.start_btn.setEnabled(True)
            self.status_label.setText("两个视频都已选择，可以开始对齐")
        else:
            self.start_btn.setEnabled(False)

    def toggle_processing(self):
        """切换视频处理状态"""
        if not self.video_processor.is_processing:
            # 开始处理
            params = self.control_panel.get_parameters()
            if self.video_processor.load_videos(
                self.video1_path,
                self.video2_path,
                progress_callback=self.update_progress
            ):
                self.video_processor.start_processing()
                self.start_btn.setText("停止对齐")
                self.pause_btn.setEnabled(True)
                self.video1_btn.setEnabled(False)
                self.video2_btn.setEnabled(False)
                self.control_panel.setEnabled(False)
                self.status_label.setText("正在对齐视频...")
                self.timer.start()
            else:
                QMessageBox.critical(self, "错误", "无法加载视频文件")
        else:
            # 停止处理
            self.stop_processing()

    def toggle_pause(self):
        """切换暂停/继续状态"""
        if self.video_processor.is_paused:
            self.video_processor.resume_processing()
            self.pause_btn.setText("暂停")
            self.status_label.setText("正在对齐视频...")
        else:
            self.video_processor.pause_processing()
            self.pause_btn.setText("继续")
            self.status_label.setText("视频对齐已暂停")

    def stop_processing(self):
        """停止视频处理"""
        self.timer.stop()
        self.video_processor.stop_processing()
        self.start_btn.setText("开始对齐")
        self.pause_btn.setEnabled(False)
        self.video1_btn.setEnabled(True)
        self.video2_btn.setEnabled(True)
        self.control_panel.setEnabled(True)
        self.status_label.setText("视频对齐已停止")
        self.video_widget.display_frame(None)

    def update_progress(self, progress):
        """更新进度显示"""
        self.progress_bar.setValue(int(progress))
        
        # 更新性能信息
        frames, total, progress, fps = self.video_processor.get_progress()
        self.status_label.setText(
            f"处理进度: {frames}/{total} 帧 ({progress:.1f}%) | "
            f"处理速度: {fps:.1f} FPS"
        )
        
        # 检查是否有错误
        error = self.video_processor.get_error()
        if error:
            self.status_label.setText(f"处理出错: {error}")
            self.stop_processing()

    def update_frame(self):
        """更新显示的视频帧"""
        try:
            frame_info = self.video_processor.get_next_frame()
            if frame_info is None:
                if not self.video_processor.is_processing:
                    self.stop_processing()
                return
            
            # 获取帧数据并验证
            if 'error' in frame_info:
                # 处理错误帧
                error_msg = frame_info['error']
                self.status_label.setText(f"处理出错: {error_msg}")
                self.status_label.setStyleSheet("color: red")
                self.video_widget.display_frame(None)
                return
                
            frame = frame_info.get('frame1')  # 使用第一个视频帧作为主显示
            if frame is not None and isinstance(frame, np.ndarray):
                # 更新视频帧
                self.video_widget.display_frame(frame)
                
                # 基本状态信息
                status_text = f"处理进度: {frame_info['frame_index']}/{frame_info['total_frames']} 帧"
                buffer_status = frame_info.get('buffer_status', {})
                speed_ratio = frame_info.get('speed_ratio', 1.0)
                
                # 添加性能信息
                if 'processing_time' in frame_info:
                    status_text += f" | 处理时间: {frame_info['processing_time']*1000:.1f}ms"
                if 'fps' in frame_info:
                    status_text += f" | FPS: {frame_info['fps']:.1f}"
                
                # 添加缓冲区状态信息
                if buffer_status:
                    status_text += f" | 预处理: {buffer_status['preprocess']:.0%}"
                    status_text += f" | 处理: {buffer_status['process']:.0%}"
                    status_text += f" | 显示: {buffer_status['display']:.0%}"
                    
                # 添加速度比率信息
                if abs(speed_ratio - 1.0) > 0.1:
                    status_text += f" | 速度比: {speed_ratio:.2f}x"
                    
                # 处理状态颜色
                if buffer_status.get('is_preprocessing', False):
                    status_text = "预处理中... " + status_text
                    self.status_label.setStyleSheet("color: blue")
                elif 'processing_warning' in frame_info:
                    status_text += f" | 警告: {frame_info['processing_warning']}"
                    self.status_label.setStyleSheet("color: orange")
                else:
                    self.status_label.setStyleSheet("")
                    
                # 更新状态文本
                self.status_label.setText(status_text)
                
                # 如果需要显示特征点或匹配线
                params = self.control_panel.get_parameters()
                if params['show_features'] or params['show_matches']:
                    overlay_info = {}
                    if 'keypoints' in frame_info:
                        overlay_info['keypoints'] = frame_info['keypoints']
                    if 'matches' in frame_info:
                        overlay_info['matches'] = frame_info['matches']
                    self.video_widget.set_overlay(True, overlay_info)
                else:
                    self.video_widget.set_overlay(False)
                    
                # 更新进度信息
                progress = (frame_info['frame_index'] / frame_info['total_frames']) * 100
                self.progress_bar.setValue(int(progress))
            else:
                self.video_widget.display_frame(None)
                self.status_label.setText("等待帧...")
                
        except Exception as e:
            print(f"更新帧时出错: {str(e)}")
            self.video_widget.display_frame(None)

    def on_parameters_changed(self, params):
        """处理参数变更"""
        if self.video_processor.is_processing:
            self.video_processor.update_parameters(params)

    def on_control_action(self, action, data):
        """处理控制动作"""
        if action == 'reset':
            self.video_processor.reset_parameters()
        elif action == 'apply':
            params = self.control_panel.get_parameters()
            self.video_processor.update_parameters(params)
        elif action == 'display_changed':
            if self.video_processor.is_processing:
                self.video_processor.update_display_options(data)

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.video_processor.release()
        event.accept()
