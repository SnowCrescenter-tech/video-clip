import logging
import os
import sys
from datetime import datetime

class VideoLogger:
    """视频处理日志记录器"""
    
    def __init__(self, name="video_aligner", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 如果已经有处理器，不再添加
        if not self.logger.handlers:
            self._setup_handlers()
            
    def _setup_handlers(self):
        """设置日志处理器"""
        # 创建日志目录
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 文件处理器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"video_aligner_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def debug(self, message):
        """记录调试信息"""
        self.logger.debug(message)
        
    def info(self, message):
        """记录一般信息"""
        self.logger.info(message)
        
    def warning(self, message):
        """记录警告信息"""
        self.logger.warning(message)
        
    def error(self, message):
        """记录错误信息"""
        self.logger.error(message)
        
    def critical(self, message):
        """记录严重错误信息"""
        self.logger.critical(message)
        
    def log_frame_info(self, frame_info):
        """记录帧处理信息"""
        if not isinstance(frame_info, dict):
            return
            
        # 基本信息
        info = f"帧 {frame_info.get('frame_index', '?')}/{frame_info.get('total_frames', '?')}"
        
        # 性能信息
        if 'processing_time' in frame_info:
            info += f" | 处理时间: {frame_info['processing_time']*1000:.1f}ms"
        if 'fps' in frame_info:
            info += f" | FPS: {frame_info['fps']:.1f}"
            
        # 处理状态
        if 'processing_warning' in frame_info:
            info += f" | 警告: {frame_info['processing_warning']}"
        if 'error' in frame_info:
            info += f" | 错误: {frame_info['error']}"
            
        # 记录信息
        self.debug(info)
        
    def log_performance(self, fps, buffer_status):
        """记录性能信息"""
        info = f"性能: {fps:.1f} FPS"
        if buffer_status:
            info += f" | 预处理缓冲: {buffer_status.get('preprocess', 0):.0%}"
            info += f" | 处理缓冲: {buffer_status.get('process', 0):.0%}"
            info += f" | 显示缓冲: {buffer_status.get('display', 0):.0%}"
        self.debug(info)

# 创建全局日志实例
logger = VideoLogger()
