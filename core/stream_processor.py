import cv2.cuda
import numpy as np
import threading
from queue import Queue, Full, Empty
import time
from collections import deque
from core.speed_aligner import SpeedAwareAligner

class StreamProcessor:
    """流式视频处理器，实现预处理缓冲和实时处理"""
    
    def __init__(self, preprocess_seconds=5, process_buffer_seconds=2, display_buffer_seconds=1):
        """
        初始化流式处理器
        :param preprocess_seconds: 预处理缓冲时间（秒）
        :param process_buffer_seconds: 处理缓冲时间（秒）
        :param display_buffer_seconds: 显示缓冲时间（秒）
        """
        self.fps = 30  # 默认帧率
        self.preprocess_size = int(preprocess_seconds * self.fps)
        self.process_size = int(process_buffer_seconds * self.fps)
        self.display_size = int(display_buffer_seconds * self.fps)
        
        # 缓冲队列
        self.preprocess_buffer = Queue(maxsize=self.preprocess_size)
        self.process_buffer = Queue(maxsize=self.process_size)
        self.display_buffer = Queue(maxsize=self.display_size)
        
        # 状态控制
        self.is_running = False
        self.is_preprocessing = True
        self.preprocess_complete = threading.Event()
        
        # 速度感知对齐器
        self.speed_aligner = SpeedAwareAligner()
        
        # 处理线程
        self.preprocess_thread = None
        self.process_thread = None
        self.display_thread = None
        
        # 性能统计
        self.process_times = deque(maxlen=30)
        self.current_fps = 0
        
    def start(self, cap1, cap2, callback=None):
        """
        启动流式处理
        :param cap1: 第一个视频捕获器
        :param cap2: 第二个视频捕获器
        :param callback: 帧处理完成回调函数
        """
        self.cap1 = cap1
        self.cap2 = cap2
        self.callback = callback
        self.is_running = True
        self.is_preprocessing = True
        
        # 启动处理线程
        self.preprocess_thread = threading.Thread(target=self._preprocess_loop)
        self.process_thread = threading.Thread(target=self._process_loop)
        self.display_thread = threading.Thread(target=self._display_loop)
        
        self.preprocess_thread.start()
        self.process_thread.start()
        self.display_thread.start()
        
    def stop(self):
        """停止处理"""
        self.is_running = False
        if self.preprocess_thread:
            self.preprocess_thread.join()
        if self.process_thread:
            self.process_thread.join()
        if self.display_thread:
            self.display_thread.join()
            
    def _preprocess_loop(self):
        """预处理循环"""
        frame_count = 0
        while self.is_running:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                break
                
            try:
                self.preprocess_buffer.put((frame1, frame2), timeout=1)
                frame_count += 1
                
                # 预处理缓冲区已满
                if frame_count >= self.preprocess_size:
                    self.is_preprocessing = False
                    self.preprocess_complete.set()
                    
            except Full:
                time.sleep(0.01)
                
    def _process_loop(self):
        """处理循环"""
        # 等待预处理完成
        self.preprocess_complete.wait()
        
        while self.is_running:
            try:
                # 从预处理缓冲区获取帧
                frame1, frame2 = self.preprocess_buffer.get(timeout=1)
                
                start_time = time.time()
                
                try:
                    # 估计速度并进行补偿
                    speed_ratio = self.speed_aligner.update_speed(frame1, frame2)
                    frame1, frame2 = self.speed_aligner.compensate_speed(frame1, frame2, speed_ratio)
                    
                    # 计算处理时间
                    process_time = time.time() - start_time
                    self.process_times.append(process_time)
                    self.current_fps = 1.0 / (sum(self.process_times) / len(self.process_times))
                    
                    # 放入处理后的帧
                    frame_data = {
                        'frame1': frame1,
                        'frame2': frame2,
                        'speed_ratio': speed_ratio,
                        'fps': self.current_fps,
                        'timestamp': time.time(),
                        'processing_time': process_time
                    }
                    
                    # 添加处理状态信息
                    if hasattr(self.speed_aligner, 'last_match_time'):
                        frame_data['last_match_time'] = self.speed_aligner.last_match_time
                        
                    # 如果帧处理时间过长，记录警告
                    if process_time > 1.0 / 30:  # 假设目标是30fps
                        frame_data['processing_warning'] = f"处理时间过长: {process_time:.3f}秒"
                        
                    try:
                        self.process_buffer.put(frame_data, timeout=1)
                    except Full:
                        print("处理缓冲区已满，丢弃当前帧")
                        continue
                        
                except Exception as e:
                    print(f"帧处理出错: {str(e)}")
                    # 发送错误帧信息
                    error_frame = {
                        'frame1': frame1,
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    try:
                        self.process_buffer.put(error_frame, timeout=1)
                    except Full:
                        continue
                    
            except Empty:
                continue
                
    def _display_loop(self):
        """显示循环"""
        while self.is_running:
            try:
                # 从处理缓冲区获取帧
                frame_data = self.process_buffer.get(timeout=1)
                
                # 放入显示缓冲区
                try:
                    self.display_buffer.put(frame_data, timeout=1)
                except Full:
                    continue
                    
                # 回调通知新帧可用
                if self.callback:
                    self.callback(frame_data)
                    
            except Empty:
                continue
                
    def get_next_frame(self):
        """
        获取下一帧
        :return: 帧数据字典或None
        """
        try:
            return self.display_buffer.get_nowait()
        except Empty:
            return None
            
    def get_buffer_status(self):
        """
        获取缓冲区状态
        :return: 各缓冲区的填充比例
        """
        preprocess_ratio = self.preprocess_buffer.qsize() / self.preprocess_size
        process_ratio = self.process_buffer.qsize() / self.process_size
        display_ratio = self.display_buffer.qsize() / self.display_size
        
        return {
            'preprocess': preprocess_ratio,
            'process': process_ratio,
            'display': display_ratio,
            'is_preprocessing': self.is_preprocessing,
            'current_fps': self.current_fps
        }
