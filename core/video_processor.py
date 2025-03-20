import cv2.cuda
import numpy as np
import threading
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from core.feature_extractor import EnhancedFeatureExtractor
from core.temporal_aligner import TemporalAligner
from core.spatial_aligner import SpatialAligner
from utils.logger import logger

class VideoProcessor:
    """视频处理器类，整合所有对齐功能"""

    def __init__(self, buffer_size=30, max_workers=4):
        """
        初始化视频处理器
        :param buffer_size: 缓冲区大小
        :param max_workers: 最大工作线程数
        """
        self.feature_extractor = EnhancedFeatureExtractor()
        self.temporal_aligner = TemporalAligner()
        self.spatial_aligner = SpatialAligner()
        
        # 流处理器
        self.stream_processor = None
        
        # 超时设置
        self.alignment_timeout = 5.0  # 5秒对齐超时
        self.force_display_threshold = 3.0  # 3秒强制显示阈值
        self.last_successful_frame = None
        self.last_match_time = 0
        
        # 视频属性
        self.cap1 = None
        self.cap2 = None
        self.fps = None
        self.frame_size = None
        self.total_frames = 0
        self.processed_frames = 0
        
        # 处理状态
        self.is_processing = False
        self.is_paused = False
        self.processing_error = None
        
        # 处理参数
        self.parameters = {
            'orb_points': 1000,
            'multi_scale': True,
            'window_size': 30,
            'align_algorithm': "动态规划",
            'ransac_threshold': 3.0,
            'min_matches': 10,
            'blend_ratio': 0.5,
            'show_features': False,
            'show_matches': False
        }
        
        # 缓冲设置
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.aligned_frame_queue = Queue(maxsize=buffer_size)
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.frame_futures = []
        
        # 性能监控
        self.processing_times = []
        self.fps_stats = []
        
        # 进度回调
        self.progress_callback = None
        
        logger.info("VideoProcessor初始化完成")

    def load_videos(self, video1_path, video2_path, progress_callback=None):
        """
        加载视频文件
        :param video1_path: 第一个视频路径
        :param video2_path: 第二个视频路径
        :return: 是否成功加载
        """
        try:
            logger.info(f"开始加载视频: {video1_path}, {video2_path}")
            
            # 获取规范化的视频捕获对象和帧率
            self.cap1, self.cap2, fps1, fps2 = self.temporal_aligner.normalize_frame_rates(
                video1_path, video2_path
            )
            
            # 检查视频是否成功打开
            if not self.cap1.isOpened() or not self.cap2.isOpened():
                raise ValueError("无法打开视频文件")
            
            # 获取视频属性
            self.fps = self.temporal_aligner.target_fps
            width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
            height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 获取总帧数
            self.total_frames = min(
                int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            )
            
            # 选择合适的输出大小
            self.frame_size = (
                max(width1, width2),
                max(height1, height2)
            )
            
            # 设置进度回调
            self.progress_callback = progress_callback
            
            # 重置状态
            self.processed_frames = 0
            self.processing_times.clear()
            self.fps_stats.clear()
            self.processing_error = None
            
            logger.info(f"视频加载成功: {self.frame_size}, {self.fps}fps, {self.total_frames}帧")
            return True
            
        except Exception as e:
            logger.error(f"加载视频时出错: {str(e)}")
            return False

    def start_processing(self, start_frame=0):
        """启动视频处理"""
        if self.is_processing:
            return
            
        logger.info("开始视频处理")
        self.is_processing = True
        self.is_paused = False
        
        # 设置视频起始位置
        if start_frame > 0:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.processed_frames = start_frame
            logger.info(f"从第{start_frame}帧开始处理")
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_videos)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # 启动性能监控
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_processing(self):
        """停止视频处理"""
        logger.info("停止视频处理")
        self.is_processing = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        self._clear_queues()

    def pause_processing(self):
        """暂停视频处理"""
        logger.info("暂停视频处理")
        self.is_paused = True

    def resume_processing(self):
        """恢复视频处理"""
        logger.info("恢复视频处理")
        self.is_paused = False

    def _clear_queues(self):
        """清空帧缓冲队列"""
        logger.debug("清空缓冲队列")
        while not self.frame_queue.empty():
            self.frame_queue.get()
        while not self.aligned_frame_queue.empty():
            self.aligned_frame_queue.get()

    def _process_videos(self):
        """视频处理主循环"""
        try:
            from core.stream_processor import StreamProcessor
            
            # 创建流处理器
            self.stream_processor = StreamProcessor(
                preprocess_seconds=5,
                process_buffer_seconds=2,
                display_buffer_seconds=1
            )
            
            logger.info("正在启动流处理器")
            # 启动流处理器
            self.stream_processor.start(
                self.cap1, 
                self.cap2,
                callback=self._process_frame_callback
            )
            
            # 等待预处理完成
            while self.stream_processor.is_preprocessing:
                if not self.is_processing:
                    break
                time.sleep(0.1)
                
            logger.info("预处理完成，开始主循环")
            
            # 主循环
            while self.is_processing:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # 获取下一帧
                frame_data = self.stream_processor.get_next_frame()
                if frame_data is None:
                    continue
                    
                # 更新处理进度
                self.processed_frames += 1
                
                # 获取缓冲区状态
                buffer_status = self.stream_processor.get_buffer_status()
                
                # 记录性能信息
                current_fps = buffer_status['current_fps']
                self.fps_stats.append(current_fps)
                logger.debug(f"当前FPS: {current_fps:.1f}")
                
                # 构造完整的帧信息
                aligned_frame = {
                    'frame': frame_data['frame1'],
                    'frame1': frame_data['frame1'],
                    'frame2': frame_data['frame2'],
                    'frame_index': self.processed_frames,
                    'total_frames': self.total_frames,
                    'speed_ratio': frame_data.get('speed_ratio', 1.0),
                    'processing_time': frame_data.get('processing_time', 0),
                    'fps': frame_data.get('fps', 0),
                    'timestamp': frame_data.get('timestamp', time.time()),
                    'buffer_status': buffer_status
                }
                
                # 添加处理状态信息
                if 'processing_warning' in frame_data:
                    aligned_frame['processing_warning'] = frame_data['processing_warning']
                    logger.warning(frame_data['processing_warning'])
                
                if 'error' in frame_data:
                    aligned_frame['error'] = frame_data['error']
                    logger.error(frame_data['error'])
                
                # 记录帧处理信息
                logger.log_frame_info(aligned_frame)
                
                # 放入输出队列
                if not self.aligned_frame_queue.full():
                    self.aligned_frame_queue.put(aligned_frame)
                
                # 更新进度
                if self.progress_callback:
                    progress = (self.processed_frames / self.total_frames) * 100
                    self.progress_callback(progress)
                    
        except Exception as e:
            self.processing_error = str(e)
            logger.error(f"视频处理出错: {str(e)}")
        finally:
            if self.stream_processor:
                self.stream_processor.stop()
            self.is_processing = False
            logger.info("视频处理结束")

    def _monitor_performance(self):
        """监控处理性能"""
        while self.is_processing:
            if not self.is_paused and self.processing_times:
                # 计算当前FPS
                recent_times = self.processing_times[-10:]
                if recent_times:
                    current_fps = 1.0 / (sum(recent_times) / len(recent_times))
                    self.fps_stats.append(current_fps)
                    # 记录性能信息
                    buffer_status = self.stream_processor.get_buffer_status() if self.stream_processor else None
                    logger.log_performance(current_fps, buffer_status)
            time.sleep(1.0)

    def get_next_frame(self):
        """
        获取下一个对齐后的帧
        :return: 帧信息字典，如果没有则返回None
        """
        try:
            frame_info = self.aligned_frame_queue.get_nowait()
            if frame_info is None:
                return None
                
            # 验证frame_info的完整性
            if not isinstance(frame_info, dict):
                logger.warning("帧信息格式错误")
                return None
                
            # 确保必要的键存在
            required_keys = ['frame', 'frame_index', 'total_frames']
            if not all(key in frame_info for key in required_keys):
                logger.warning("帧信息缺少必要字段")
                return None
                
            # 验证frame数据
            if frame_info['frame'] is None or not isinstance(frame_info['frame'], np.ndarray):
                logger.warning("帧数据格式错误")
                return None
                
            # 记录帧信息
            logger.debug(f"获取帧 {frame_info['frame_index']}/{frame_info['total_frames']}")
            return frame_info
        except:
            return None

    def get_progress(self):
        """
        获取处理进度
        :return: (当前帧数, 总帧数, 进度百分比, 当前FPS)
        """
        if not self.total_frames:
            return 0, 0, 0, 0
            
        progress = (self.processed_frames / self.total_frames) * 100
        current_fps = self.fps_stats[-1] if self.fps_stats else 0
        
        logger.debug(f"当前进度: {progress:.1f}%, FPS: {current_fps:.1f}")
        return self.processed_frames, self.total_frames, progress, current_fps

    def get_error(self):
        """
        获取处理过程中的错误信息
        :return: 错误信息，如果没有错误则返回None
        """
        if self.processing_error:
            logger.error(f"处理错误: {self.processing_error}")
        return self.processing_error

    def update_parameters(self, params):
        """
        更新处理参数
        :param params: 新的参数字典
        """
        logger.info(f"更新处理参数: {params}")
        self.parameters.update(params)
        
        # 更新相关组件的参数
        self.feature_extractor = EnhancedFeatureExtractor(
            n_features=self.parameters['orb_points']
        )
        self.spatial_aligner = SpatialAligner(
            ransac_threshold=self.parameters['ransac_threshold'],
            min_matches=self.parameters['min_matches']
        )
        self.temporal_aligner = TemporalAligner(
            window_size=self.parameters['window_size']
        )
        
        logger.debug("处理参数更新完成")

    def update_display_options(self, options):
        """
        更新显示选项
        :param options: 显示选项字典
        """
        logger.debug(f"更新显示选项: {options}")
        self.parameters['show_features'] = options.get('show_features', False)
        self.parameters['show_matches'] = options.get('show_matches', False)

    def reset_parameters(self):
        """重置所有参数到默认值"""
        logger.info("重置处理参数到默认值")
        self.parameters = {
            'orb_points': 1000,
            'multi_scale': True,
            'window_size': 30,
            'align_algorithm': "动态规划",
            'ransac_threshold': 3.0,
            'min_matches': 10,
            'blend_ratio': 0.5,
            'show_features': False,
            'show_matches': False
        }
        self.update_parameters(self.parameters)

    def release(self):
        """释放资源"""
        logger.info("释放资源")
        self.stop_processing()
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        logger.info("资源释放完成")

    def _process_frame_callback(self, frame_data):
        """
        帧处理回调函数
        :param frame_data: 处理后的帧数据
        """
        if frame_data:
            logger.debug("帧处理回调")
            if 'error' in frame_data:
                logger.error(f"帧处理错误: {frame_data['error']}")
            elif 'processing_warning' in frame_data:
                logger.warning(f"帧处理警告: {frame_data['processing_warning']}")
