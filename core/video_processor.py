import cv2
import numpy as np
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from core.feature_extractor import EnhancedFeatureExtractor
from core.temporal_aligner import TemporalAligner
from core.spatial_aligner import SpatialAligner

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

    def load_videos(self, video1_path, video2_path, progress_callback=None):
        """
        加载视频文件
        :param video1_path: 第一个视频路径
        :param video2_path: 第二个视频路径
        :return: 是否成功加载
        """
        try:
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
            
            return True
            
        except Exception as e:
            print(f"加载视频时出错: {str(e)}")
            return False

    def start_processing(self, start_frame=0):
        """
        启动视频处理
        :param start_frame: 开始处理的帧索引
        """
        if self.is_processing:
            return
            
        self.is_processing = True
        self.is_paused = False
        
        # 设置视频起始位置
        if start_frame > 0:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.processed_frames = start_frame
        
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
        self.is_processing = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        self._clear_queues()

    def pause_processing(self):
        """暂停视频处理"""
        self.is_paused = True

    def resume_processing(self):
        """恢复视频处理"""
        self.is_paused = False

    def _clear_queues(self):
        """清空帧缓冲队列"""
        while not self.frame_queue.empty():
            self.frame_queue.get()
        while not self.aligned_frame_queue.empty():
            self.aligned_frame_queue.get()

    def _process_videos(self):
        """视频处理主循环"""
        import time
        try:
            frame_buffer1 = []
            frame_buffer2 = []
            features_buffer1 = []
            features_buffer2 = []
            
            while self.is_processing:
                if self.is_paused:
                    continue
                    
                start_time = time.time()
                
                # 读取帧
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                
                if not ret1 or not ret2:
                    break
                
                self.processed_frames += 1
                    
                # 在线程池中调整帧大小
                resize_future1 = self.thread_pool.submit(cv2.resize, frame1, self.frame_size)
                resize_future2 = self.thread_pool.submit(cv2.resize, frame2, self.frame_size)
                
                # 等待调整完成
                frame1 = resize_future1.result()
                frame2 = resize_future2.result()
                
                # 在线程池中提取特征
                features_future1 = self.thread_pool.submit(
                    self.feature_extractor.extract_features, frame1
                )
                features_future2 = self.thread_pool.submit(
                    self.feature_extractor.extract_features, frame2
                )
                
                # 等待特征提取完成
                features1 = features_future1.result()
                features2 = features_future2.result()
                
                # 更新缓冲区
                frame_buffer1.append(frame1)
                frame_buffer2.append(frame2)
                features_buffer1.append(features1)
                features_buffer2.append(features2)
                
                # 当缓冲区足够大时进行对齐
                buffer_size = 30  # 滑动窗口大小
                if len(frame_buffer1) >= buffer_size:
                    # 时间对齐
                    alignment = self.temporal_aligner.compute_frame_correspondence(
                        features_buffer1[-buffer_size:],
                        features_buffer2[-buffer_size:]
                    )
                    
                    # 优化对齐结果
                    optimized_alignment = self.temporal_aligner.optimize_alignment(
                        alignment,
                        features_buffer1[-buffer_size:],
                        features_buffer2[-buffer_size:]
                    )
                    
                    # 处理中间帧
                    mid_idx = buffer_size // 2
                    frame1 = frame_buffer1[mid_idx]
                    
                    # 根据对齐结果选择对应帧
                    aligned_idx = None
                    for i, j in optimized_alignment:
                        if i == mid_idx:
                            aligned_idx = j
                            break
                    
                    if aligned_idx is not None and 0 <= aligned_idx < len(frame_buffer2):
                        frame2 = frame_buffer2[aligned_idx]
                        
                        # 空间对齐
                        if features1['orb']['descriptors'] is not None and features2['orb']['descriptors'] is not None:
                            matches = self.feature_extractor.match_features(
                                features1['orb']['descriptors'],
                                features2['orb']['descriptors']
                            )
                            
                            if len(matches) >= self.spatial_aligner.min_matches:
                                src_pts = np.float32([
                                    features1['orb']['keypoints'][m.queryIdx].pt 
                                    for m in matches
                                ]).reshape(-1, 1, 2)
                                
                                dst_pts = np.float32([
                                    features2['orb']['keypoints'][m.trainIdx].pt 
                                    for m in matches
                                ]).reshape(-1, 1, 2)
                                
                                # 估计并应用变换
                                transform = self.spatial_aligner.estimate_transform(
                                    src_pts, dst_pts
                                )
                                
                                if transform is not None:
                                    frame2 = self.spatial_aligner.apply_transform(
                                        frame2, transform, self.frame_size
                                    )
                        
                        # 混合对齐后的帧
                        aligned_frame = self.spatial_aligner.blend_frames(frame1, frame2)
                        
                        # 将对齐后的帧放入输出队列
                        if not self.aligned_frame_queue.full():
                            self.aligned_frame_queue.put({
                                'frame': aligned_frame,
                                'frame_index': self.processed_frames,
                                'total_frames': self.total_frames
                            })
                            
                        # 记录处理时间
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        # 更新进度
                        if self.progress_callback:
                            progress = (self.processed_frames / self.total_frames) * 100
                            self.progress_callback(progress)
                    
                    # 移除旧帧
                    frame_buffer1.pop(0)
                    frame_buffer2.pop(0)
                    features_buffer1.pop(0)
                    features_buffer2.pop(0)
                    
        except Exception as e:
            self.processing_error = str(e)
            print(f"视频处理出错: {str(e)}")
        finally:
            self.is_processing = False

    def _monitor_performance(self):
        """监控处理性能"""
        import time
        while self.is_processing:
            if not self.is_paused and self.processing_times:
                # 计算当前FPS
                recent_times = self.processing_times[-10:]  # 最近10帧的处理时间
                if recent_times:
                    current_fps = 1.0 / (sum(recent_times) / len(recent_times))
                    self.fps_stats.append(current_fps)
                    
            time.sleep(1.0)  # 每秒更新一次

    def get_next_frame(self):
        """
        获取下一个对齐后的帧
        :return: 帧信息字典，如果没有则返回None
        """
        try:
            return self.aligned_frame_queue.get_nowait()
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
        
        return self.processed_frames, self.total_frames, progress, current_fps

    def get_error(self):
        """
        获取处理过程中的错误信息
        :return: 错误信息，如果没有错误则返回None
        """
        return self.processing_error

    def update_parameters(self, params):
        """
        更新处理参数
        :param params: 新的参数字典
        """
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

    def update_display_options(self, options):
        """
        更新显示选项
        :param options: 显示选项字典
        """
        self.parameters['show_features'] = options.get('show_features', False)
        self.parameters['show_matches'] = options.get('show_matches', False)

    def reset_parameters(self):
        """重置所有参数到默认值"""
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
        self.stop_processing()
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
