import cv2.cuda
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d

class TemporalAligner:
    """时间对齐器类，用于处理不同帧率视频的同步"""

    def __init__(self, window_size=30):
        """
        初始化时间对齐器
        :param window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.target_fps = None

    def normalize_frame_rates(self, video1_path, video2_path):
        """
        统一两个视频的帧率
        :param video1_path: 第一个视频路径
        :param video2_path: 第二个视频路径
        :return: 标准化后的两个视频捕获对象
        """
        # 打开视频
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)

        if not cap1.isOpened() or not cap2.isOpened():
            raise ValueError("无法打开视频文件")

        # 获取原始帧率
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)

        # 选择合适的目标帧率
        if abs(fps1 - fps2) > 5:  # 如果帧率差异大于5fps
            # 使用较低帧率的1.5倍作为目标，避免生成过多的插值帧
            self.target_fps = min(fps1, fps2) * 1.5
        else:
            # 帧率接近时使用较高的帧率
            self.target_fps = max(fps1, fps2)

        # 创建带缓冲的视频捕获器
        cap1 = BufferedVideoCapture(cap1, self.target_fps, fps1)
        cap2 = BufferedVideoCapture(cap2, self.target_fps, fps2)

        return cap1, cap2, fps1, fps2

class BufferedVideoCapture:
    """带帧率调整的视频捕获器"""

    def __init__(self, cap, target_fps, original_fps):
        self.cap = cap
        self.target_fps = target_fps
        self.original_fps = original_fps
        self.frame_buffer = []
        self.last_pts = 0
        
    def read(self):
        """读取经过帧率调整的帧"""
        if not self.frame_buffer:
            # 读取原始帧
            ret, frame = self.cap.read()
            if not ret:
                return False, None
                
            if self.target_fps > self.original_fps:
                # 需要插值生成额外的帧
                ret2, next_frame = self.cap.read()
                if ret2:
                    # 计算需要插入的帧数
                    ratio = self.target_fps / self.original_fps
                    n_frames = int(ratio) - 1
                    
                    # 生成插值帧
                    for i in range(n_frames):
                        alpha = (i + 1) / (n_frames + 1)
                        interp_frame = cv2.addWeighted(frame, 1 - alpha, next_frame, alpha, 0)
                        self.frame_buffer.append(interp_frame)
                        
                    self.frame_buffer.append(next_frame)
                    # 回退一帧，因为已经读取了下一帧
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                
            elif self.target_fps < self.original_fps:
                # 需要丢弃一些帧以降低帧率
                skip_ratio = self.original_fps / self.target_fps
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                next_pos = current_pos + skip_ratio
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(next_pos))
            
            self.frame_buffer.append(frame)
            
        if self.frame_buffer:
            return True, self.frame_buffer.pop(0)
        return False, None
        
    def isOpened(self):
        """检查视频是否打开"""
        return self.cap.isOpened()
        
    def get(self, propId):
        """获取视频属性"""
        if propId == cv2.CAP_PROP_FPS:
            return self.target_fps
        return self.cap.get(propId)
        
    def set(self, propId, value):
        """设置视频属性"""
        return self.cap.set(propId, value)
        
    def release(self):
        """释放资源"""
        self.cap.release()

    def compute_frame_correspondence(self, features1, features2):
        """
        计算两组特征序列之间的对应关系
        :param features1: 第一个视频的特征序列
        :param features2: 第二个视频的特征序列
        :return: 帧对应关系
        """
        n = len(features1)
        m = len(features2)
        
        # 初始化动态规划表
        dp = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            dp[i][0] = float('inf')
        for j in range(m + 1):
            dp[0][j] = float('inf')
        dp[0][0] = 0
        
        # 记录路径
        path = np.zeros((n + 1, m + 1, 2), dtype=int)
        
        # 填充动态规划表
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._compute_feature_distance(features1[i-1], features2[j-1])
                
                # 找到最小代价路径
                candidates = [
                    (dp[i-1][j-1], (i-1, j-1)),  # 对角线移动
                    (dp[i-1][j], (i-1, j)),      # 垂直移动
                    (dp[i][j-1], (i, j-1))       # 水平移动
                ]
                
                min_cost, min_path = min(candidates, key=lambda x: x[0])
                dp[i][j] = cost + min_cost
                path[i][j] = min_path
        
        # 回溯找到最优路径
        alignment = []
        i, j = n, m
        while i > 0 and j > 0:
            alignment.append((i-1, j-1))
            prev_i, prev_j = path[i][j]
            i, j = prev_i, prev_j
            
        return list(reversed(alignment))

    def _compute_feature_distance(self, feature1, feature2):
        """
        计算两个特征向量之间的距离
        :param feature1: 第一个特征向量
        :param feature2: 第二个特征向量
        :return: 距离值
        """
        # 分别计算不同类型特征的距离并加权
        distance = 0
        
        # 计算全局特征的欧氏距离
        if 'global' in feature1 and 'global' in feature2:
            for key in feature1['global']:
                if key in feature2['global']:
                    distance += np.linalg.norm(
                        feature1['global'][key] - feature2['global'][key]
                    )
        
        # 计算边缘特征的相似度
        if 'edge' in feature1 and 'edge' in feature2:
            edge_dist = np.linalg.norm(
                feature1['edge']['edge_magnitude'] - feature2['edge']['edge_magnitude']
            )
            distance += edge_dist * 0.5  # 边缘特征权重
        
        return distance

    def interpolate_frame(self, prev_frame, next_frame, ratio):
        """
        在两帧之间进行插值
        :param prev_frame: 前一帧
        :param next_frame: 后一帧
        :param ratio: 插值比例 (0-1)
        :return: 插值后的帧
        """
        # 使用OpenCV的光流计算
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # 根据光流进行插值
        h, w = prev_frame.shape[:2]
        map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
        
        map_x += flow[:,:,0] * ratio
        map_y += flow[:,:,1] * ratio
        
        # 使用重映射获取插值帧
        return cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR)

    def optimize_alignment(self, alignment, features1, features2):
        """
        优化对齐结果
        :param alignment: 初始对齐结果
        :param features1: 第一个视频的特征序列
        :param features2: 第二个视频的特征序列
        :return: 优化后的对齐结果
        """
        optimized = []
        window_start = 0
        
        while window_start < len(alignment):
            # 提取当前窗口的对齐片段
            window_end = min(window_start + self.window_size, len(alignment))
            window = alignment[window_start:window_end]
            
            # 对窗口内的对齐进行局部优化
            optimized_window = self._optimize_window(
                window, features1, features2
            )
            
            optimized.extend(optimized_window)
            window_start = window_end
            
        return optimized

    def _optimize_window(self, window, features1, features2):
        """
        优化单个窗口内的对齐
        :param window: 要优化的对齐窗口
        :param features1: 第一个视频的特征序列
        :param features2: 第二个视频的特征序列
        :return: 优化后的窗口对齐
        """
        if len(window) < 2:
            return window
            
        # 提取窗口内的对应帧索引
        idx1 = [pair[0] for pair in window]
        idx2 = [pair[1] for pair in window]
        
        # 使用多项式拟合平滑对应关系
        poly = np.polyfit(idx1, idx2, 2)
        smooth_idx2 = np.polyval(poly, idx1)
        
        # 创建优化后的对应关系
        optimized = []
        for i, idx in enumerate(idx1):
            optimized.append((idx, int(round(smooth_idx2[i]))))
            
        return optimized
