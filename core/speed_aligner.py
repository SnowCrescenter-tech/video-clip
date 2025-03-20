import cv2.cuda
import numpy as np
from collections import deque
from scipy.signal import medfilt

class SpeedAwareAligner:
    """速度感知的对齐器，处理视频速度不一致的情况"""
    
    def __init__(self, window_size=30, speed_threshold=0.5):
        """
        初始化速度感知对齐器
        :param window_size: 速度检测窗口大小
        :param speed_threshold: 速度差异阈值
        """
        self.window_size = window_size
        self.speed_threshold = speed_threshold
        self.speed_buffer1 = deque(maxlen=window_size)
        self.speed_buffer2 = deque(maxlen=window_size)
        self.prev_points1 = None
        self.prev_points2 = None
        self.prev_gray1 = None
        self.prev_gray2 = None
        
    def estimate_speed(self, frame, prev_gray, prev_points):
        """
        估计帧间速度
        :param frame: 当前帧
        :param prev_gray: 前一帧的灰度图
        :param prev_points: 前一帧的特征点
        :return: 速度估计值, 当前灰度图, 当前特征点
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is None:
            # 首次处理，检测特征点
            points = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                           qualityLevel=0.3, 
                                           minDistance=7, 
                                           blockSize=7)
            return 0.0, gray, points
            
        if prev_points is None:
            points = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                           qualityLevel=0.3, 
                                           minDistance=7, 
                                           blockSize=7)
            return 0.0, gray, points
            
        # 使用光流法计算特征点移动
        new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, 
                                                          prev_points, None)
                                                          
        # 筛选有效点
        good_old = prev_points[status == 1]
        good_new = new_points[status == 1]
        
        if len(good_new) == 0 or len(good_old) == 0:
            points = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                           qualityLevel=0.3, 
                                           minDistance=7, 
                                           blockSize=7)
            return 0.0, gray, points
            
        # 计算点的移动距离
        distances = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
        
        # 使用中值滤波去除异常值
        if len(distances) > 3:
            speed = np.median(distances)
        else:
            speed = np.mean(distances) if len(distances) > 0 else 0.0
            
        # 检测新的特征点
        points = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                       qualityLevel=0.3, 
                                       minDistance=7, 
                                       blockSize=7)
                                       
        return speed, gray, points
        
    def update_speed(self, frame1, frame2):
        """
        更新两个视频的速度估计
        :param frame1: 视频1的当前帧
        :param frame2: 视频2的当前帧
        :return: 速度比率
        """
        # 估计视频1的速度
        speed1, self.prev_gray1, self.prev_points1 = self.estimate_speed(
            frame1, self.prev_gray1, self.prev_points1
        )
        self.speed_buffer1.append(speed1)
        
        # 估计视频2的速度
        speed2, self.prev_gray2, self.prev_points2 = self.estimate_speed(
            frame2, self.prev_gray2, self.prev_points2
        )
        self.speed_buffer2.append(speed2)
        
        # 计算平均速度并应用平滑处理
        if len(self.speed_buffer1) > 0 and len(self.speed_buffer2) > 0:
            # 使用中值滤波去除异常值
            speeds1 = np.array(self.speed_buffer1)
            speeds2 = np.array(self.speed_buffer2)
            
            # 计算平均速度，避免除零
            avg_speed1 = np.mean(speeds1) if np.any(speeds1) else 0.0001
            avg_speed2 = np.mean(speeds2) if np.any(speeds2) else 0.0001
            
            # 如果速度接近于0，使用默认比率1.0
            if avg_speed1 < 0.0001 or avg_speed2 < 0.0001:
                return 1.0
                
            # 计算速度比并限制在合理范围内
            ratio = avg_speed1 / avg_speed2
            ratio = np.clip(ratio, 0.1, 10.0)  # 限制比率在0.1到10倍之间
            
            return ratio
        else:
            # 缓冲区为空时返回默认比率
            return 1.0
        
    def get_matching_window(self, speed_ratio):
        """
        根据速度比率获取匹配窗口大小
        :param speed_ratio: 速度比率
        :return: 建议的匹配窗口大小
        """
        # 速度差异越大，窗口越大
        base_window = self.window_size
        if abs(speed_ratio - 1.0) > self.speed_threshold:
            return int(base_window * max(speed_ratio, 1/speed_ratio))
        return base_window
        
    def compensate_speed(self, frame1, frame2, speed_ratio):
        """
        对速度差异进行补偿
        :param frame1: 视频1的帧
        :param frame2: 视频2的帧
        :param speed_ratio: 速度比率
        :return: 补偿后的帧
        """
        if abs(speed_ratio - 1.0) <= self.speed_threshold:
            return frame1, frame2
            
        # 根据速度比例调整帧
        if speed_ratio > 1:
            # 视频1比视频2快，对视频1进行模糊处理
            kernel_size = int(min(5, speed_ratio * 2))
            if kernel_size % 2 == 0:
                kernel_size += 1
            frame1 = cv2.GaussianBlur(frame1, (kernel_size, kernel_size), 0)
        else:
            # 视频2比视频1快，对视频2进行模糊处理
            kernel_size = int(min(5, 1/speed_ratio * 2))
            if kernel_size % 2 == 0:
                kernel_size += 1
            frame2 = cv2.GaussianBlur(frame2, (kernel_size, kernel_size), 0)
            
        return frame1, frame2
