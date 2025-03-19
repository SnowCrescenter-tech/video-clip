import cv2
import numpy as np
from scipy.signal import find_peaks

class EnhancedFeatureExtractor:
    """增强特征提取器类，用于从视频帧中提取多种特征"""
    
    def __init__(self, n_features=1000):
        """
        初始化特征提取器
        :param n_features: ORB特征点数量
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.n_features = n_features
        
    def extract_features(self, frame):
        """
        提取多尺度特征
        :param frame: 输入帧
        :return: 特征字典，包含不同类型的特征
        """
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        features = {
            'orb': self._extract_orb(gray),
            'global': self._extract_global_features(frame),
            'edge': self._extract_edge_features(gray)
        }
        return features

    def _extract_orb(self, gray):
        """
        提取ORB特征点和描述符
        :param gray: 灰度图像
        :return: 特征点和描述符
        """
        keypoints = self.orb.detect(gray, None)
        keypoints, descriptors = self.orb.compute(gray, keypoints)
        return {
            'keypoints': keypoints,
            'descriptors': descriptors if descriptors is not None else np.array([])
        }

    def _extract_global_features(self, frame):
        """
        提取全局特征
        :param frame: 输入帧
        :return: 全局特征字典
        """
        features = {}
        
        # 计算颜色直方图
        if len(frame.shape) == 3:
            for i, channel in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([frame], [i], None, [32], [0, 256])
                features[f'hist_{channel}'] = hist.flatten()
        
        # 计算图像的统计特征
        features['mean'] = np.mean(frame)
        features['std'] = np.std(frame)
        
        return features

    def _extract_edge_features(self, gray):
        """
        提取边缘特征
        :param gray: 灰度图像
        :return: 边缘特征字典
        """
        # Sobel边缘检测
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        # 计算边缘强度直方图
        hist_magnitude = np.histogram(magnitude, bins=32)[0]
        hist_direction = np.histogram(direction, bins=36, range=(-np.pi, np.pi))[0]
        
        return {
            'edge_magnitude': hist_magnitude,
            'edge_direction': hist_direction,
            'mean_magnitude': np.mean(magnitude)
        }

    def match_features(self, desc1, desc2):
        """
        特征匹配
        :param desc1: 第一组特征描述符
        :param desc2: 第二组特征描述符
        :return: 匹配结果
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
            
        # 创建BF匹配器
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 执行匹配
        matches = bf.match(desc1, desc2)
        
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches
