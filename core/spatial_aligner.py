import cv2.cuda
import numpy as np
from scipy.spatial import distance

class SpatialAligner:
    """空间对齐器类，用于处理视频的几何变换和空间对齐"""

    def __init__(self, ransac_threshold=3.0, min_matches=10):
        """
        初始化空间对齐器
        :param ransac_threshold: RANSAC算法的阈值
        :param min_matches: 最小匹配点数
        """
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.prev_transform = None
        self.kalman = None

    def init_kalman_filter(self):
        """
        初始化卡尔曼滤波器
        用于平滑变换矩阵的变化
        """
        # 状态向量: [dx, dy, da, ds] (平移x,y, 旋转角度, 尺度)
        self.kalman = cv2.KalmanFilter(4, 4)
        self.kalman.measurementMatrix = np.eye(4, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-3

    def estimate_transform(self, src_points, dst_points):
        """
        估计两组点之间的几何变换
        :param src_points: 源图像上的特征点
        :param dst_points: 目标图像上的特征点
        :return: 变换矩阵
        """
        if len(src_points) < self.min_matches:
            return None

        # 使用RANSAC估计变换矩阵
        transform, mask = cv2.findHomography(
            src_points, 
            dst_points, 
            cv2.RANSAC, 
            self.ransac_threshold
        )

        if transform is None:
            return None

        # 分解变换矩阵获取基本变换参数
        translation, rotation, scale = self._decompose_transform(transform)

        # 如果是第一次变换，初始化卡尔曼滤波器
        if self.kalman is None:
            self.init_kalman_filter()
            self.prev_transform = transform
            return transform

        # 使用卡尔曼滤波器平滑变换参数
        state = np.array([
            translation[0], 
            translation[1], 
            rotation, 
            scale
        ], dtype=np.float32)

        self.kalman.predict()
        filtered_state = self.kalman.correct(state)

        # 重建变换矩阵
        smoothed_transform = self._compose_transform(
            filtered_state[:2],
            filtered_state[2],
            filtered_state[3]
        )

        self.prev_transform = smoothed_transform
        return smoothed_transform

    def _decompose_transform(self, transform):
        """
        分解变换矩阵为基本变换参数
        :param transform: 变换矩阵
        :return: (平移, 旋转, 缩放)
        """
        # 提取平移分量
        translation = transform[:2, 2]

        # 提取旋转和缩放
        A = transform[:2, :2]
        scale = np.sqrt(np.mean(np.sum(A * A, axis=0)))
        rotation = np.arctan2(A[1, 0], A[0, 0])

        return translation, rotation, scale

    def _compose_transform(self, translation, rotation, scale):
        """
        从基本变换参数重建变换矩阵
        :param translation: 平移向量
        :param rotation: 旋转角度
        :param scale: 缩放因子
        :return: 变换矩阵
        """
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)

        transform = np.array([
            [scale * cos_theta, -scale * sin_theta, translation[0]],
            [scale * sin_theta, scale * cos_theta, translation[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        return transform

    def check_transform_validity(self, transform):
        """
        检查变换矩阵的有效性
        :param transform: 变换矩阵
        :return: 是否有效
        """
        if transform is None:
            return False

        # 检查变换矩阵是否包含非法值
        if not np.all(np.isfinite(transform)):
            return False

        # 提取变换参数
        translation, rotation, scale = self._decompose_transform(transform)

        # 检查缩放是否在合理范围内
        if not 0.5 < scale < 2.0:
            return False

        # 检查旋转是否在合理范围内 (比如限制在±30度)
        if abs(rotation) > np.pi/6:
            return False

        # 检查平移是否在合理范围内
        if np.any(np.abs(translation) > 100):
            return False

        return True

    def apply_transform(self, frame, transform, output_size):
        """
        将变换应用到帧上
        :param frame: 输入帧
        :param transform: 变换矩阵
        :param output_size: 输出大小 (width, height)
        :return: 变换后的帧
        """
        if transform is None or not self.check_transform_validity(transform):
            return frame

        return cv2.warpPerspective(
            frame,
            transform,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    def blend_frames(self, frame1, frame2, alpha=0.5):
        """
        混合两帧图像
        :param frame1: 第一帧
        :param frame2: 第二帧
        :param alpha: 混合比例
        :return: 混合后的帧
        """
        if frame1 is None or frame2 is None:
            return None

        # 确保两帧具有相同的大小
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        return cv2.addWeighted(frame1, alpha, frame2, 1-alpha, 0)
