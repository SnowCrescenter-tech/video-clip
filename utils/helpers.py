import cv2
import numpy as np
from pathlib import Path
import time

def load_video(video_path):
    """
    加载视频文件，检查其有效性
    :param video_path: 视频文件路径
    :return: (是否成功, 错误信息或视频capture对象)
    """
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            return False, "视频文件不存在"
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "无法打开视频文件"
            
        return True, cap
    except Exception as e:
        return False, f"加载视频时出错: {str(e)}"

def get_video_info(cap):
    """
    获取视频信息
    :param cap: VideoCapture对象
    :return: 视频信息字典
    """
    return {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }

def calculate_feature_matches(kp1, des1, kp2, des2, ratio_thresh=0.75):
    """
    计算特征点匹配
    :param kp1: 第一组特征点
    :param des1: 第一组描述符
    :param kp2: 第二组特征点
    :param des2: 第二组描述符
    :param ratio_thresh: 比率测试阈值
    :return: 好的匹配点列表
    """
    if des1 is None or des2 is None:
        return []
        
    # 创建FLANN匹配器
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,
        key_size=12,
        multi_probe_level=1
    )
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        # 对描述符进行匹配
        matches = flann.knnMatch(des1, des2, k=2)
        
        # 应用比率测试
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
                    
        return good_matches
    except Exception:
        # 如果FLANN匹配失败，回退到暴力匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)[:50]

def draw_matches_with_info(img1, kp1, img2, kp2, matches, max_matches=50):
    """
    绘制带有信息的匹配结果
    :return: 包含匹配线和信息的图像
    """
    # 选择最佳匹配
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
    
    # 准备匹配图像
    match_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # 添加匹配信息
    text = f"匹配点数: {len(matches)}"
    cv2.putText(
        match_img,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    return match_img

def estimate_transform_with_confidence(src_pts, dst_pts, method=cv2.RANSAC, threshold=3.0):
    """
    估计变换矩阵并计算置信度
    :return: (变换矩阵, 置信度, 内点掩码)
    """
    if len(src_pts) < 4:
        return None, 0.0, None
        
    try:
        # 估计单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, method, threshold)
        
        if H is None:
            return None, 0.0, None
            
        # 计算置信度（内点比例）
        inlier_ratio = np.sum(mask) / len(mask)
        
        return H, float(inlier_ratio), mask
    except Exception:
        return None, 0.0, None

def measure_execution_time(func):
    """
    测量函数执行时间的装饰器
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper

def create_progress_callback(total_steps):
    """
    创建进度回调函数
    :param total_steps: 总步骤数
    :return: 进度回调函数
    """
    def progress_callback(step, info=None):
        progress = (step / total_steps) * 100
        print(f"进度: {progress:.1f}% - {info if info else ''}")
    return progress_callback

def save_debug_image(image, name, debug_dir="debug"):
    """
    保存调试图像
    :param image: 要保存的图像
    :param name: 图像名称
    :param debug_dir: 调试目录
    """
    debug_path = Path(debug_dir)
    debug_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = debug_path / f"{name}_{timestamp}.png"
    cv2.imwrite(str(file_path), image)
    print(f"调试图像已保存: {file_path}")

def get_frame_timestamp(cap, frame_index):
    """
    获取帧的时间戳
    :param cap: VideoCapture对象
    :param frame_index: 帧索引
    :return: 时间戳（秒）
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        return frame_index / fps
    return 0.0

def format_timestamp(seconds):
    """
    格式化时间戳
    :param seconds: 秒数
    :return: 格式化的时间字符串 (HH:MM:SS.mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def interpolate_missing_frames(prev_frame, next_frame, num_frames):
    """
    在两帧之间插值生成中间帧
    :param prev_frame: 前一帧
    :param next_frame: 后一帧
    :param num_frames: 要生成的帧数
    :return: 插值帧列表
    """
    if num_frames < 1:
        return []
        
    frames = []
    # 计算光流
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, 
        next_gray, 
        None, 
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    h, w = prev_frame.shape[:2]
    flow_map = np.zeros((h, w, 2), np.float32)
    
    for i in range(1, num_frames + 1):
        ratio = i / (num_frames + 1)
        
        # 计算当前插值位置的光流
        current_flow = flow * ratio
        
        # 创建插值帧的映射
        map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
        
        # 应用光流偏移
        map_x += current_flow[:,:,0]
        map_y += current_flow[:,:,1]
        
        # 使用重映射进行插值
        interpolated = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR)
        frames.append(interpolated)
        
    return frames

def enhance_frame(frame, brightness=1.0, contrast=1.0):
    """
    增强视频帧
    :param frame: 输入帧
    :param brightness: 亮度调整因子
    :param contrast: 对比度调整因子
    :return: 增强后的帧
    """
    # 亮度和对比度调整
    enhanced = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    
    # 自适应直方图均衡化
    if len(frame.shape) == 3:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    return enhanced

def stabilize_sequence(frames):
    """
    稳定视频序列
    :param frames: 帧序列列表
    :return: 稳定后的帧序列
    """
    if len(frames) < 2:
        return frames
        
    # 初始化GFTT特征检测器
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Lucas-Kanade光流参数
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # 累积变换矩阵
    transforms = []
    for frame in frames[1:]:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        
        # 筛选好的特征点
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # 估计变换矩阵
        if len(good_new) >= 3:
            transform = cv2.estimateAffinePartial2D(good_old, good_new)[0]
        else:
            transform = np.eye(2, 3)
            
        transforms.append(transform)
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    # 应用平滑的变换
    stabilized = [frames[0]]
    accumulated = np.eye(2, 3)
    
    for i, transform in enumerate(transforms):
        # 平滑变换
        smoothed = np.array([
            [1, 0, transform[0,2] * 0.8],
            [0, 1, transform[1,2] * 0.8]
        ])
        
        accumulated = accumulated.dot(np.vstack([smoothed, [0, 0, 1]]))[:2]
        frame = frames[i + 1]
        
        # 应用变换
        h, w = frame.shape[:2]
        stabilized_frame = cv2.warpAffine(
            frame,
            accumulated,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        stabilized.append(stabilized_frame)
    
    return stabilized

def detect_scene_changes(frames, threshold=30):
    """
    检测场景变化
    :param frames: 帧序列
    :param threshold: 差异阈值
    :return: 场景变化的帧索引列表
    """
    changes = []
    if len(frames) < 2:
        return changes
        
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # 计算帧差
        diff = cv2.absdiff(curr_frame, prev_frame)
        mean_diff = np.mean(diff)
        
        if mean_diff > threshold:
            changes.append(i)
            
        prev_frame = curr_frame
        
    return changes

def analyze_motion(frame1, frame2):
    """
    分析两帧之间的运动
    :param frame1: 第一帧
    :param frame2: 第二帧
    :return: 运动信息字典
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # 计算运动统计信息
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    return {
        'mean_magnitude': np.mean(magnitude),
        'max_magnitude': np.max(magnitude),
        'mean_angle': np.mean(angle),
        'flow': flow
    }

def get_frame_quality(frame):
    """
    评估帧的质量
    :param frame: 输入帧
    :return: 质量分数(0-100)和质量信息字典
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 计算拉普拉斯变换
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 计算基本统计信息
    mean = np.mean(frame)
    std = np.std(frame)
    sharpness = np.var(laplacian)
    
    # 检测过度曝光和欠曝光
    overexposed = np.mean(frame > 250)
    underexposed = np.mean(frame < 5)
    
    # 计算整体质量分数
    score = 100
    
    # 根据各项指标调整分数
    if sharpness < 100:
        score -= 30
    if overexposed > 0.1:
        score -= 20
    if underexposed > 0.1:
        score -= 20
    if std < 20:
        score -= 10
        
    score = max(0, min(100, score))
    
    return score, {
        'sharpness': sharpness,
        'brightness': mean,
        'contrast': std,
        'overexposed_ratio': overexposed,
        'underexposed_ratio': underexposed
    }

def extract_keyframes(frames, max_frames=10):
    """
    从视频序列中提取关键帧
    :param frames: 帧序列
    :param max_frames: 最大关键帧数
    :return: 关键帧索引列表
    """
    if len(frames) <= max_frames:
        return list(range(len(frames)))
        
    keyframes = [0]  # 始终包含第一帧
    last_keyframe = frames[0]
    
    for i in range(1, len(frames)):
        frame = frames[i]
        
        # 计算与上一个关键帧的差异
        score = calculate_frame_difference(last_keyframe, frame)
        
        # 如果差异显著，添加为关键帧
        if score > 0.3 and len(keyframes) < max_frames:
            keyframes.append(i)
            last_keyframe = frame
            
    # 确保包含最后一帧
    if keyframes[-1] != len(frames) - 1:
        keyframes.append(len(frames) - 1)
        
    return keyframes

def calculate_frame_difference(frame1, frame2):
    """
    计算两帧之间的差异程度
    :param frame1: 第一帧
    :param frame2: 第二帧
    :return: 差异分数 (0-1)
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 计算直方图
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0,256])
    
    # 归一化直方图
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # 计算直方图差异
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 转换为差异分数
    score = (1.0 - diff) / 2.0
    return score

def generate_thumbnail(frame, size=(120, 120)):
    """
    生成缩略图
    :param frame: 输入帧
    :param size: 目标大小 (宽, 高)
    :return: 缩略图
    """
    h, w = frame.shape[:2]
    
    # 计算缩放比例
    scale = min(size[0]/w, size[1]/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 调整大小
    resized = cv2.resize(frame, (new_w, new_h))
    
    # 创建目标大小的画布
    thumbnail = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # 计算居中位置
    x = (size[0] - new_w) // 2
    y = (size[1] - new_h) // 2
    
    # 将调整后的图像放置在中心
    thumbnail[y:y+new_h, x:x+new_w] = resized
    
    return thumbnail
