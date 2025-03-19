import os
import cv2
import numpy as np
from core.feature_extractor import EnhancedFeatureExtractor
from core.temporal_aligner import TemporalAligner
from core.spatial_aligner import SpatialAligner
from core.video_processor import VideoProcessor
from utils.helpers import load_video, get_video_info, save_debug_image

def test_feature_extraction(video_path):
    """测试特征提取功能"""
    print("\n测试特征提取...")
    
    # 加载视频
    success, cap = load_video(video_path)
    if not success:
        print(f"无法加载视频: {cap}")
        return False
        
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return False
        
    # 创建特征提取器
    extractor = EnhancedFeatureExtractor(n_features=1000)
    
    try:
        # 提取特征
        features = extractor.extract_features(frame)
        
        # 验证特征
        if features['orb']['keypoints'] is None or len(features['orb']['keypoints']) == 0:
            print("未检测到特征点")
            return False
            
        # 在图像上绘制特征点
        debug_frame = frame.copy()
        for kp in features['orb']['keypoints']:
            x, y = map(int, kp.pt)
            size = int(kp.size)
            cv2.circle(debug_frame, (x, y), size, (0, 255, 0), 1)
            
        # 保存调试图像
        save_debug_image(debug_frame, "feature_extraction")
        
        print(f"成功检测到 {len(features['orb']['keypoints'])} 个特征点")
        return True
        
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return False
    finally:
        cap.release()

def test_temporal_alignment(video1_path, video2_path):
    """测试时间对齐功能"""
    print("\n测试时间对齐...")
    
    # 创建对齐器
    aligner = TemporalAligner(window_size=30)
    
    try:
        # 加载视频
        cap1, cap2, fps1, fps2 = aligner.normalize_frame_rates(video1_path, video2_path)
        
        # 提取特征序列
        extractor = EnhancedFeatureExtractor()
        features1 = []
        features2 = []
        
        # 读取30帧
        for _ in range(30):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
                
            feat1 = extractor.extract_features(frame1)
            feat2 = extractor.extract_features(frame2)
            
            features1.append(feat1)
            features2.append(feat2)
            
        # 计算对应关系
        alignment = aligner.compute_frame_correspondence(features1, features2)
        
        print(f"计算得到 {len(alignment)} 个帧对应关系")
        return True
        
    except Exception as e:
        print(f"时间对齐失败: {str(e)}")
        return False
    finally:
        cap1.release()
        cap2.release()

def test_spatial_alignment(video1_path, video2_path):
    """测试空间对齐功能"""
    print("\n测试空间对齐...")
    
    try:
        # 加载视频帧
        success1, cap1 = load_video(video1_path)
        success2, cap2 = load_video(video2_path)
        
        if not success1 or not success2:
            print("无法加载视频")
            return False
            
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("无法读取视频帧")
            return False
            
        # 提取特征并匹配
        extractor = EnhancedFeatureExtractor()
        features1 = extractor.extract_features(frame1)
        features2 = extractor.extract_features(frame2)
        
        matches = extractor.match_features(
            features1['orb']['descriptors'],
            features2['orb']['descriptors']
        )
        
        if len(matches) < 10:
            print("匹配点太少")
            return False
            
        # 估计变换
        aligner = SpatialAligner()
        src_pts = np.float32([
            features1['orb']['keypoints'][m.queryIdx].pt 
            for m in matches
        ]).reshape(-1, 1, 2)
        
        dst_pts = np.float32([
            features2['orb']['keypoints'][m.trainIdx].pt 
            for m in matches
        ]).reshape(-1, 1, 2)
        
        transform = aligner.estimate_transform(src_pts, dst_pts)
        
        if transform is None:
            print("无法估计变换矩阵")
            return False
            
        # 应用变换
        h, w = frame1.shape[:2]
        aligned_frame = aligner.apply_transform(frame2, transform, (w, h))
        
        # 保存结果
        debug_frame = np.hstack([frame1, aligned_frame])
        save_debug_image(debug_frame, "spatial_alignment")
        
        print("空间对齐成功")
        return True
        
    except Exception as e:
        print(f"空间对齐失败: {str(e)}")
        return False
    finally:
        cap1.release()
        cap2.release()

def test_complete_pipeline(video1_path, video2_path):
    """测试完整处理流程"""
    print("\n测试完整处理流程...")
    
    processor = VideoProcessor()
    
    def progress_callback(progress):
        print(f"处理进度: {progress:.1f}%")
        
    try:
        # 加载视频
        if not processor.load_videos(video1_path, video2_path, progress_callback):
            print("无法加载视频")
            return False
            
        # 开始处理
        processor.start_processing()
        
        # 处理几帧进行测试
        frames_processed = 0
        while processor.is_processing and frames_processed < 50:
            frame_info = processor.get_next_frame()
            if frame_info is not None:
                frames_processed += 1
                
                if frames_processed == 25:  # 保存中间帧
                    save_debug_image(
                        frame_info['frame'],
                        "complete_pipeline_result"
                    )
                    
        print(f"成功处理 {frames_processed} 帧")
        return True
        
    except Exception as e:
        print(f"处理流程失败: {str(e)}")
        return False
    finally:
        processor.release()

def main():
    """运行所有测试"""
    # 确保存在测试视频
    video1_path = "test_videos/video1.mp4"
    video2_path = "test_videos/video2.mp4"
    
    if not (os.path.exists(video1_path) and os.path.exists(video2_path)):
        print("请提供测试视频文件")
        return
        
    # 创建调试目录
    os.makedirs("debug", exist_ok=True)
    
    # 运行测试
    tests = [
        (test_feature_extraction, [video1_path]),
        (test_temporal_alignment, [video1_path, video2_path]),
        (test_spatial_alignment, [video1_path, video2_path]),
        (test_complete_pipeline, [video1_path, video2_path])
    ]
    
    total = len(tests)
    passed = 0
    
    for test_func, args in tests:
        if test_func(*args):
            passed += 1
            
    print(f"\n测试完成: {passed}/{total} 通过")

if __name__ == "__main__":
    main()
