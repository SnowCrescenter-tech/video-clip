# 核心模块包初始化文件
from core.feature_extractor import EnhancedFeatureExtractor
from core.temporal_aligner import TemporalAligner
from core.spatial_aligner import SpatialAligner
from core.video_processor import VideoProcessor

__all__ = [
    'EnhancedFeatureExtractor',
    'TemporalAligner',
    'SpatialAligner',
    'VideoProcessor'
]
