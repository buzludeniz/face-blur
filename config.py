# ==================== config.py ====================
"""
Application configuration settings
"""

class Config:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Detection settings
    SCALE_FACTOR = 1.2
    MIN_NEIGHBORS = 5
    MIN_FACE_SIZE = (40, 40)
    FRAME_SKIP = 3
    
    # Blur settings
    BLUR_STRENGTH = 51
    BLUR_TYPE = 'box'
    PADDING_RATIO = 0.2
    
    # Optimization settings
    OVERLAP_THRESHOLD = 0.3
    RESIZE_FOR_DETECTION = 0.5
    USE_PROFILE_DETECTION = True