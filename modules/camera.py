# ==================== modules/camera.py ====================
"""
Camera management module
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from config import Config


class Camera:
    """Handles camera operations"""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror
        return ret, frame
    
    def release(self):
        """Release camera"""
        self.cap.release()

