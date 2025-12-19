# ==================== modules/blur.py ====================
"""
Blur effects module
"""

import cv2
import numpy as np


class BlurEffect:
    """Handles different blur effects"""
    
    @staticmethod
    def apply(frame: np.ndarray, x: int, y: int, w: int, h: int, 
             blur_type: str = 'box', strength: int = 51) -> np.ndarray:
        """Apply blur to specified region"""
        roi = frame[y:y+h, x:x+w]
        
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(roi, (strength, strength), 0)
        elif blur_type == 'box':
            return cv2.blur(roi, (strength, strength))
        elif blur_type == 'pixelate':
            small = cv2.resize(roi, (w//15, h//15), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return roi
