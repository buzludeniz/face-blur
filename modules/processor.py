# ==================== modules/processor.py ====================
"""
Face processing module
"""

import numpy as np
from typing import Tuple


class FaceProcessor:
    """Handles face region processing and merging"""
    
    @staticmethod
    def merge_overlapping(faces: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Merge overlapping face detections"""
        if len(faces) == 0:
            return faces
        
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        merged = []
        
        for face in faces:
            x1, y1, w1, h1 = face
            overlap_found = False
            
            for i, (x2, y2, w2, h2) in enumerate(merged):
                x_l = max(x1, x2)
                y_t = max(y1, y2)
                x_r = min(x1 + w1, x2 + w2)
                y_b = min(y1 + h1, y2 + h2)
                
                if x_r > x_l and y_b > y_t:
                    intersection = (x_r - x_l) * (y_b - y_t)
                    overlap = intersection / min(w1 * h1, w2 * h2)
                    
                    if overlap > threshold:
                        x_new = min(x1, x2)
                        y_new = min(y1, y2)
                        merged[i] = [x_new, y_new, 
                                   max(x1+w1, x2+w2) - x_new,
                                   max(y1+h1, y2+h2) - y_new]
                        overlap_found = True
                        break
            
            if not overlap_found:
                merged.append(list(face))
        
        return np.array(merged)
    
    @staticmethod
    def add_padding(x: int, y: int, w: int, h: int, 
                   frame_shape: Tuple, ratio: float = 0.2) -> Tuple[int, int, int, int]:
        """Add padding around face region"""
        pad = int(ratio * w)
        x_pad = max(0, x - pad)
        y_pad = max(0, y - pad)
        w_pad = min(frame_shape[1] - x_pad, w + 2 * pad)
        h_pad = min(frame_shape[0] - y_pad, h + 2 * pad)
        return x_pad, y_pad, w_pad, h_pad
    
    @staticmethod
    def scale_faces(faces: np.ndarray, scale: float) -> np.ndarray:
        """Scale face coordinates back to original size"""
        if len(faces) == 0:
            return faces
        return (faces / scale).astype(int)