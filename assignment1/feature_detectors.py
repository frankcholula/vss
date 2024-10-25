import cv2
import numpy as np
class FeatureDetector:
    def __init__(self, detector_type: str):
        self.detector_type = detector_type
        self.detector = self.create_detector()

    def create_detector(self):
        # Create the detector object based on the detector_type
        match self.detector_type:
            case "SIFT":
                detector = cv2.SIFT_create()
            case "Harris":
                detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
            case _:
                raise ValueError(f"Invalid detector type: {self.detector_type}")
        return detector
    
    def detect_keypoints_compute_descriptors(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors