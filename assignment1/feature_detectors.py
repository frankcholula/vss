import cv2
import numpy as np
class FeatureDetector:
    def __init__(self, detector_type: str):
        self.detector_type = detector_type
        self.detector = self.create_detector()

    def create_detector(self):
        match self.detector_type:
            case "SIFT":
                return cv2.SIFT_create()
            case "Harris":
                return None
            case _:
                raise ValueError(f"Invalid detector type: {self.detector_type}")

    def detect_keypoints_compute_descriptors(self, img: np.ndarray):
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        match self.detector_type:
            case "SIFT":
                detector = cv2.SIFT_create()
                keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            case "Harris":
                cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
            case _:
                raise ValueError(f"Invalid detector type: {self.detector_type}")
        return keypoints, descriptors
