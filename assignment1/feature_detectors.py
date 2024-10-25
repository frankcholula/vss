import cv2
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
    
