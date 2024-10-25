import unittest
from assignment1.feature_detectors import FeatureDetector
from assignment1.ground_truth import ImageLabeler

class TestFeatureDetectors(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img_labeler = ImageLabeler(DATASET_FOLDER)
        self.img1_filename = "1_1_s.bmp"
        self.fd = FeatureDetector("SIFT")
    
    def test_create_detector(self):
        self.assertIsNotNone(self.fd.detector, "The detector should not be None.")
    
    def test_detect_keypoints(self):
        # fd.detect_keypoints(self.img_labeler.load_img(self.img1_filename))
        pass