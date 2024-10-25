import cv2
import unittest
from feature_detectors import FeatureDetector
class TestFeatureDetectors(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img1 = cv2.imread(f"{DATASET_FOLDER}/Images/1_1_s.bmp")
        self.img2 = cv2.imread(f"{DATASET_FOLDER}/Images/5_24_s.bmp")
        self.fd = FeatureDetector("SIFT")
    
    def test_create_detector(self):
        self.assertIsNotNone(self.fd.detector, "The detector should not be None.")
    
    def test_detect_keypoints_compute_descriptors(self):
        keypoints1, descriptors1 = self.fd.detect_keypoints_compute_descriptors(self.img1)
        self.assertTrue(all(descriptor.shape[0] == 128 for descriptor in descriptors1), "Each descriptor should have a shape of 128.")
        keypoints2, descriptors2 = self.fd.detect_keypoints_compute_descriptors(self.img2)
        self.assertTrue(all(descriptor.shape[0] == 128 for descriptor in descriptors2), "Each descriptor should have a shape of 128.")
        self.assertNotEqual(descriptors1.shape, descriptors2.shape, "The descriptors should not have the same shape.")