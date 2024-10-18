import unittest
import numpy as np
import cv2

from assignment1.descriptor import Extractor

class TestExtractor(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img1 = cv2.imread(f"{DATASET_FOLDER}/Images/1_1_s.bmp").astype(np.float64) / 255.0
        self.img2 = cv2.imread(f"{DATASET_FOLDER}/Images/5_24_s.bmp").astype(np.float64) / 255.0
    
    def test_extract_globalRGBhisto_equality(self):
        result1 = Extractor.extract_globalRGBhisto(self.img1)
        result2 = Extractor.extract_globalRGBhisto(self.img2)
        self.assertFalse(np.array_equal(result1, result2), "The histograms extracted should not be equal.")


    def test_extract_globalRGBencoding_equality(self):
        result1 = Extractor.extract_globalRGBencoding(self.img1, 256)
        result2 = Extractor.extract_globalRGBencoding(self.img2, 256)
        self.assertFalse(np.array_equal(result1, result2), "The histograms extracted should not be equal.")
