import unittest
import numpy as np
import cv2
from retrievers import Retriever
from descriptors import Extractor

class TestRetrievers(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img1 = cv2.imread(f"{DATASET_FOLDER}/Images/1_1_s.bmp").astype(np.float64) / 255.0
        self.img2 = cv2.imread(f"{DATASET_FOLDER}/Images/5_24_s.bmp").astype(np.float64) / 255.0
        base = 256
        self.desc1 = Extractor.extract_globalRGBhisto_quant(self.img1, base)
        self.desc2 = Extractor.extract_globalRGBhisto_quant(self.img2, base)

    def test_cvpr_compare(self):
        result = Retriever.cvpr_compare(self.desc1, self.desc2, metric="L2")
        self.assertTrue(result >= 0, "The CVPR comparison should be greater than or equal to 0.")
