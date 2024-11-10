import unittest
import numpy as np
import cv2
from descriptors import Descriptor

class TestBoVW(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img1 = (
            cv2.imread(f"{DATASET_FOLDER}/Images/1_1_s.bmp").astype(np.float64) / 255.0
        )
        self.img2 = (
            cv2.imread(f"{DATASET_FOLDER}/Images/5_24_s.bmp").astype(np.float64) / 255.0
        )
        self.descriptor1 = Descriptor(
            DATASET_FOLDER, "descriptors", "globalRGBhisto_quant", quant_lvl=4
        )
        self.descriptor2 = Descriptor(
            DATASET_FOLDER, "descriptors", "globalRGBhisto_quant", quant_lvl=8
        )
