import unittest
import numpy as np
import cv2
from assignment1.descriptor import Descriptor

class TestDescriptor(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img1 = cv2.imread(f"{DATASET_FOLDER}/Images/1_1_s.bmp").astype(np.float64) / 255.0
        self.img2 = cv2.imread(f"{DATASET_FOLDER}/Images/5_24_s.bmp").astype(np.float64) / 255.0
        self.descriptor = Descriptor(DATASET_FOLDER, "descriptors", "globalRGBencoding", base=256)

    def test_image_descriptor_mapping(self):
        self.descriptor.extract()
        mapping = self.descriptor.get_image_descriptor_mapping()
        self.assertEqual(len(mapping), 591, "The number of images should be 591.")