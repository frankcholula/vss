import unittest
import numpy as np
import cv2
from assignment1.descriptor import Descriptor, Extractor

class TestDescriptor(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img1 = cv2.imread(f"{DATASET_FOLDER}/Images/1_1_s.bmp").astype(np.float64) / 255.0
        self.img2 = cv2.imread(f"{DATASET_FOLDER}/Images/5_24_s.bmp").astype(np.float64) / 255.0
        self.descriptor = Descriptor(DATASET_FOLDER, "descriptors", "globalRGBquantization", base=4)

    def test_image_descriptor_mapping(self):
        self.descriptor.extract()
        mapping = self.descriptor.get_image_descriptor_mapping()
        for key, value in mapping.items():
            print(key, value.shape)
            # self.assertEqual(value.shape, (512,), "The shape of the descriptor should be (4,).")
        print(len(mapping))
        # self.assertEqual(len(mapping), 591, "The number of images should be 591.")


class TestExtractor(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img1 = cv2.imread(f"{DATASET_FOLDER}/Images/1_1_s.bmp").astype(np.float64) / 255.0
        self.img2 = cv2.imread(f"{DATASET_FOLDER}/Images/5_24_s.bmp").astype(np.float64) / 255.0
    
    def test_extract_globalRGBhisto_equality(self):
        result1 = Extractor.extract_globalRGBhisto(self.img1)
        result2 = Extractor.extract_globalRGBhisto(self.img2)


    def test_extract_globalRGBquantization_equality(self):
        result1 = Extractor.extract_globalRGBquantization(self.img1, 256)
        result2 = Extractor.extract_globalRGBquantization(self.img2, 256)

