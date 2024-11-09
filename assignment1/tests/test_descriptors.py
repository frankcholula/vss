import unittest
import numpy as np
import cv2
from descriptors import Descriptor, Extractor


class TestDescriptors(unittest.TestCase):
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

    def test_image_descriptor_mapping(self):
        self.descriptor1.extract(recompute=True)
        mapping = self.descriptor1.get_image_descriptor_mapping()
        for key, value in mapping.items():
            self.assertEqual(
                value.shape, (64,), "The shape of the descriptor should be (64,)."
            )
        self.assertEqual(len(mapping), 591, "The number of images should be 591.")
        self.descriptor2.extract(recompute=True)
        mapping = self.descriptor2.get_image_descriptor_mapping()
        for key, value in mapping.items():
            self.assertEqual(
                value.shape, (512,), "The shape of the descriptor should be (512,)."
            )
        self.assertEqual(len(mapping), 591, "The number of images should be 591.")

    def test_perform_pca(self):
        self.descriptor1.extract(recompute=True)
        original_descriptors = self.descriptor1.get_image_descriptor_mapping()
        for key, value in original_descriptors.items():
            self.assertEqual(
                value.shape, (64,), "The shape of the descriptor should be (64,)."
            )
        reduced_descriptors = self.descriptor1.perform_pca(variance_ratio=0.95)
        for key, value in reduced_descriptors.items():
            self.assertLess(
                value.shape, (64,), "The shape of the descriptor should be (2,)."
            )


class TestExtractors(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img1 = (
            cv2.imread(f"{DATASET_FOLDER}/Images/1_1_s.bmp").astype(np.float64) / 255.0
        )
        # specifically choose an image of different sizes
        self.img2 = (
            cv2.imread(f"{DATASET_FOLDER}/Images/16_19_s.bmp").astype(np.float64)
            / 255.0
        )

    def test_extract_globalRGBhisto_equality(self):
        result1 = Extractor.extract_globalRGBhisto(self.img1)
        result2 = Extractor.extract_globalRGBhisto(self.img2)
        self.assertTrue(
            np.array_equal(result1, result2),
            "The histograms extracted should be equal.",
        )

    def test_extract_globalRGBhisto_quant_equality(self):
        result1 = Extractor.extract_globalRGBhisto_quant(self.img1, 8)
        result2 = Extractor.extract_globalRGBhisto_quant(self.img2, 8)
        self.assertTrue(
            result1.shape[0] == 512, "The Dimension of the histogram should be 512."
        )
        self.assertTrue(
            np.array_equal(result1, result2),
            "The histograms extracted should be equal.",
        )

    def test_extract_gridRGB(self):
        result1 = Extractor.extract_gridRGB(self.img1, 4)
        result2 = Extractor.extract_gridRGB(self.img2, 8)
        # 4 by 4 grid, 3 channels = 48 final length
        self.assertEqual(result1.shape[0], 48, "The dimension of result1 should be 48.")
        # 8 by 8 grid, 3 channels = 192 final length
        self.assertEqual(
            result2.shape[0], 192, "The dimension of result2 should be 48."
        )

    def test_extract_gridEOhisto(self):
        result1 = Extractor.extract_gridEOhisto(self.img1, 4, 3)
