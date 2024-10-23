import unittest
import numpy as np
import cv2
from assignment1.ground_truth import ImageLabeler

class TestImageLabeler(unittest.TestCase):
    def setUp(self):
        DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
        self.img_labeler = ImageLabeler(DATASET_FOLDER)
        self.img1_filename = "1_1_s.bmp"

    
    def test_labels_path(self):
        correct_path = "MSRC_ObjCategImageDatabase_v2_local/GroundTruth/labels.json"
        self.assertEqual(
            self.img_labeler.labels_path,
            correct_path,
            f"The labels path should be {correct_path}."
        )
    def test_get_gt_filename(self):
        correct_filename = "1_1_s_GT.bmp"
        gt_filename = self.img_labeler.get_gt_filename(self.img1_filename)
        self.assertEqual(
            gt_filename,
            correct_filename, 
            f"The ground truth filename should be {correct_filename}."
        )
    
    def test_load_img(self):
        img = self.img_labeler.load_img(self.img1_filename)
        self.assertEqual(
            img.shape[-1],
            3,
            "The image should have 3 channels."
        )
    
    def test_get_labels(self):
        labels = self.img_labeler.get_labels(self.img1_filename)
        self.assertIsInstance(labels, list, "The labels should be a list.")
        self.assertGreater(len(labels), 0, "The labels list should not be empty.")


    def test_compute_all_labels(self):
        labels_dict = self.img_labeler.compute_all_labels()
        self.assertIsInstance(labels_dict, dict, "The labels should be a dictionary.")
        self.assertGreater(len(labels_dict), 0, "The labels dictionary should not be empty.")
    
    def test_get_all_labels(self):
        results = self.img_labeler.get_all_labels()
        self.assertIsInstance(results, dict, "The results should be a dictionary.")
        self.assertGreater(len(results), 0, "The results dictionary should not be empty.")
