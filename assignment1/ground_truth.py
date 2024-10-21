import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageLabeler():
    def __init__(self, ground_truth_folder: str):
        self.ground_truth_folder = ground_truth_folder
        self.class_mapping = {
            (0, 0, 0): 'void',
            (128, 0, 0): 'building',
            (0, 128, 0): 'grass',
            (128, 128, 0): 'tree',
            (0, 0, 128): 'cow',
            (128, 0, 128): 'horse',
            (0, 128, 128): 'sheep',
            (128, 128, 128): 'sky',
            (64, 0, 0): 'mountain',
            (192, 0, 0): 'aeroplane',
            (64, 128, 0): 'water',
            (192, 128, 0): 'face',
            (64, 0, 128): 'car',
            (192, 0, 128): 'bicycle',
            (64, 128, 128): 'flower',
            (192, 128, 128): 'sign',
            (0, 64, 0): 'bird',
            (128, 64, 0): 'book',
            (0, 192, 0): 'chair',
            (128, 64, 128): 'road',
            (0, 192, 128): 'cat',
            (128, 192, 128): 'dog',
            (64, 64, 0): 'body',
            (192, 64, 0): 'boat'
        }
    
    def load_img (self, selected_img: str):
        filename, ext = os.path.splitext(selected_img)
        gt_filename = f"{filename}_GT{ext}"
        img = cv2.imread(os.path.join(self.ground_truth_folder, gt_filename))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img
    
    def get_labels(self, selected_img: str) -> np.ndarray:
        labels = set()
        rgb_img = self.load_img(selected_img)
        for rgb, label in self.class_mapping.items():
            mask = np.all(rgb_img == np.array(rgb), axis=-1)
            if np.any(mask):
                labels.add(label)
        return labels