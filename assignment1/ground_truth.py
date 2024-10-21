import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import json

logging.basicConfig(level=logging.INFO)
class ImageLabeler():
    def __init__(self, ground_truth_folder: str):
        self.ground_truth_folder = ground_truth_folder
        self.class_mapping = {
            # (0, 0, 0): 'void',
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

    def get_gt_filename(self, img_filename):
        filename, ext = os.path.splitext(img_filename)
        return f"{filename}_GT{ext}"
    
    def load_img(self, selected_img: str):
        img_path = os.path.join(self.ground_truth_folder, selected_img)
        img = cv2.imread(self.get_gt_filename(img_path))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img
    
    def get_labels(self, selected_img: str) -> np.ndarray:
        labels_path = os.path.join(self.ground_truth_folder, 'labels.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels_dict = json.load(f)
            return labels_dict[selected_img]
        else:
            labels = set()
            rgb_img = self.load_img(selected_img)
            for rgb, label in self.class_mapping.items():
                mask = np.all(rgb_img == np.array(rgb), axis=-1)
                if np.any(mask):
                    labels.add(label)
            return list(labels)
    
    def get_all_labels(self):
        labels_path = os.path.join(self.ground_truth_folder, 'labels.json')
        if os.path.exists(labels_path):
            logging.info(f"Loading labels from {labels_path}...")
            with open(labels_path, 'r') as f:
                labels_dict = json.load(f)
        else:
            logging.info(f"No labels found. Computing labels for all images...")
            labels_dict = self.compute_all_labels()
            with open(labels_path, 'w') as f:
                json.dump(labels_dict, f, indent=4)
                logging.info(f"Labels saved to {labels_path}")
        return labels_dict

    def compute_all_labels(self):
        labels_dict = {}
        # Process all images in the directory
        for image_file in os.listdir(self.ground_truth_folder):
            if image_file.endswith('.bmp'):  # Assuming PNG format
                try:
                    labels = self.get_labels(image_file)
                    labels_dict[image_file] = labels
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        return labels_dict