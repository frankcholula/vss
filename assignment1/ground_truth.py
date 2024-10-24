import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from typing import Dict, List

logging.basicConfig(level=logging.INFO)


class ImageLabeler:
    def __init__(self, dataset_folder: str):
        self.ground_truth_folder = os.path.join(dataset_folder, "GroundTruth")
        self.image_files = os.path.join(dataset_folder, "Images")
        self.labels_path = os.path.join(self.ground_truth_folder, "labels.json")
        self.label_mapping = {
            # (0, 0, 0): 'void',
            (128, 0, 0): "building",
            (0, 128, 0): "grass",
            (128, 128, 0): "tree",
            (0, 0, 128): "cow",
            (128, 0, 128): "horse",
            (0, 128, 128): "sheep",
            (128, 128, 128): "sky",
            (64, 0, 0): "mountain",
            (192, 0, 0): "aeroplane",
            (64, 128, 0): "water",
            (192, 128, 0): "face",
            (64, 0, 128): "car",
            (192, 0, 128): "bicycle",
            (64, 128, 128): "flower",
            (192, 128, 128): "sign",
            (0, 64, 0): "bird",
            (128, 64, 0): "book",
            (0, 192, 0): "chair",
            (128, 64, 128): "road",
            (0, 192, 128): "cat",
            (128, 192, 128): "dog",
            (64, 64, 0): "body",
            (192, 64, 0): "boat",
        }

        self.class_mapping = {
            1: ["grass", "cow"],
            2: ["tree", "grass", "sky"],
            3: ["building", "sky"],
            4: ["aeroplane", "grass", "sky"],
            5: ["cow", "grass", "mountain"],
            6: ["face", "body"],
            7: ["car", "building"],
            8: ["bike", "building"],
            9: ["sheep", "grass"],
            10: ["flower"],
            11: ["sign"],
            12: ["bird", "sky", "grass", "water"],
            13: ["book"],
            14: ["chair"],
            15: ["cat"],
            16: ["dog"],
            17: ["road", "building"],
            18: ["water", "boat"],
            19: ["body", "face"],
            20: ["water", "boat", "sky", "mountain"],
        }

    def get_gt_filename(self, img_filename: str) -> str:
        filename, ext = os.path.splitext(img_filename)
        return f"{filename}_GT{ext}"

    def load_img(self, selected_img: str):
        img_path = os.path.join(self.ground_truth_folder, selected_img)
        img = cv2.imread(self.get_gt_filename(img_path))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img

    def get_labels(self, selected_img: str) -> np.ndarray:
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                labels_dict = json.load(f)
            return labels_dict[selected_img]["labels"]
        else:
            labels = set()
            rgb_img = self.load_img(selected_img)
            for rgb, label in self.label_mapping.items():
                mask = np.all(rgb_img == np.array(rgb), axis=-1)
                if np.any(mask):
                    labels.add(label)
            return list(labels)

    def get_class(self, selected_img: str) -> str:
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                labels_dict = json.load(f)
            return labels_dict[selected_img]["class"]
        else:
            return selected_img.split("_")[0]

    def get_labels_dict(self) -> Dict[str, Dict]:
        labels_dict = {}
        if os.path.exists(self.labels_path):
            labels_dict = self.load_labels()
        else:
            logging.info(f"No labels found. Computing labels for all images...")
            labels_dict = self.compute_all_labels()
            self.save_labels(labels_dict)
        return labels_dict

    def load_labels(self):
        with open(self.labels_path, "r") as f:
            return json.load(f)

    def save_labels(self, labels_dict):
        with open(self.labels_path, "w") as f:
            json.dump(labels_dict, f, indent=4)
            logging.info(f"Labels saved to {self.labels_path}")

    def compute_all_labels(self) -> Dict[str, Dict]:
        result = {}
        for image_file in os.listdir(self.image_files):
            if image_file.endswith(".bmp"):
                class_id = image_file.split("_")[0]
                try:
                    labels = self.get_labels(image_file)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
            result[image_file] = {"labels": labels, "class": class_id}
        return result
