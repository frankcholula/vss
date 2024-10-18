import os
import cv2
import numpy as np
from extractor import Extractor
from typing import Dict

class Descriptor:
    def __init__(self, dataset_folder: str, descriptor_folder: str, extract_method: str, **kwargs):
        print("Generating a new Descriptor object...")
        self.DATASET_FOLDER = dataset_folder
        self.DESCRIPTOR_FOLDER = descriptor_folder
        self.extract_method = extract_method
        self.AVAILABLE_EXTRACTORS = {
            'rgb': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'rgb'),
                'method': Extractor.extract_rgb
            },
            'random': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'random'),
                'method': Extractor.extract_random
            },
            'globalRGBhisto': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'globalRGBhisto'),
                'method': lambda img: Extractor.extract_globalRGBhisto(img, bins=kwargs.get('bins'))
            },
            'globalRGBencoding': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'globalRGBencoding'),
                'method': lambda img: Extractor.extract_globalRGBencoding(img, base=kwargs.get('base'))
            }

        }

    def extract(self):
        if self.extract_method not in self.AVAILABLE_EXTRACTORS:
            raise ValueError(f"Invalid extract_method: {self.extract_method}")

        descriptor_path = self.AVAILABLE_EXTRACTORS[self.extract_method]['path']
        if not os.path.exists(descriptor_path):
            # compute the descriptors if they don't exist, otherwise load them
            os.makedirs(descriptor_path, exist_ok=True)
            for filename in os.listdir(os.path.join(self.DATASET_FOLDER, 'Images')):
                if filename.endswith(".bmp"):
                    img_path = os.path.join(self.DATASET_FOLDER, 'Images', filename)
                    img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
                    fout = os.path.join(descriptor_path, filename).replace('.bmp', '.npy')
                    F = self.AVAILABLE_EXTRACTORS[self.extract_method]['method'](img)
                    np.save(fout, F)


    def get_image_descriptor_mapping(self) -> Dict[str, np.ndarray]:
        descriptor_path = os.path.join(self.DESCRIPTOR_FOLDER, self.extract_method)
        img_to_descriptor = {}
        for filename in os.listdir(descriptor_path):
            if filename.endswith('.npy'):
                img_path = os.path.join(self.DATASET_FOLDER, 'Images', filename.replace('.npy', '.bmp'))
                descriptor_data = np.load(os.path.join(descriptor_path, filename))
                img_to_descriptor[img_path] = descriptor_data
        return img_to_descriptor