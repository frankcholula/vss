import os
import cv2
import numpy as np
from typing import Dict
import logging
logging.basicConfig(level=logging.INFO)

class Descriptor:
    def __init__(self, dataset_folder: str, descriptor_folder: str, extract_method: str, **kwargs):
        logging_message = f"Using the Descriptor object with {extract_method} "
        self.DATASET_FOLDER = dataset_folder
        self.DESCRIPTOR_FOLDER = descriptor_folder
        self.extract_method = extract_method
        self.AVAILABLE_EXTRACTORS = {
            'rgb': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'rgb'),
                'method': Extractor.extract_rgb,
                'log_message': logging_message
            },
            'random': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'random'),
                'method': Extractor.extract_random,
                'log_message': logging_message
            },
            'globalRGBhisto': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'globalRGBhisto'),
                'method': lambda img: Extractor.extract_globalRGBhisto(img, bins=kwargs.get('bins')),
                'log_message': logging_message + f"{kwargs}"
            },
            'globalRGBhisto_quant': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'globalRGBhisto_quant'),
                'method': lambda img: Extractor.extract_globalRGBhisto_quant(img, quant_lvl=kwargs.get('quant_lvl')),
                'log_message': logging_message + f"{kwargs}"
            },
            'gridRGB': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'gridRGB'),
                'method': lambda img: Extractor.extract_gridRGB(img, grid_size=kwargs.get('grid_size')),
                'log_message': logging_message + f"{kwargs}"
            }
        }
        logging.info(self.AVAILABLE_EXTRACTORS[self.extract_method]['log_message'])

    def extract(self, recompute: bool = False):
        if self.extract_method not in self.AVAILABLE_EXTRACTORS:
            raise ValueError(f"Invalid extract_method: {self.extract_method}")

        descriptor_path = self.AVAILABLE_EXTRACTORS[self.extract_method]['path']
        if not os.path.exists(descriptor_path) or recompute:
            # compute the descriptors if they don't exist, otherwise load them
            os.makedirs(descriptor_path, exist_ok=True)
            for filename in os.listdir(os.path.join(self.DATASET_FOLDER, 'Images')):
                if filename.endswith(".bmp"):
                    img_path = os.path.join(self.DATASET_FOLDER, 'Images', filename)
                    img = cv2.imread(img_path).astype(np.float64)
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

class Extractor:
    @staticmethod
    def extract_random(img) -> np.ndarray:
        return np.random.rand(1, 30)
    
    @staticmethod
    def extract_rgb(img) -> np.ndarray:
        img = img / 255.0 # normalize the image to [0, 1]
        B = np.mean(img[:, :, 0])
        G = np.mean(img[:, :, 1])
        R = np.mean(img[:, :, 2])
        return np.array([R, G, B])
    
    @staticmethod
    def extract_gridRGB(img, grid_size:int = 4) -> np.ndarray:
        img_height, img_width, img_channel = img.shape
        grid_height = img_height // grid_size
        grid_width = img_width // grid_size
        grid_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid_cell = img[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width, :]
                R = np.mean(grid_cell[:, :, 0])
                G = np.mean(grid_cell[:, :, 1])
                B = np.mean(grid_cell[:, :, 2])
                grid_features.extend([R, G, B])
        return np.array(grid_features)
    
    @staticmethod
    def extract_gridEOhisto(img, grid_size:int = 4) -> np.ndarray:
        pass

    @staticmethod
    def extract_grid_combined(img, grid_size:int = 4) -> np.ndarray:
        pass

    @staticmethod
    def extract_globalRGBhisto(img, bins=32) -> np.ndarray:
        hist = [np.histogram(img[:, :, i], bins=bins, range=(0, 256))[0] for i in range(3)]
        hist_flat = np.concatenate(hist)
        hist_normalized = hist_flat / np.sum(hist_flat)
        return hist_normalized

    @staticmethod
    def extract_globalRGBhisto_quant(img, quant_lvl=4) -> np.ndarray:
        # Quantize the RGB values
        R = np.floor(img[:, :, 0] * quant_lvl / 256).astype(int)
        G = np.floor(img[:, :, 1] * quant_lvl / 256).astype(int)
        B = np.floor(img[:, :, 2] * quant_lvl / 256).astype(int)

        poly_repr = R * quant_lvl ** 2 + G * quant_lvl + B
        hist = np.histogram(poly_repr, bins=np.arange(quant_lvl**3+1), density=False)[0]
        hist_flat = hist.flatten()
        hist_normalized = hist_flat / np.sum(hist_flat)
        return hist_normalized
