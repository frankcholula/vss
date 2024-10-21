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
            'globalRGBquantization': {
                'path': os.path.join(self.DESCRIPTOR_FOLDER, 'globalRGBquantization'),
                'method': lambda img: Extractor.extract_globalRGBquantization(img, quant_lvl=kwargs.get('quant_lvl')),
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
        """
        Generate a random row vector with 30 elements.
        
        This function returns a row vector [rand rand .... rand] representing 
        an image descriptor computed from the image 'img'.
        
        Note: 'img' is expected to be a normalized RGB image (colors range [0,1] not [0,255]).
        
        Parameters:
        img (numpy.ndarray): The input image.
        
        Returns:
        numpy.ndarray: A random row vector with 30 elements.
        """
        return np.random.rand(1, 30)
    
    @staticmethod
    def extract_rgb(img) -> np.ndarray:
        """
        Compute the average red, green, and blue values as a basic color descriptor.
        
        This function calculates the average values for the blue, green, and red channels
        of the input image and returns them as a feature vector.
        
        Note: OpenCV uses BGR format, so the channels are accessed in the order B, G, R.
        
        Parameters:
        img (numpy.ndarray): The input image.
        
        Returns:
        numpy.ndarray: A feature vector containing the average B, G, and R values.
        """
        img = img / 255.0 # normalize the image to [0, 1]
        B = np.mean(img[:, :, 0])
        G = np.mean(img[:, :, 1])
        R = np.mean(img[:, :, 2])
        return np.array([R, G, B])
    
    @staticmethod
    def extract_globalRGBhisto(img, bins=32) -> np.ndarray:
        hist = [np.histogram(img[:, :, i], bins=bins, range=(0, 256))[0] for i in range(3)]
        hist_flat = np.concatenate(hist)
        hist_normalized = hist_flat / np.sum(hist_flat)
        return hist_normalized

    @staticmethod
    def extract_globalRGBquantization(img, quant_lvl) -> np.ndarray:
        # Quantize the RGB values
        R = np.floor(img[:, :, 0] / quant_lvl).astype(int)
        G = np.floor(img[:, :, 1] / quant_lvl).astype(int)
        B = np.floor(img[:, :, 2] / quant_lvl).astype(int)

        poly_repr = R * quant_lvl ** 2 + G * quant_lvl + B
        hist = np.histogram(poly_repr, bins=np.arange(quant_lvl**3+1), density=False)[0]
        hist_flat = hist.flatten()
        hist_normalized = hist_flat / np.sum(hist_flat)
        return hist_normalized
