import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from descriptors import Descriptor
import os
import logging
from typing import Dict
logging.basicConfig(level=logging.DEBUG)

class BoVW:
    def __init__(self, dataset_folder:str , descriptor_folder: str, vocab_size: int = 500, random_state: int = 42):
        self.DATASET_FOLDER = dataset_folder
        self.DESCRIPTOR_FOLDER = descriptor_folder
        self.vocab_size = vocab_size
        self.random_state = random_state
        self.codebook = None

    def extract_sift_features(self, img_path: str):
        # TOOD: Refactor this to use Descriptor class later
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT.create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        return descriptors

    def extract_all_sift_features(self) -> Dict[str, np.ndarray]:
        all_descriptors = {}
        logging.debug(f"Processing all images in {self.DATASET_FOLDER}")
        for filename in os.listdir(self.DATASET_FOLDER):
            if filename.endswith(".bmp"):  # Adjust file type as needed
                img_path = os.path.join(self.DATASET_FOLDER, filename)
                descriptors = self.extract_sift_features(img_path)
                if descriptors is not None:
                    all_descriptors[img_path] = descriptors
        self.save_sift_descriptors(all_descriptors)
        return all_descriptors

    def save_sift_descriptors(self, descriptors):
        logging.debug(f"Saving SIFT descriptors to {self.DESCRIPTOR_FOLDER}")
        save_folder = os.path.join(self.DESCRIPTOR_FOLDER, "SIFT_BoVW")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for img_path, descriptor in descriptors.items():
            # Generate the .npy file name based on the image file name
            filename = os.path.basename(img_path).replace(".bmp", ".npy")
            save_path = os.path.join(self.DESCRIPTOR_FOLDER, "SIFT_BoVW", filename)
            np.save(save_path, descriptor)
            logging.info(f"Saved descriptors for {img_path} to {save_path}")

    def build_codebook(self, img_paths: list):
        all_descriptors = self.extract_all_sift_features(img_paths)
        self.kmeans = KMeans(n_clusters=self.vocab_size, random_state=self.random_state)
        self.kmeans.fit(all_descriptors)
        self.codebook = self.kmeans.cluster_centers_

    def quantize_descriptors(self, descriptors: np.ndarray):
        words = self.kmeans.predict(descriptors)
        return words

    def build_histogram(self, img_path: str) -> np.ndarray:
        descriptors = self.extract_sift_features(img_path)
        words = self.quantize_descriptors(descriptors)
        histogram = np.histogram(words, bins=np.arange(self.vocab_size + 1))[0]
        return histogram/ np.sum(histogram)

    def build_histograms(self, img_paths: list) -> np.ndarray:
        histograms = [self.build_histogram(img_path) for img_path in img_paths]
        return np.array(histograms)


if __name__ == "__main__":
    bovw = BoVW(dataset_folder="MSRC_ObjCategImageDatabase_v2_local/Images",
                descriptor_folder="descriptors",
                vocab_size=500, random_state=42)
    bovw.extract_all_sift_features()
    logging.info("hello")