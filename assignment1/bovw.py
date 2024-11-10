import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC

class BoVW:
    def __init__(self, vocab_size: int = 500, random_state: int = 42):
        self.vocab_size = vocab_size
        self.random_state = random_state
        self.codebook = None

    def extract_sift_features(self, img_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT.create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        return descriptors

    def extract_all_sift_features(self, img_paths: list):
        all_descriptors = []
        for img_path in img_paths:
            descriptors = self.extract_sift_features(img_path)
            all_descriptors.append(descriptors)
        return np.vstack(all_descriptors)

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


if __name__ == "main":
    bovw = BoVW()