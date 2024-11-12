import cv2
import os
import logging
import pickle
import numpy as np
from typing import Dict
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)


class BoVW:
    def __init__(
        self,
        dataset_folder: str,
        descriptor_folder: str,
        vocab_size: int = 500,
        random_state: int = 42,
    ):
        self.DATASET_FOLDER = dataset_folder
        self.DESCRIPTOR_FOLDER = descriptor_folder
        self.vocab_size = vocab_size
        self.random_state = random_state
        self.codebook = None
        self.idf_vector = None

    def extract_sift_features(self, img_path: str):
        # TOOD: Refactor this to use Descriptor class later
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT.create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        return descriptors

    def extract_all_sift_features(self) -> Dict[str, np.ndarray]:
        all_descriptors = {}
        save_descriptors = True
        sift_descriptor_folder = os.path.join(self.DESCRIPTOR_FOLDER, "SIFT")
        os.makedirs(sift_descriptor_folder, exist_ok=True)  # Ensure folder exists once
        logging.info(f"Processing all images in {self.DATASET_FOLDER}")
        for filename in os.listdir(self.DATASET_FOLDER):
            if filename.endswith(".bmp"):  # Adjust file type as needed
                img_path = os.path.join(self.DATASET_FOLDER, filename)
                descriptor_path = os.path.join(
                    sift_descriptor_folder, filename.replace(".bmp", ".npy")
                )

                if os.path.exists(descriptor_path):
                    save_descriptors = False
                    logging.debug(f"Loading SIFT descriptor from {descriptor_path}")
                    all_descriptors[img_path] = np.load(descriptor_path)
                    continue

                logging.info(f"Computing SIFT descriptor for {img_path}")
                descriptors = self.extract_sift_features(img_path)
                if descriptors is not None:
                    all_descriptors[img_path] = descriptors
        self.save_sift_descriptors(all_descriptors) if save_descriptors else None
        return all_descriptors

    def save_sift_descriptors(self, descriptors: Dict[str, np.ndarray]):
        save_folder = os.path.join(self.DESCRIPTOR_FOLDER, "SIFT")
        os.makedirs(save_folder, exist_ok=True)
        for img_path, descriptor in descriptors.items():
            filename = os.path.basename(img_path).replace(".bmp", ".npy")
            save_path = os.path.join(save_folder, filename)
            np.save(save_path, descriptor)
            logging.info(f"Saved descriptor for {img_path} to {save_path}")

    def build_codebook(self):
        codebook_folder = os.path.join(self.DESCRIPTOR_FOLDER, "SIFT_BoVW")
        codebook_path = os.path.join(codebook_folder, "kmeans.pkl")
        os.makedirs(codebook_folder, exist_ok=True)

        if os.path.exists(codebook_path):
            logging.debug("Codebook already exists. Loading existing codebook...")
            with open(codebook_path, "rb") as f:
                self.kmeans = pickle.load(f)
            self.codebook = self.kmeans.cluster_centers_
            logging.debug(f"Codebook loaded with size: {self.codebook.shape}.")
        else:
            logging.info("Building codebook...")
            all_descriptors = np.vstack(
                [
                    descriptors
                    for descriptors in self.extract_all_sift_features().values()
                    if descriptors is not None
                ]
            )
            if all_descriptors.size == 0:
                raise ValueError("No SIFT descriptors found to build codebook.")
            logging.info(
                f"Collected {all_descriptors.shape[0]} descriptors for clustering."
            )
            self.kmeans = KMeans(
                n_clusters=self.vocab_size, random_state=self.random_state
            )
            self.kmeans.fit(all_descriptors)
            self.codebook = self.kmeans.cluster_centers_
            with open(codebook_path, "wb") as f:
                pickle.dump(self.kmeans, f)
            logging.info(f"Codebook built and saved to {codebook_path}.")

        return self.codebook

    def quantize_descriptors(self, descriptors: np.ndarray):
        words = self.kmeans.predict(descriptors)
        return words

    def build_histogram(self, img_path: str) -> np.ndarray:
        descriptors = self.extract_sift_features(img_path)
        words = self.quantize_descriptors(descriptors)
        histogram = np.histogram(words, bins=np.arange(self.vocab_size + 1))[0]
        return histogram / np.sum(histogram)

    def build_histograms(self, img_paths: list) -> np.ndarray:
        histograms = [self.build_histogram(img_path) for img_path in img_paths]
        return np.array(histograms)

    def compute_idf(self, histograms: np.ndarray):
        n_images = histograms.shape[0]
        word_occurrence = np.sum(histograms > 0, axis=0)
        idf = np.log((1 + n_images) / (1 + word_occurrence)) + 1  # Smoothing
        self.idf_vector = idf
        logging.info(f"Computed IDF vector with shape: {idf.shape}")
        return idf

    def apply_tfidf(self, histograms: np.ndarray):
        if self.idf_vector is None:
            raise ValueError("IDF vector not computed. Call `compute_idf` first.")
        tfidf_histograms = histograms * self.idf_vector  # Element-wise multiplication
        tfidf_histograms = normalize(
            tfidf_histograms, norm="l2"
        )  # Normalize histograms
        logging.info("Applied TF-IDF weighting to histograms.")
        return tfidf_histograms

    def build_tf_idf_histogram(self, histogram: np.ndarray) -> np.ndarray:
        if self.idf_vector is None:
            raise ValueError("IDF vector not computed. Call `compute_idf` first.")
        tfidf_histogram = histogram * self.idf_vector
        tfidf_histogram = tfidf_histogram / np.linalg.norm(
            tfidf_histogram, ord=2
        )  # L2 normalization

        return tfidf_histogram

    def build_tfidf_histograms(self, img_paths: list) -> np.ndarray:
        histograms = self.build_histograms(img_paths)
        if self.idf_vector is None:
            self.compute_idf(histograms)

        tfidf_histograms = [
            self.build_tf_idf_histogram(histogram) for histogram in histograms
        ]
        return np.array(tfidf_histograms)

    def save_tfidf_histograms(self, img_paths: list, tfidf_histograms: np.ndarray):
        tfidf_folder = os.path.join(self.DESCRIPTOR_FOLDER, "tfidf")
        os.makedirs(tfidf_folder, exist_ok=True)  # Create folder if it doesn't exists
        for img_path, tfidf_histogram in zip(img_paths, tfidf_histograms):
            filename = os.path.basename(img_path).replace(".bmp", ".npy")
            save_path = os.path.join(tfidf_folder, filename)
            np.save(save_path, tfidf_histogram)
            logging.info(f"Saved TF-IDF histogram for {img_path} to {save_path}")


if __name__ == "__main__":
    bovw = BoVW(
        dataset_folder="MSRC_ObjCategImageDatabase_v2_local/Images",
        descriptor_folder="descriptors",
        vocab_size=500,
        random_state=42,
    )

    bovw.build_codebook()
    img_paths = [
        os.path.join(bovw.DATASET_FOLDER, fname)
        for fname in os.listdir(bovw.DATASET_FOLDER)
        if fname.endswith(".bmp")
    ]
    histograms = bovw.build_histograms(img_paths)
    bovw.compute_idf(histograms)
    query_image = "MSRC_ObjCategImageDatabase_v2_local/Images/1_1_s.bmp"
    query_histogram = bovw.build_histogram(query_image)
    tfidf_query_histogram = bovw.build_tf_idf_histogram(query_histogram)
    logging.info(f"TF-IDF query histogram shape: {tfidf_query_histogram.shape}")

    tfidf_histograms = bovw.build_tfidf_histograms(img_paths)
    logging.info(f"TF-IDF histograms shape: {tfidf_histograms.shape}")
    bovw.save_tfidf_histograms(img_paths, tfidf_histograms)
    logging.info("TF-IDF histograms saved successfully.")
