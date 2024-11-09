import os
import cv2
import numpy as np
from typing import Dict
import logging
from feature_detectors import FeatureDetector
from sklearn.decomposition import PCA

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class Descriptor:
    def __init__(
        self, dataset_folder: str, descriptor_folder: str, extract_method: str, **kwargs
    ):
        logging_message = f"Using the Descriptor object with {extract_method} "
        self.DATASET_FOLDER = dataset_folder
        self.DESCRIPTOR_FOLDER = descriptor_folder
        self.extract_method = extract_method
        if kwargs.get("feature_detector") == "SIFT":
            self.feature_detector = FeatureDetector("SIFT")
        # TODO: add new descriptors here
        self.AVAILABLE_EXTRACTORS = {
            "rgb": {
                "path": os.path.join(self.DESCRIPTOR_FOLDER, "rgb"),
                "method": Extractor.extract_rgb,
                "log_message": logging_message,
            },
            "random": {
                "path": os.path.join(self.DESCRIPTOR_FOLDER, "random"),
                "method": Extractor.extract_random,
                "log_message": logging_message,
            },
            "globalRGBhisto": {
                "path": os.path.join(self.DESCRIPTOR_FOLDER, "globalRGBhisto"),
                "method": lambda img: Extractor.extract_globalRGBhisto(
                    img, bins=kwargs.get("bins")
                ),
                "log_message": logging_message + f"{kwargs}",
            },
            "globalRGBhisto_quant": {
                "path": os.path.join(self.DESCRIPTOR_FOLDER, "globalRGBhisto_quant"),
                "method": lambda img: Extractor.extract_globalRGBhisto_quant(
                    img, quant_lvl=kwargs.get("quant_lvl")
                ),
                "log_message": logging_message + f"{kwargs}",
            },
            "gridRGB": {
                "path": os.path.join(self.DESCRIPTOR_FOLDER, "gridRGB"),
                "method": lambda img: Extractor.extract_gridRGB(
                    img, grid_size=kwargs.get("grid_size")
                ),
                "log_message": logging_message + f"{kwargs}",
            },
            "gridEOhisto": {
                "path": os.path.join(self.DESCRIPTOR_FOLDER, "gridEOhisto"),
                "method": lambda img: Extractor.extract_gridEOhisto(
                    img,
                    grid_size=kwargs.get("grid_size"),
                    sobel_filter_size=kwargs.get("sobel_filter_size"),
                    ang_quant_lvl=kwargs.get("ang_quant_lvl"),
                ),
                "log_message": logging_message + f"{kwargs}",
            },
            "gridCombined": {
                "path": os.path.join(self.DESCRIPTOR_FOLDER, "gridCombined"),
                "method": lambda img: Extractor.extract_grid_combined(
                    img,
                    grid_size=kwargs.get("grid_size"),
                    sobel_filter_size=kwargs.get("sobel_filter_size"),
                    ang_quant_lvl=kwargs.get("ang_quant_lvl"),
                    norm_method=kwargs.get("norm_method"),
                ),
                "log_message": logging_message + f"{kwargs}",
            },
            "SIFT": {
                "path": os.path.join(self.DESCRIPTOR_FOLDER, "SIFT"),
                # TODO: implement here
                "method":lambda img: self.feature_detector.detect_keypoints_compute_descriptors(img)[1],
                "log_message": logging_message + "using SIFT"
            }
        }
        LOGGER.debug(self.AVAILABLE_EXTRACTORS[self.extract_method]["log_message"])
        self.descriptors = None

    def extract(self, recompute: bool = False):
        if self.extract_method not in self.AVAILABLE_EXTRACTORS:
            raise ValueError(f"Invalid extract_method: {self.extract_method}")

        descriptor_path = self.AVAILABLE_EXTRACTORS[self.extract_method]["path"]
        if not os.path.exists(descriptor_path) or recompute:
            # compute the descriptors if they don't exist, otherwise load them
            os.makedirs(descriptor_path, exist_ok=True)
            descriptors = {}
            for filename in os.listdir(os.path.join(self.DATASET_FOLDER, "Images")):
                if filename.endswith(".bmp"):
                    img_path = os.path.join(self.DATASET_FOLDER, "Images", filename)
                    # TODO: maybe change the image loading functionalities uint8 or float32
                    img = cv2.imread(img_path).astype(np.float64)

                    F = self.AVAILABLE_EXTRACTORS[self.extract_method]["method"](img)
                    descriptors[img_path] = F
            self.save_descriptors(descriptors, descriptor_path)

    def save_descriptors(self, descriptors: Dict[str, np.ndarray], save_path: str):
        """
        Save descriptors to the specified path.

        Args:
            descriptors (Dict[str, np.ndarray]): A dictionary of descriptors to save.
            save_path (str): Directory where the descriptors will be saved.
        """
        os.makedirs(save_path, exist_ok=True)
        for img_path, descriptor in descriptors.items():
            file_name = os.path.basename(img_path).replace(".bmp", ".npy")
            np.save(os.path.join(save_path, file_name), descriptor)
        LOGGER.info(f"Saved descriptors to {save_path}")

    def get_image_descriptor_mapping(self) -> Dict[str, np.ndarray]:
        descriptor_path = os.path.join(self.DESCRIPTOR_FOLDER, self.extract_method)
        img_to_descriptor = {}
        for filename in os.listdir(descriptor_path):
            if filename.endswith(".npy"):
                img_path = os.path.join(
                    self.DATASET_FOLDER, "Images", filename.replace(".npy", ".bmp")
                )
                descriptor_data = np.load(os.path.join(descriptor_path, filename))
                img_to_descriptor[img_path] = descriptor_data
        self.descriptors = img_to_descriptor
        return img_to_descriptor

    def perform_pca(self, n_components: int = None, variance_ratio: float = 0.99) -> Dict[str, np.ndarray]:
        descriptor_pca_path = os.path.join(self.DESCRIPTOR_FOLDER, self.extract_method + "_pca")
        if self.descriptors is None:
            raise ValueError("Descriptors have not been extracted yet.")
        descriptor_list = []
        image_paths = []
        for img_path, descriptor in self.descriptors.items():
            descriptor_list.append(descriptor)
            image_paths.append(img_path)
        descriptor_matrix = np.vstack(descriptor_list)

        # Initialize PCA
        if n_components is None:
            pca = PCA(n_components=variance_ratio)
        else:
            pca = PCA(n_components=n_components)

        # Fit PCA and transform descriptors
        reduced_matrix = pca.fit_transform(descriptor_matrix)
        LOGGER.info(f"PCA reduced dimensions from {descriptor_matrix.shape[1]} to {reduced_matrix.shape[1]}")

        # Map reduced descriptors back to their image paths
        reduced_descriptors = {
            img_path: reduced_matrix[i, :] for i, img_path in enumerate(image_paths)
        }
        self.save_descriptors(reduced_descriptors, descriptor_pca_path)
        return reduced_descriptors

class Extractor:
    @staticmethod
    def extract_random(img) -> np.ndarray:
        return np.random.rand(1, 30)

    @staticmethod
    def extract_rgb(img) -> np.ndarray:
        img = img / 255.0  # normalize the image to [0, 1]
        B = np.mean(img[:, :, 0])
        G = np.mean(img[:, :, 1])
        R = np.mean(img[:, :, 2])
        return np.array([R, G, B])

    @staticmethod
    def extract_gridRGB(img, grid_size: int = 4) -> np.ndarray:
        img_height, img_width, img_channel = img.shape
        grid_height = img_height // grid_size
        grid_width = img_width // grid_size
        grid_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid_cell = img[
                    i * grid_height : (i + 1) * grid_height,
                    j * grid_width : (j + 1) * grid_width,
                    :,
                ]
                R = np.mean(grid_cell[:, :, 0])
                G = np.mean(grid_cell[:, :, 1])
                B = np.mean(grid_cell[:, :, 2])
                grid_features.extend([R, G, B])
        return np.array(grid_features)

    @staticmethod
    def extract_gridEOhisto(
        img, grid_size: int = 4, sobel_filter_size: int = 3, ang_quant_lvl=8
    ) -> np.ndarray:
        img_height, img_width, channel = img.shape
        grid_height = img_height // grid_size
        grid_width = img_width // grid_size
        grid_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid_cell = img[
                    i * grid_height : (i + 1) * grid_height,
                    j * grid_width : (j + 1) * grid_width,
                    :,
                ]
                # convert to uint8 for cv2
                grid_cell = grid_cell.astype(np.uint8)
                gray = cv2.cvtColor(grid_cell, cv2.COLOR_BGR2GRAY)
                sobelx = cv2.Sobel(
                    gray, cv2.CV_64F, dx=1, dy=0, ksize=sobel_filter_size
                )
                sobely = cv2.Sobel(
                    gray, cv2.CV_64F, dx=0, dy=1, ksize=sobel_filter_size
                )
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                orientation = (
                    np.arctan2(sobely, sobelx) * 180 / np.pi
                )  # convert to degrees
                # Normalize orientation to [0, 180] because direction of an angle is ambiguous up to 180.
                norm_orientation = np.mod(orientation + 180, 180)
                bin_width = 180 / ang_quant_lvl
                quantized_orientation = np.floor(norm_orientation / bin_width).astype(
                    int
                )
                hist = np.histogram(
                    quantized_orientation,
                    bins=ang_quant_lvl,
                    range=(0, ang_quant_lvl),
                    weights=magnitude,
                )[
                    0
                ]  # emphasize stronger edges
                hist_normalized = hist / (
                    np.sum(hist) + 1e-6
                )  # so we don't divide by zero
                grid_features.extend(hist_normalized)
        return np.array(grid_features)

    @staticmethod
    def min_max_normalize(features: np.ndarray) -> np.ndarray:
        min_val = np.min(features)
        max_val = np.max(features)
        if max_val - min_val == 0:
            return features
        return (features - min_val) / (max_val - min_val)

    @staticmethod
    def z_score_normalize(features: np.ndarray) -> np.ndarray:
        mean = np.mean(features)
        std = np.std(features)
        if std == 0:
            return features
        return (features - mean) / std

    # TODO: Try out z-score normalization for gridCombined
    @staticmethod
    def extract_grid_combined(
        img,
        grid_size: int = 4,
        sobel_filter_size: int = 3,
        ang_quant_lvl: int = 8,
        norm_method: str = "minmax",
    ) -> np.ndarray:
        # Reuse the individual functions for RGB and Edge Orientation
        rgb_features = Extractor.extract_gridRGB(img, grid_size)
        eohisto_features = Extractor.extract_gridEOhisto(
            img, grid_size, sobel_filter_size, ang_quant_lvl
        )
        match norm_method:
            case "minmax":
                rgb_features_normalized = Extractor.min_max_normalize(rgb_features)
                eohisto_features_normalized = Extractor.min_max_normalize(
                    eohisto_features
                )
            case "zscore":
                rgb_features_normalized = Extractor.z_score_normalize(rgb_features)
                eohisto_features_normalized = Extractor.z_score_normalize(
                    eohisto_features
                )
        # Concatenate the two feature vectors
        combined_features = np.concatenate(
            (rgb_features_normalized, eohisto_features_normalized)
        )
        return combined_features

    @staticmethod
    def extract_globalRGBhisto(img, bins=32) -> np.ndarray:
        hist = [
            np.histogram(img[:, :, i], bins=bins, range=(0, 256))[0] for i in range(3)
        ]
        hist_flat = np.concatenate(hist)
        hist_normalized = hist_flat / np.sum(hist_flat)
        return hist_normalized

    @staticmethod
    def extract_globalRGBhisto_quant(img, quant_lvl=4) -> np.ndarray:
        # Quantize the RGB values
        R = np.floor(img[:, :, 0] * quant_lvl / 256).astype(int)
        G = np.floor(img[:, :, 1] * quant_lvl / 256).astype(int)
        B = np.floor(img[:, :, 2] * quant_lvl / 256).astype(int)

        poly_repr = R * quant_lvl**2 + G * quant_lvl + B
        hist = np.histogram(
            poly_repr, bins=np.arange(quant_lvl**3 + 1), density=False
        )[0]
        hist_flat = hist.flatten()
        hist_normalized = hist_flat / np.sum(hist_flat)
        LOGGER.debug(f"Dimensions of the descriptor: {hist_normalized.shape}")
        return hist_normalized

    @staticmethod
    def extract_SIFT(img):
        fd = FeatureDetector("SIFT")
        return fd.detect_keypoints_compute_descriptors(img)[1]