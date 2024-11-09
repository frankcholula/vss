import cv2
import numpy as np
from numpy.linalg import LinAlgError
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple
import logging
import streamlit as st
import os

LOGGER = logging.getLogger(__name__)

class Retriever:
    def __init__(self, img_desc_dict: Dict[str, np.ndarray], metric: str):
        self.img_desc_dict = img_desc_dict
        self.metric = metric
        self.cov_matrix_inv = None
        if metric == "Mahalanobis":
            self.cov_matrix_inv = self.calculate_covariance_matrix_inverse()
    
    def calculate_covariance_matrix_inverse(self) -> np.array:
        all_descriptors = np.array(list(self.img_desc_dict.values()))
        try:
            cov_matrix = np.cov(all_descriptors.T)
            cov_matrix_inv = np.linalg.inv(cov_matrix)
        except LinAlgError as e:
            LOGGER.error(f"Error in calculating the inverse covariance matrix: {str(e)}")
            cov_matrix_inv = None
        return cov_matrix_inv

    @staticmethod
    def cvpr_compare(F1, F2, metric, cov_matrix_inv=None) -> float:
        # This function should compare F1 to F2 - i.e. compute the distance
        # between the two descriptors
        if F1.shape != F2.shape:
            raise ValueError(
                f"The two feature vectors must have the same shape. \nF1 shape: {F1.shape} \nF2 shape: {F2.shape}"
            )
        # TODO: Add new metrics here
        match metric:
            case "L2":
                dst = np.linalg.norm(F1 - F2)
            case "L1":
                dst = np.linalg.norm(F1 - F2, ord=1)
            case "Mahalanobis":
                try:
                    # sqrt((F1 - F2)^T * cov_matrix_inv * (F1 - F2))
                    diff = F1 - F2
                    dst = np.sqrt(np.dot(diff.T, np.dot(cov_matrix_inv, diff)))
                except Exception as e:
                    LOGGER.error(f"Error in calculating Mahalanobis distance: {str(e)}")
                    dst = float("inf")

                
        return dst

    def compute_distance(self, query_img: str) -> List[Tuple[float, str]]:
        # Compute the distance between the query and all other descriptors
        dst = []
        query_img_desc = self.img_desc_dict[query_img]

        for img_path, candidate_desc in self.img_desc_dict.items():
            if img_path != query_img:  # Skip the query image itself
                distance = Retriever.cvpr_compare(
                    query_img_desc, candidate_desc, self.metric, self.cov_matrix_inv
                )
                dst.append((distance, img_path))

        dst.sort(key=lambda x: x[0])
        return dst

    def get_image_class(self, img_path: str) -> str:
        imagename = img_path.split("/")[2]
        return imagename.split("_")[0]

    def retrieve(self, query_img: str, total_relevant_images: int) -> list:
        distances = self.compute_distance(query_img)
        target_class = self.get_image_class(query_img)
        seen_count = 0
        find_all_images_at = 0
        for distance, img_path in distances:
            seen_class = self.get_image_class(img_path)
            find_all_images_at +=1
            if seen_class == target_class:
                seen_count += 1
            if seen_count >= total_relevant_images:
                break
        print(f"Found all images after {find_all_images_at} images.")
        # edge case: if the Mahalanobis distance is infinite for some images
        # if float("inf") in [distance for distance, _ in top_similar_images]:
        #     return []
        return [img_path for _, img_path in distances], find_all_images_at

    def display_images(self, vse, similar_images: list, result_num: int):
        images_to_display = similar_images[:result_num]
        for i in range(0, len(images_to_display), 5):
            cols = vse.columns(5)
            for col, img_path in zip(cols, images_to_display[i : i + 5]):
                col.image(
                    img_path, use_column_width=True, caption=os.path.basename(img_path)
                )
                if st.session_state["debug_mode"]:
                    col.write(f"Class: {self.get_image_class(os.path.basename(img_path))}")
                    col.write(self.get_image_class(os.path.basename(img_path)))
            pass
        return images_to_display
    #     fig, axes = plt.subplots(1, number + 1, figsize=(20, 5))
    #     distances = []
    #     # Display the query image
    #     query_img_data = cv2.imread(query_img)
    #     query_img_data = cv2.cvtColor(query_img_data, cv2.COLOR_BGR2RGB)
    #     axes[0].imshow(query_img_data)
    #     axes[0].set_title("Query Image")
    #     axes[0].axis("off")

    #     # Display the top similar images
    #     for ax, (distance, img_path) in zip(axes[1:], top_similar_images):
    #         img = cv2.imread(img_path)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         ax.imshow(img)
    #         ax.axis("off")
    #         distances.append(distance)
    #     if float("inf") in distances:
    #         LOGGER.error(f"Mahalanobis distance is infinite for some images.")
    #         st.error(f"Error in calculating {self.metric} distances for some images. Please try another metric or use PCA to lower the dimensionality.", icon="🚨")
    #     LOGGER.info(f"{self.metric} Distances: {distances} \n ")
