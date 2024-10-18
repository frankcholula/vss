import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple

class Retriever:
    def __init__(self, img_desc_dict: Dict[str, np.ndarray], metric: str):
        self.img_desc_dict = img_desc_dict
        self.metric = metric

    @staticmethod
    def cvpr_compare(F1, F2, metric) -> float:
        # This function should compare F1 to F2 - i.e. compute the distance
        # between the two descriptors
        match metric:
            case "l2":
                dst = np.linalg.norm(F1 - F2)
            case "l1":
                dst = np.linalg.norm(F1 - F2, ord=1)
        return dst

    def compute_distance(self, query_img: str) -> List[Tuple[float, str]]:
        # Compute the distance between the query and all other descriptors
        dst = []
        query_img_desc = self.img_desc_dict[query_img]
        
        for img_path, candidate_desc in self.img_desc_dict.items():
            if img_path != query_img:  # Skip the query image itself
                distance = Retriever.cvpr_compare(query_img_desc, candidate_desc, self.metric)
                dst.append((distance, img_path))
        
        dst.sort(key=lambda x: x[0])
        return dst

    def retrieve(self, query_img: str, number: int = 10) -> list:
        # Compute distances
        distances = self.compute_distance(query_img)
        top_similar_images = distances[:number]
        self.display_images(query_img, top_similar_images, number)
        return [img_path for _, img_path in top_similar_images]

    def display_images(self, query_img: str, top_similar_images: list, number: int):
        fig, axes = plt.subplots(1, number + 1, figsize=(20, 5))
        distances = []
        # Display the query image
        query_img_data = cv2.imread(query_img)
        query_img_data = cv2.cvtColor(query_img_data, cv2.COLOR_BGR2RGB)
        axes[0].imshow(query_img_data)
        axes[0].set_title('Query Image')
        axes[0].axis('off')
        
        # Display the top similar images
        for ax, (distance, img_path) in zip(axes[1:], top_similar_images):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis('off')
            distances.append(distance)
        print(f"{self.metric} Distances: {distances} \n ")
