import logging
import numpy as np
from keras.api.applications import ResNet50
from keras.api.preprocessing.image import load_img, img_to_array
import torchvision.models as models
import torch
import torch.nn as nn
import cv2
from torchvision.transforms.functional import normalize, to_tensor

logging.basicConfig(level=logging.DEBUG)


class ResNet:
    def __init__(self, model_name: str):
        self.feature_extractor = None
        self.model_name = model_name

    def use_pretrained_feature_extractor(self):
        if self.model_name == "resnet50":
            self.feature_extractor = ResNet50(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(224, 224, 3),
            )
        elif self.model_name == "resnet34":
            resnet34 = models.resnet34(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet34.children())[:-1])
            self.feature_extractor.eval()

    def preprocess_image(self, img_path: str):
        if self.model_name == "resnet50":
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array / 255.0
        elif self.model_name == "resnet34":
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_tensor = to_tensor(img)
            img_tensor = normalize(
                img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            return img_tensor.unsqueeze(0)

    def generate_single_feature(self, img_path: str):
        if not self.feature_extractor:
            raise ValueError(
                "Feature extractor not initialized. Call 'use_pretrained_feature_extractor' first."
            )

        preprocessed_image = self.preprocess_image(img_path)

        if self.model_name == "resnet50":
            feature_vector = self.feature_extractor.predict(preprocessed_image)
        elif self.model_name == "resnet34":
            with torch.no_grad():
                feature_vector = self.feature_extractor(preprocessed_image).squeeze()
                feature_vector = feature_vector.numpy()
        return feature_vector.flatten()
