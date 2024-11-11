import logging
import numpy as np
from keras.api.applications import ResNet50
from keras.api.preprocessing.image import load_img, img_to_array

logging.basicConfig(level=logging.DEBUG)


class ResNet:
    def __init__(self, image_size=(224, 224), learning_rate=0.001):
        self.image_size = image_size
        self.learning_rate = learning_rate
    
    def use_pretrained_feature_extractor(self):
        feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
        return feature_extractor

    def preprocess_image(self, img_path: str, target_size=(224, 224)):
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def generate_single_feature(self, img_path: str, feature_extractor):
        img_array = self.preprocess_image(img_path)
        feature_vector = feature_extractor.predict(img_array)
        return feature_vector.flatten()
