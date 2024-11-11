import tensorflow as tf
from keras.api.applications import ResNet50
from keras.api.models import Model

from keras.api.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.api.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

class ResNet:
    def __init__(self, image_size=(224, 224), learning_rate=0.001):
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.num_classes = None
        self.model = None

    def build_model(self, num_classes):
        self.num_classes = num_classes
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Global average pooling
        x = Dropout(0.5)(x)              # Add dropout for regularization
        x = Dense(256, activation='relu')(x)  # Fully connected layer
        predictions = Dense(self.num_classes, activation='softmax')(x)  # Final classification layer
        self.model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def prepare_data(self, dataset_dir, batch_size=32, validation_split=0.2):
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=validation_split
        )

        train_generator = datagen.flow_from_directory(
            dataset_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_generator = datagen.flow_from_directory(
            dataset_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        self.num_classes = train_generator.num_classes  # Dynamically determine number of classes
        return train_generator, val_generator

    def train_model(self, train_generator, val_generator, epochs=10):
        if self.model is None:
            raise ValueError("Model is not built. Call 'build_model' first.")

        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            steps_per_epoch=len(train_generator),
            validation_steps=len(val_generator)
        )

        return history

    def save_model(self, save_path):
        if self.model is None:
            raise ValueError("Model is not built. Call 'build_model' first.")

        self.model.save(save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    dataset_dir = "resnet_dataset"  # Update if the path differs
    resnet = ResNet(image_size=(224, 224), learning_rate=0.001)
    train_gen, val_gen = resnet.prepare_data(dataset_dir, batch_size=32, validation_split=0.2)
    resnet.build_model(num_classes=resnet.num_classes)
    history = resnet.train_model(train_gen, val_gen, epochs=10)
    resnet.save_model("resnet50_trained_model.keras")