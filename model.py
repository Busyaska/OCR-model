import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import image_dataset_from_directory
from Indexes import indexes


class OpticalCharacterRecognition:

    def __init__(self):
        self.__model = keras.Sequential()
        self.__model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.__model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.__model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.__model.add(Flatten())
        self.__model.add(Dense(128, activation='relu'))
        self.__model.add(Dropout(0.3))
        self.__model.add(Dense(256, activation='relu'))
        self.__model.add(Dense(69, activation='softmax'))
        self.__model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.__model_training_history = None

    def process_dataset(self, dataset_path: str, ignore_autotune: bool = False) -> (tensorflow.data.Dataset, tensorflow.data.Dataset):
        train_data, val_data = image_dataset_from_directory(dataset_path,
                                                            label_mode='int',
                                                            class_names=[str(x) for x in range(0, 69)],
                                                            color_mode='grayscale',
                                                            batch_size=32,
                                                            image_size=(28, 28),
                                                            subset='both',
                                                            validation_split=0.25,
                                                            seed=19)
        if not ignore_autotune:
            AUTOTUNE = tensorflow.data.AUTOTUNE
            train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
            val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

        return train_data, val_data

    def train_model(self, train_data: tensorflow.data.Dataset, val_data: tensorflow.data.Dataset, epochs: int = 10):
        self.__model_training_history = self.__model.fit(train_data, validation_data=val_data, epochs=epochs)

    def show_training_history(self):
        if self.__model_training_history:
            plt.plot(self.__model_training_history.history['loss'])
            plt.plot(self.__model_training_history.history['val_loss'])
            plt.grid(True)
            plt.show()

    def check_accuracy(self, data) -> [float, float]:
        return self.__model.evaluate(data)

    def save_model(self, file_name: str, file_path: str = ''):
        model_save_path = f'{file_path}{file_name}.h5'
        self.__model.save(model_save_path)

    def save_model_weights(self, file_name: str, file_path: str = ''):
        model_weights_save_path = f'{file_path}{file_name}.h5'
        self.__model.save_weights(model_weights_save_path)

    def predict_data(self, image_path: str) -> str:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_data = np.reshape(img, (1, 28, 28, 1))
        prediction = self.__model.predict(image_data)
        highest_index = np.argmax(prediction)
        character = indexes[highest_index]
        return character
    