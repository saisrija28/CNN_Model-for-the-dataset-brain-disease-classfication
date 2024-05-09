import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class PreProcess_Data:
    def visualization_images(self, dir_path, nimages):
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in range(nimages):
                img = os.path.join(dpath, i, train_class[j])
                img = cv2.imread(img)
                axs[count][j].title.set_text(i)
                axs[count][j].imshow(img)
            count += 1
            fig.tight_layout()
        plt.show(block=True)

    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                label.append(i)
        print('Number of train images : {}\n'.format(len(train)))
        print('Number of train images labels: {}\n'.format(len(label)))
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        print(retina_df)
        return retina_df, train, label

    def generate_train_test_images(self, train, label):
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        print(retina_df)
        train_data, test_data = train_test_split(retina_df, test_size=0.2)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.15
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            subset='training'
        )

        validation_generator = train_datagen.flow_from_dataframe(
            dataframe=train_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            subset='validation'
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32
        )

        print(f"Train images shape: {train_data.shape}")
        print(f"Testing images shape: {test_data.shape}")

        return train_generator, test_generator, validation_generator