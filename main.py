import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import preprocess as dp
import classification  as cm

if __name__ == "__main__":
    images_folder_path = 'test'
    imdata = dp.PreProcess_Data()
    retina_df, train, label = imdata.preprocess(images_folder_path)
    imdata.visualization_images(images_folder_path, 4)
    tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(train, label)

    Annmodel = cm.DeepANN()
    m1 = Annmodel.cnn_model()
    Ann_history = m1.fit(tr_gen, epochs=5, validation_data=va_gen)

    # Plotting training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(Ann_history.history['loss'], label='Training Loss')
    plt.plot(Ann_history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(Ann_history.history['accuracy'], label='Training Accuracy')
    plt.plot(Ann_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    Ann_test_loss, Ann_test_acc = m1.evaluate(tr_gen)
    print(f'Test Accuracy: {Ann_test_acc}')
    m1.save('CNNModel.keras')
    print(m1.summary())