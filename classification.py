from keras import *
from keras.layers import *


class DeepANN:
    def cnn_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 3)))
        model.add(MaxPooling2D((2, 2)))
        # model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(4, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model