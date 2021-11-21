from Tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_test_and_train_data(train_data_path, test_data_path):
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    return train_data, test_data


def load_test_and_train_labels(train_labels_path, test_labels_path):
    train_labels = np.load(train_labels_path)
    test_labels = np.load(test_labels_path)
    return train_labels, test_labels


# CNN Classifier with two convolutional layers and one fully connected layer for 32x32 gray images
def get_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    X_train, X_test = load_test_and_train_data('../data/X_train.npy', '../data/X_test.npy')
    y_train, y_test = load_test_and_train_labels('../data/y_train.npy', '../data/y_test.npy')
    model = get_cnn_model()
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('../models/cnn_model.h5')
    print('Model saved')
    print('Done')
    # model = load_model('../models/cnn_model.h5')
    # print(model.summary())
    # print(model.evaluate(X_test, y_test))
    # print(model.predict(X_test[0:1]))
    # print(y_test[0:1])
    # print(np.argmax(model.predict(X_test[0:1]), axis=1))
    # print(np.argmax(y_test[0:1], axis=1))
    # print(model.predict_classes(X_test[0:1]))

