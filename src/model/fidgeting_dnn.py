from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Input, Concatenate
from keras.backend import concatenate
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.optimizers import SGD
import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras import Model
import numpy as np

class Fidgeting_DNN():


    def __init__(self, input_dim, num_classes=2):
        self.input_dim = input_dim
        self.model = Sequential()
        self.num_classes = num_classes

    def build_multi_class_model(self):
        input_fft = self.input_dim[0]
        input_std = self.input_dim[1]
        input_mean = self.input_dim[2]

        inputA = Input(shape=(input_fft,))
        inputB = Input(shape=(input_std,))
        inputC = Input(shape=(input_mean,))

        x = Dense(64, activation="relu")(inputA)
        #x = Dropout(0.2)(x)
        x = Dense(4, activation="relu")(x)

        y = Dense(8, activation="relu")(inputB)
        y = Dense(1, activation="relu")(y)

        m = Dense(4, activation="relu")(inputC)
        m = Dense(1, activation="relu")(m)

        combined = Concatenate(axis=1)([x, y, m])

        z = Dense(self.num_classes, activation="softmax")(combined)
        # z = Dense(3, activation="softmax")(z)

        self.model = Model(inputs=[inputA, inputB, inputC], outputs=z)
        # self.model.add(Dense(64, activation='relu', input_dim= self.input_dim))
        # self.model.add(Dropout(0.1))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dropout(0.1))
        # self.model.add(Dense(self.num_classes, activation='softmax'))

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(self.model.summary())

    def slice_input(self, X):
        print(X.shape)
        x = 0
        result = []
        for i in self.input_dim:
            result.append(X[:, x:x+i])
            x += i
        return result

    def train_multi_class_model(self, X_train, y_train, X_dev, y_dev, class_weight={0: 1, 1: 1}):
        X_train = self.slice_input(X_train)
        X_dev = self.slice_input(X_dev)

        y_train = to_categorical(y_train, num_classes=self.num_classes, dtype='float32')
        y_dev = to_categorical(y_dev, num_classes=self.num_classes, dtype='float32')

        self.model.fit(X_train, y_train,
                       batch_size=64, epochs=300,
                       validation_data=(X_dev, y_dev),
                       class_weight=class_weight,
                       )

    def evaluate_multi_class(self, X, y):
        X = self.slice_input(X)

        y_pred = self.model.predict(X, batch_size=128)
        y = to_categorical(y, num_classes=self.num_classes, dtype='float32')
        y = np.argmax(y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        print(classification_report(y, y_pred, range(self.num_classes)))
        return classification_report(y, y_pred, range(self.num_classes), output_dict=True)

    def build_model(self):
        self.model = Sequential()

        self.model.add(Dense(units=64, activation='relu', input_dim=self.input_dim))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=16, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_dev, y_dev, class_weight={0:1, 1:1}):
        self.model.fit(X_train, y_train,
                  batch_size=64, epochs=1000,
                  validation_data=(X_dev, y_dev),
                       class_weight=class_weight,
                  )

    def evaluate(self, X, y):
        y_pred = self.model.predict(X, batch_size=128)

        # for i in range(X.shape[0]):
        #     print(X[i, :])
        #     print(y[i], y_pred[i])
        #     print('\n')

        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1

        print(classification_report(y, y_pred, [0, 1]))

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)