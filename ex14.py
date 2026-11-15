import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
for activation in ["sigmoid", "relu", "tanh"]:
    model = Sequential([Flatten(input_shape=(28, 28)),
                        Dense(128, activation=activation),
                        Dense(10, activation="softmax")])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    start = time.time()
    model.fit(X_train, y_train, epochs=3, batch_size=128, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(activation, "accuracy:", acc, "training time:", round(time.time()-start, 2), "seconds")
