import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
for lr, batch, epochs in [(0.001, 64, 3), (0.01, 128, 3), (0.0005, 128, 5)]:
    model = Sequential([Flatten(input_shape=(28, 28)),
                        Dense(128, activation="relu"),
                        Dense(10, activation="softmax")])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print("Learning rate:", lr, "Batch:", batch, "Epochs:", epochs, "Accuracy:", acc)
