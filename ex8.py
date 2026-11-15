
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Use smaller dataset
X_train = X_train[:5000].reshape(5000, -1) / 255.0
y_train = y_train[:5000]

X_test = X_test[:1000].reshape(1000, -1) / 255.0
y_test = y_test[:1000]

# KNN with different K values
for k in [3, 5, 7]:
    model = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("K =", k, "Accuracy:", accuracy_score(y_test, y_pred))
