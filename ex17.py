import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, y = load_digits(return_X_y=True)
for k in [8, 10, 12]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    print("K =", k, "Inertia:", model.inertia_,
          "Silhouette:", silhouette_score(X, labels))
plt.imshow(X[0].reshape(8, 8), cmap="gray")
plt.title("Sample digit"); plt.axis("off"); plt.show()
