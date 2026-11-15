import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X = StandardScaler().fit_transform(X)
for name, model in [
    ("K-Means", KMeans(n_clusters=5, random_state=42, n_init=10)),
    ("Agglomerative", AgglomerativeClustering(n_clusters=5))
]:
    labels = model.fit_predict(X)
    print(name, "Silhouette score:", silhouette_score(X, labels))
    if name == "K-Means":
        print("Inertia:", model.inertia_)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(name); plt.xlabel("Annual income (scaled)"); plt.ylabel("Spending score (scaled)")
    plt.show()
