import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, :2]                 # sepal length and width
y = (iris.target == 0).astype(int)   # Setosa = 1, others = 0
model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1))
model.fit(X, y)
print("Accuracy:", accuracy_score(y, model.predict(X)))
DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict",
                                       alpha=0.4, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
plt.xlabel("Sepal length"); plt.ylabel("Sepal width"); plt.title("Linear SVM")
plt.show()
