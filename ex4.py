from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
# Large C approximates unregularized MLE; L2 and L1 are MAP with priors
models = {
    "MLE (approximately)": LogisticRegression(C=1e6, max_iter=5000),
    "MAP with L2 prior": LogisticRegression(C=1.0, penalty="l2", max_iter=5000),
    "MAP with L1 prior": LogisticRegression(C=1.0, penalty="l1", solver="liblinear", max_iter=5000)
}
for name, clf in models.items():
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train, y_train)
    print(name, "accuracy:", accuracy_score(y_test, model.predict(X_test)))
    print("First five coefficients:", model[-1].coef_[0][:5])

