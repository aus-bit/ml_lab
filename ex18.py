import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier(random_state=42)
# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("5-fold CV accuracies:", scores)
print("Mean CV accuracy:", scores.mean())

# Bootstrap: sample rows with replacement, test on rows not selected
rng = np.random.default_rng(42)
accuracies, f1s = [], []
for _ in range(100):
    train_idx = rng.choice(len(X), size=len(X), replace=True)
    test_idx = np.setdiff1d(np.arange(len(X)), np.unique(train_idx))
    if len(test_idx) == 0:
        continue
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    accuracies.append(accuracy_score(y[test_idx], pred))
    f1s.append(f1_score(y[test_idx], pred, average="weighted"))
print("Bootstrap mean accuracy:", np.mean(accuracies))
print("Bootstrap mean F1:", np.mean(f1s))
