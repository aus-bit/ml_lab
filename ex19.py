from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = fetch_openml("titanic", version=1, as_frame=True)
X = data.data[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
y = data.target.astype(int)
num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns
prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), cat)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
models = {
    "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(),
                                 n_estimators=50, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
}
for name, clf in models.items():
    model = make_pipeline(prep, clf)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, pred),
          "Precision:", precision_score(y_test, pred),
          "Recall:", recall_score(y_test, pred), "F1:", f1_score(y_test, pred))
