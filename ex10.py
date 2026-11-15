from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = fetch_openml("adult", version=2, as_frame=True)
X, y = data.data, data.target
y = (y.astype(str).str.strip() == ">50K").astype(int)
num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), cat)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
for name, clf in [("Logistic Regression", LogisticRegression(max_iter=2000)),
                  ("Decision Tree", DecisionTreeClassifier(max_depth=8, random_state=42))]:
    model = make_pipeline(prep, clf)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n", name, "Accuracy:", accuracy_score(y_test, pred),
          "Precision:", precision_score(y_test, pred),
          "Recall:", recall_score(y_test, pred), "F1:", f1_score(y_test, pred))
