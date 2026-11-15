from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                          {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}, cv=5),
    "Lasso": GridSearchCV(make_pipeline(StandardScaler(), Lasso(max_iter=10000)),
                          {"lasso__alpha": [0.001, 0.01, 0.1, 1, 10]}, cv=5)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MSE:", mean_squared_error(y_test, pred), "R2:", r2_score(y_test, pred))
    if hasattr(model, "best_params_"):
        print("Best parameters:", model.best_params_)
