import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")
X = df.drop("quality", axis=1)
y = (df["quality"] >= 6).astype(int)  # good quality / other
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
for layers in [(50,), (50, 30), (100, 50)]:
    model = make_pipeline(StandardScaler(),
        MLPClassifier(hidden_layer_sizes=layers, max_iter=500, random_state=42))
    model.fit(X_train, y_train)
    print("Hidden layers", layers, "Accuracy:", accuracy_score(y_test, model.predict(X_test)))
