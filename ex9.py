import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_excel("Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
# Create customer-level features and a simple target: high-value vs other customers
customer = df.groupby("CustomerID").agg(
    TotalSpend=("UnitPrice", lambda s: (s * df.loc[s.index, "Quantity"]).sum()),
    TotalQuantity=("Quantity", "sum"),
    Transactions=("InvoiceNo", "nunique")
).reset_index()
customer["Segment"] = (customer["TotalSpend"] >= customer["TotalSpend"].median()).astype(int)
X = customer[["TotalSpend", "TotalQuantity", "Transactions"]]
y = customer["Segment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# entropy criterion gives an ID3-style tree
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print("Feature importance:", dict(zip(X.columns, model.feature_importances_)))
plot_tree(model, feature_names=X.columns, class_names=["Other", "High value"], filled=True)
plt.show()
