from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

# MLE: alpha=0 conceptually; smoothing avoids zero probabilities, so use tiny alpha.
# MAP: alpha acts like a symmetric Dirichlet prior.
for alpha, label in [(1e-9, "MLE (approx.)"), (0.1, "MAP weak prior"), (1.0, "MAP stronger prior")]:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, train.target)
    print(label, "accuracy:", accuracy_score(test.target, model.predict(X_test)))
