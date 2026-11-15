from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
vectorizer = CountVectorizer(stop_words="english", max_features=10000, binary=False)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)
for name, model in [("Multinomial NB", MultinomialNB()),
                    ("Bernoulli NB", BernoulliNB())]:
    model.fit(X_train, train.target)
    pred = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(test.target, pred),
          "F1:", f1_score(test.target, pred, average="weighted"))

