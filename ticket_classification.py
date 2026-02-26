import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
"text":[
"Payment failed",
"Refund not received",
"App crashed",
"Login issue"
],
"category":[
"Billing",
"Billing",
"Technical",
"Technical"
]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["category"]

model = MultinomialNB()
model.fit(X,y)

test = ["Payment issue"]
test_vector = vectorizer.transform(test)

prediction = model.predict(test_vector)

print(prediction)