import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "text": [
        "Payment failed",
        "Refund not received",
        "App crashed",
        "Unable to login",
        "Password reset needed",
        "Billing issue"
    ],
    "category": [
        "Billing",
        "Billing",
        "Technical",
        "Technical",
        "Technical",
        "Billing"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["category"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Test ticket
test_ticket = ["Payment problem"]

# Convert test ticket
test_vector = vectorizer.transform(test_ticket)

# Predict category
prediction = model.predict(test_vector)

# Output
print("Ticket:", test_ticket[0])
print("Predicted Category:", prediction[0])