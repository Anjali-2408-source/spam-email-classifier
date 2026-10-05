 # Spam Email Filter using Machine Learning

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("===== Spam Email Filter =====")

# Training data
emails = [
    "Congratulations! You won a free lottery prize",
    "Win money now by clicking this link",
    "Claim your free gift today",
    "You have won a cash reward",
    "Get free coupons and special offers",
    "Meeting is scheduled for tomorrow at 10 AM",
    "Please submit your assignment before Friday",
    "Your project report has been received",
    "Can we discuss the project today?",
    "Please attend the class tomorrow"
]

# Labels
labels = [
    "Spam",
    "Spam",
    "Spam",
    "Spam",
    "Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam"
]

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train Machine Learning model
model = MultinomialNB()
model.fit(X, labels)

# User input
email = input("\nEnter an email/message to check: ")

# Convert user message into features
email_vector = vectorizer.transform([email])

# Prediction
prediction = model.predict(email_vector)

print("\n===== RESULT =====")

if prediction[0] == "Spam":
    print("This message is likely SPAM.")
else:
    print(" This message is likely NOT SPAM.")
