
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Create a sample dataset
data = {
    "message": [
        "Win a free iPhone now",
        "Congratulations you won a lottery",
        "Claim your free prize today",
        "Click here to get free money",
        "You have won a cash reward",
        "Limited offer buy now",
        "Get a free gift card",
        "Urgent claim your reward",
        "Win cash by clicking this link",
        "You are the lucky winner",
        "Hey, how are you?",
        "Are we meeting today?",
        "Please call me when you are free",
        "I will reach home soon",
        "Can you send me the notes?",
        "Let's have lunch together",
        "Your class starts at 10 AM",
        "Thank you for your help",
        "Please bring my notebook",
        "I will call you tomorrow"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

# Step 2: Separate messages and labels
X = df["message"]
y = df["label"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 4: Convert text into numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 5: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vectorized)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Predict new messages
while True:
    message = input("\nEnter a message (or type 'exit' to stop): ")

    if message.lower() == "exit":
        break

    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)[0]

    if prediction == "spam":
        print("Result: SPAM MESSAGE")
    else:
        print("Result: NOT SPAM")
