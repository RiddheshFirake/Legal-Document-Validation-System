import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dummy dataset with spam and non-spam messages
emails = [
    "Win a $1000 gift card now!", 
    "Get cheap loans easily", 
    "Hello, I wanted to follow up on our meeting", 
    "Congratulations, you have won a free trip", 
    "Please find attached the report for the last quarter",
    "Get your free credit score today",
    "Let's catch up soon, it’s been a while!"
]
labels = [1, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert text to numerical data
count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(emails)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the model
spam_detector = MultinomialNB()
spam_detector.fit(X_train, y_train)

# Test the model
predictions = spam_detector.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Save the vectorizer and model to files
with open('count_vectorizer.pickle', 'wb') as f:
    pickle.dump(count_vectorizer, f)
with open('spam_detector_model.pickle', 'wb') as f:
    pickle.dump(spam_detector, f)

# Example use of the model
def check_email(email_content):
    # Load the vectorizer and model
    count_vectorizer = pickle.load(open('count_vectorizer.pickle', 'rb'))
    spam_detector = pickle.load(open('spam_detector_model.pickle', 'rb'))
    
    # Transform the email content
    email_vector = count_vectorizer.transform([email_content])
    
    # Predict
    prediction = spam_detector.predict(email_vector)
    if prediction == 1:
        print("It's spam mail. Be careful.")
    else:
        print("It's not spam mail :)")

# Test with a new email
check_email("Congratulations! You have won a $500 voucher.")
