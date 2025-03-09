import joblib
from sklearn.tree import DecisionTreeClassifier

# Example training data
# Features: [fever, cough, fatigue, headache, shortness of breath]
X_train = [
    [1, 1, 1, 0, 0],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
]

# Example labels
# 0: Healthy, 1: Flu, 2: Common Cold, 3: COVID-19, 4: Allergies
y_train = [1, 3, 1, 4, 2, 0]

# Train a simple Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'medical_diagnosis_model.pkl')
print("Model saved as 'medical_diagnosis_model.pkl'")