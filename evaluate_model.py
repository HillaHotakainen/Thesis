import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sea
import matplotlib.pyplot as matp

# modify this to decide what file you use
model_filename = 'emotion_detection_model.pkl'
model = joblib.load(model_filename)
print(f"Loaded model from {model_filename}")

# Load the test data
data = np.load('real_data.npz', allow_pickle=True)
X_test = data['X_test']
y_test = data['y_test']
print("Loaded test data")

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Creating confusion matrix
cm = confusion_matrix(y_test, y_pred)
sea.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
matp.xlabel('Predicted')
matp.ylabel('Actual')
matp.title('Confusion Matrix')
matp.show()