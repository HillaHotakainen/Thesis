import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sea
import matplotlib.pyplot as matp

# modify this to decide what file you use
model_filename = 'trained_model.pkl'
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

# Creating a classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
matp.figure(figsize=(10, 6))
sea.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, fmt='.2f', cmap='Blues', linewidths=.5)
matp.title('Classification Report')
matp.show()

# Creating confusion matrix
cm = confusion_matrix(y_test, y_pred)
sea.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
matp.xlabel('Predicted')
matp.ylabel('Actual')
matp.title('Confusion Matrix')
matp.show()
