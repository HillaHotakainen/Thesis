import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# Function to load images from folders
def load_images_from_folder(folder):
    data = {'image': [], 'emotion': []}
    for emotion in os.listdir(folder):
        emotion_folder = os.path.join(folder, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                data['image'].append(img)
                data['emotion'].append(emotion)
    df = pd.DataFrame(data)
    return df

# Load image data
folder = 'assets'
train_data = load_images_from_folder(folder)
print("Loaded image data")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data['image'], train_data['emotion'], test_size=0.2, random_state=42)

# Build a simple pipeline with a classifier
model = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Flatten the images for training
X_train_flat = [img.flatten() for img in X_train]
X_test_flat = [img.flatten() for img in X_test]

# Train the model (using RandomForestClassifier)
model.fit(X_train_flat, y_train)

# Save the trained model
model_filename = 'trained_model.pkl'
joblib.dump(model, model_filename)
print(f"Trained model saved to {model_filename}")

real_data="real_data.npz"
np.savez(real_data, X_test=X_test_flat, y_test=y_test)
print("Test data saved as {test_data}")