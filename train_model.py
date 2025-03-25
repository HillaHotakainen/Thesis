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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for emotion in os.listdir(folder):
        emotion_folder = os.path.join(folder, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # Detect faces
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]  # Crop the face from the image
                    face = cv2.resize(face, (300, 300))  # Resize the face
                    data['image'].append(face)
                    data['emotion'].append(emotion)
    df = pd.DataFrame(data)
    return df

# Load image data
folder = 'assets'
train_data = load_images_from_folder(folder)
print("Loaded image data")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data['image'], train_data['emotion'], test_size=0.2, random_state=42)

# Flatten the images for training
X_train_flat = [img.flatten() for img in X_train]
X_test_flat = [img.flatten() for img in X_test]

# Build a simple pipeline with a classifier
model = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight="balanced"))
])

# Train the model (using RandomForestClassifier)
model.fit(X_train_flat, y_train)

# Save the trained model
model_filename = 'trained_model.pkl'
joblib.dump(model, model_filename)
print(f"Trained model saved to {model_filename}")

test_data="test_data.npz"
np.savez(test_data, X_test=X_test_flat, y_test=y_test)
print("Test data saved as ")