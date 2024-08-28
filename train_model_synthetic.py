from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import numpy as np

# Function to generate synthetic dataset
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    emotions = ['happy', 'sad', 'angry', 'neutral']
    data = {'image': [], 'emotion': []}

    for _ in range(num_samples):
        # Generate random synthetic image data (replace this with your actual image data loading)
        image_data = np.random.rand(100, 100, 3) * 255  # Example random image
        emotion = np.random.choice(emotions)
        data['image'].append(image_data)
        data['emotion'].append(emotion)

    df = pd.DataFrame(data)
    return df

# Generate synthetic training data
train_data = generate_synthetic_data()
print("generating synthetic data")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data['image'], train_data['emotion'], test_size=0.2, random_state=42)

# Build a simple pipeline with a classifier
model = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100))  # Use RandomForestClassifier for image data
])

# Flatten the images for training
X_train_flat = [img.flatten() for img in X_train]

# Train the model
model.fit(X_train_flat, y_train)

# Save the trained model
model_filename = 'emotion_detection_model.pkl'
joblib.dump(model, model_filename)
print(f"Trained model saved to {model_filename}")