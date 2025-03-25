import tensorflow as tf
import joblib
import numpy as np

# Directories for dataset
dataset_dir = 'assets'  # The folder containing emotion subfolders like happy, sad, etc.
image_size = (128, 128)  # Image size (height, width)
batch_size = 32

# Load the data using image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

# Prefetching for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# CNN Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(image_size[0], image_size[1], 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20
)

# Save the Keras model
model.save('trained_model.keras')

# Convert to a format compatible with scikit-learn for joblib
class KerasClassifierWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

# Wrap and save the model
wrapped_model = KerasClassifierWrapper(model)
joblib.dump(wrapped_model, 'trained_model.pkl')

# Save the test data for later evaluation
def convert_dataset_to_numpy(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

# Create a smaller test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.1,
    subset='validation',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

# Convert and save test data
X_test, y_test = convert_dataset_to_numpy(test_dataset)
np.savez('real_data.npz', X_test=X_test, y_test=y_test)
