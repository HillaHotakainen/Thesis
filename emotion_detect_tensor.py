import cv2
import cvlib as cv
import numpy as np
import tensorflow as tf

model_filename = 'trained_model.keras'

emotion_names = ['angry', 'happy', 'neutral', 'sad']

# Load the trained Keras model
print("Starting model load")
model = tf.keras.models.load_model(model_filename)
print("Model load done")

# Start capturing video from the default camera 
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Detect faces in the frame
    faces, confidences = cv.detect_face(frame)
    
    # Loop through detected faces
    for face, confidence in zip(faces, confidences):
        (start_x, start_y, end_x, end_y) = face

        # Crop the face from the frame
        face_crop = frame[start_y:end_y, start_x:end_x]

        # Resize the face
        face_resize = cv2.resize(face_crop, (128, 128))  # Match input size for the model

        # Normalize the image (assuming the model was trained with normalized inputs)
        face_resize = face_resize / 255.0  # Scale pixel values to [0, 1]

        # Expand dimensions to match the model input shape
        face_resize = np.expand_dims(face_resize, axis=0)  # Add batch dimension

        # Perform emotion prediction
        predictions = model.predict(face_resize)
        emotion_index = np.argmax(predictions[0])  # Get the index of the highest probability
        emotion = emotion_names[emotion_index]

        # Draw bounding box and label on the frame
        label = f'Emotion: {emotion}'
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()