import cv2
import cvlib as cv
import joblib

model_filename = 'trained_model.pkl'

# Load the trained model
print("staring model load")
loaded_model = joblib.load(model_filename)
print("model load done")

# Start capturing video from the default camera 
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces, confidences = cv.detect_face(frame)
    

    # Loop through detected faces
    for face, confidence in zip(faces, confidences):
        (start_x, start_y, end_x, end_y) = face

        # Crop the face from the frame
        face_crop = gray_frame[start_y:end_y, start_x:end_x]

        # Resize the face 
        face_resize = cv2.resize(face_crop, (100, 100))

        # Flatten the face image 
        face_flat = face_resize.flatten()

        # Perform emotion prediction
        emotion = loaded_model.predict([face_flat])[0]

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