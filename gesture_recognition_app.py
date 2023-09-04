import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# Load your trained model
model = tf.keras.models.load_model('gestures.h5')  # Update with the path to your model file

# Define a function to preprocess the frame for model input
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Set up Streamlit
st.title('Hand Gesture Recognition')

# Streamlit UI
st.header('Live Webcam Feed')

# Change it to something like this:
image_placeholder = st.empty()

# OpenCV VideoCapture
cap = cv2.VideoCapture(1)


# Inside your while loop, update the image placeholder with the live feed
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            continue

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Make predictions
        predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))

        # Find the predicted gesture
        predicted_gesture = np.argmax(predictions)

        # Display the recognized gesture as a label
        gesture_labels = ["Gesture 0: Fist", "Gesture 1: Open Hand", "Gesture 2: Thumbs Up", ...]
        cv2.putText(frame, gesture_labels[predicted_gesture], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the gesture image if available
        gesture_images = [cv2.imread("gesture0.png"), cv2.imread("gesture1.png"), ...]
        gesture_image = gesture_images[predicted_gesture]
        frame[10:10+gesture_image.shape[0], 200:200+gesture_image.shape[1]] = gesture_image

        # Display the frame in a window
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break the loop when the 'q' key is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except KeyboardInterrupt:
        break


# Release the webcam
cap.release()
