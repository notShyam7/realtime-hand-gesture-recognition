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
image_placeholder = st.image([], use_column_width=True, channels='BGR', caption='Live Feed')

# OpenCV VideoCapture
cap = cv2.VideoCapture(0)

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

        # Display the recognized gesture on the Streamlit app
        st.subheader(f'Recognized Gesture: {predicted_gesture}')

        # Display the webcam feed in the Streamlit app
        image_placeholder.image(frame, use_column_width=True, channels='BGR', caption='Live Feed')

    except KeyboardInterrupt:
        break

# Release the webcam
cap.release()
