
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('/Users/ghanshyam/Desktop/handgesture/realtime-hand-gesture-recognition/gestures.h5')
# Define a function to preprocess the frame for model input
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Set up Streamlit
st.title('Hand Gesture Recognition')

# OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Streamlit UI
st.header('Live Webcam Feed')
st.image([], use_column_width=True, channels='BGR', caption='Live Feed')

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
        st.image(frame, use_column_width=True, channels='BGR', caption='Live Feed')

    except KeyboardInterrupt:
        break

# Release the webcam
cap.release()
