import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_classifier_model.h5")

# Set image size (adjust based on what was used during training)
IMG_SIZE = 128  # Set your image size (same as used during training)

# Title of the app
st.title("Brain Tumor Classification")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for prediction
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    prediction_class = "Tumor" if prediction[0] > 0.5 else "No Tumor"

    # Display prediction result
    st.write(f"Prediction: {prediction_class}")
