import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import keras
from keras.models import load_model

# Load the .h5 model
model = load_model('brain_tumor_classifier_model.h5')



# Set up the page
st.set_page_config(page_title="Brain Tumor Detection App", layout="centered")
st.title("🧠 Brain Tumor Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
    st.success("File uploaded successfully!")

    # Preprocess image
    img = image.resize((224, 224))  # Adjust size based on model input
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Interpret prediction
    class_names = ['No Tumor', 'Tumor']
    predicted_index = int(np.round(prediction[0][0]))  # Binary classifier assumption

    # Ensure safety
    if predicted_index >= len(class_names):
        st.error("Unexpected prediction output.")
    else:
        predicted_class = class_names[predicted_index]
        st.subheader(f"🧾 Prediction: **{predicted_class}**")
