import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_classifier_model.h5")

# Set image size (adjust based on what was used during training)
IMG_SIZE = 128  # Set your image size (same as used during training)

# Set up the page layout and style
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")
st.markdown("""
    <style>
        .main { 
            background-color: #F4F7FA;
            padding: 20px;
            border-radius: 15px;
        }
        .header { 
            text-align: center;
            color: #3c8dbc;
            font-size: 35px;
            font-weight: bold;
        }
        .prediction { 
            font-size: 18px;
            font-weight: bold;
        }
        .message { 
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app with enhanced look
st.markdown("<div class='header'>Brain Tumor Detection: Upload Your MRI Scan</div>", unsafe_allow_html=True)

# Upload image file with customized text
uploaded_file = st.file_uploader("Please upload your MRI scan (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

# Reset session state on button click
if st.button("Start New Prediction", key="new_prediction"):
    st.session_state["prediction"] = None
    st.session_state["uploaded_file"] = None

# If the user uploaded a file, process the image
if uploaded_file is not None or "uploaded_file" in st.session_state:
    st.session_state["uploaded_file"] = uploaded_file
    
    # Open and display the uploaded image
    img = image.load_img(st.session_state["uploaded_file"], target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
    st.image(img, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess the image for prediction
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)

    # Store prediction in session_state
    if prediction[0] > 0.5:
        prediction_class = "Tumor"
        message = (
            "⚠️ **Warning**: The model has detected a **possible tumor** in the MRI scan. "
            "It is **critical** to consult a **healthcare professional** immediately. "
            "We strongly recommend seeing an **oncologist** or **neurologist** as soon as possible for a full evaluation and **treatment**."
        )
        confidence = f"Confidence: {prediction[0][0] * 100:.2f}%"
        
    else:
        prediction_class = "No Tumor"
        message = "😊 Good news: No tumor detected in the MRI scan. However, please continue with regular checkups to stay healthy."
        confidence = f"Confidence: {(1 - prediction[0][0]) * 100:.2f}%"

    # Store the prediction in session_state
    st.session_state["prediction"] = (prediction_class, message, confidence)

    # Display results with proper markdown rendering
    st.markdown(f"### Prediction: **{st.session_state['prediction'][0]}**")
    st.markdown(f"#### {st.session_state['prediction'][1]}")
    st.markdown(f"**{st.session_state['prediction'][2]}**")
