# app.py

import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing import image


def classify_image(img_path):
    try:
        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((img_width, img_height))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features using MobileNetV2
        features = feature_model.predict(img_array)

        # Predict defect using the trained model
        prediction = defect_model.predict(features)

        print(f"Prediction Value: {prediction[0][0]}")  # Add this line

        return prediction[0][0]
    except Exception as e:
        print(f"Error during classification: {e}")
        return None



# Streamlit UI
st.title("Defect Detection Dashboard")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Classify the uploaded image
    prediction = classify_image(uploaded_file)

    # Display the result
    if prediction > 0.5:
        st.write("Defect Detected!")
    else:
        st.write("No Defect Detected.")
