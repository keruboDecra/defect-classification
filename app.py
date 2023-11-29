
pip install tensorflow

import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.applications import MobileNetV2

# Rest of your code remains the same...


# Load the pre-trained MobileNetV2 model for image preprocessing
pretrained_model = MobileNetV2(weights='imagenet', include_top=True)

# Load the trained model for defect detection
defect_model = defect_model.h5  # Replace 'model' with the actual name of your model

def classify_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features using MobileNetV2
    features = feature_model.predict(img_array)

    # Predict defect using the trained model
    prediction = defect_model.predict(features)

    return prediction[0][0]

# Streamlit UI
st.title("Defect Detection Dashboard")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

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

# Save this file and confirm when you are ready to deploy.
