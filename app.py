import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.applications import MobileNetV2

# Load the pre-trained MobileNetV2 model for image preprocessing
pretrained_model = MobileNetV2(weights='imagenet', include_top=True)

# Load the trained model for defect detection
defect_model_path = 'defect_model.h5'  # Assuming your model is in the root directory
defect_model = load_model(defect_model_path)

# Define image dimensions
img_width, img_height = 150, 150

# Define the feature extraction model
feature_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

def classify_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features using MobileNetV2
    features = feature_model.predict(img_array)

    # Print information for debugging
    print("Shape of features:", features.shape)
    
    # Initialize prediction variable
    prediction = None

    try:
        # Predict defect using the trained model
        prediction = defect_model.predict(features)
        print("Prediction shape:", prediction.shape)
    except Exception as e:
        print("Error during prediction:", str(e))

    # Ensure the function returns a value
    if prediction is not None:
        return prediction[0][0]
    else:
        return None


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

