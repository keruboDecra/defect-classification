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
