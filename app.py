import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the ONNX model
onnx_model_path = 'models/deep_cnn_model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Class labels for the predicted classes
class_labels = ['car', 'dog', 'person']

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to 224x224
    image = image.resize((224, 224))
    
    # Convert image to numpy array and normalize it
    image_data = np.array(image).astype(np.float32)
    image_data = image_data.transpose(2, 0, 1)  # Change (H, W, C) to (C, H, W)

    # Normalize as per your model's normalization: [-1, 1] range
    mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1).astype(np.float32)
    std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1).astype(np.float32)
    image_data = (image_data / 255.0 - mean) / std
    
    # Add a batch dimension (N, C, H, W)
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    
    return image_data

# Streamlit UI
st.title("Image Classification App")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    # Perform inference using ONNX model
    input_name = 'input.1'  # Your model's input name
    output_name = '24'      # Your model's output name
    outputs = ort_session.run([output_name], {input_name: input_tensor})

    # Get the predicted class
    predicted_class_idx = np.argmax(outputs[0])
    predicted_class_name = class_labels[predicted_class_idx]

    # Display the prediction
    st.write(f"Predicted class: **{predicted_class_name}**")
