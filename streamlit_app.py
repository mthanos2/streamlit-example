import streamlit as st
import glob
import os
import numpy as np
import cv2
from PIL import Image
import pydicom
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

 # Mapping of class labels to names
labels_to_names = {
    11: 'Malignant'
    }

# Function to convert DICOM to RGB image
def dicom_to_rgb(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array

    # Normalize pixel values to 8-bit range (0-255)
    img = (img / img.max() * 255).astype('uint8')

    # Create an RGB image (assuming single-channel grayscale DICOM)
    img_rgb = Image.fromarray(img, 'L').convert('RGB')

    return img_rgb

# Function to process image with RetinaNet model and display output
def process_and_display_image(image_path, model, score_threshold=0.3):
    st.image(image_path, caption="Input Image", use_column_width=True)

    # Read image in BGR and convert to RGB
    image = cv2.imread(image_path)[...,::-1]

    # Copy to draw on
    draw = image.copy()

    # Preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # Process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # Correct for image scale
    boxes /= scale

    # Visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < score_threshold:
            break

        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    # Display only the output image
    st.image(draw, caption="Output Image", use_column_width=True)

# Streamlit app
def main():
    st.title("DICOM/Image Processing App")

    uploaded_files = st.file_uploader("Upload DICOM or Image files", type=["dcm", "jpg", "jpeg", "png"], accept_multiple_files=True)
    score_threshold = st.slider("Score Threshold", min_value=0.0, max_value=1.0, value=0.3)

    # Load RetinaNet model
    model_path = './snapshots/resnet152_pascal.h5'
    model = models.load_model(model_path, backbone_name='resnet152')
    model = models.convert_model(model)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if the uploaded file is a DICOM
            if uploaded_file.name.lower().endswith('.dcm'):
                img = dicom_to_rgb(uploaded_file)
            else:
                img = Image.open(uploaded_file)

            # Save temporary image
            temp_img_path = "temp_image.jpg"
            img.save(temp_img_path)

            # Process and display the image
            process_and_display_image(temp_img_path, model, score_threshold)

            # Remove temporary image
            os.remove(temp_img_path)

if __name__ == "__main__":
    main()
