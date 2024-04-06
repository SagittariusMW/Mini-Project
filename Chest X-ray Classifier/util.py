import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    #This function sets the background of a Streamlit app to an image specified by the given image file.

    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    # index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score





#
#
#
#
# from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps  # Install pillow instead of PIL
# import numpy as np
#
# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
#
# # Load the model
# model = load_model("keras_model.h5", compile=False)
#
# # Load the labels
# class_names = open("labels.txt", "r").readlines()
#
# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#
# # Replace this with the path to your image
# image = Image.open(r"C:\Users\sanika\OneDrive\Desktop\val\COVID19\COVID19(572).jpg").convert("RGB")
#
# # resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#
# # turn the image into a numpy array
# image_array = np.asarray(image)
#
# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
#
# # Load the image into the array
# data[0] = normalized_image_array
#
# # Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]
#
# # Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)
