import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
from util import classify, set_background


set_background('bgimg.jpeg')

# Set title with black color
st.markdown("<h1 style='color: black;'>CHEST X-RAY EVALUATION</h1>", unsafe_allow_html=True)

# Set header with black color
st.markdown("<h2 style='color: black;'>Please upload a chest X-ray image</h2>", unsafe_allow_html=True)

#upload a file
file = st.file_uploader('',type=['jpeg','jpg','png'])

#load classifier
model = load_model("keras_model.h5", compile=False)

# load class names
with open('labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    # st.image(image, use_column_width=True)

    # Define the desired width and height
    desired_width = 400
    desired_height = 400

    # Resize the image to the desired dimensions
    resized_image = image.resize((desired_width, desired_height))

    # Display the resized image using Streamlit
    st.image(resized_image)

    #classify image
    class_name, confidence_score = classify(image, model, class_names)

    # Write classification with black text
    st.markdown(f"<h2 style='color: black;'>{class_name}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color: black;'>Confidence Score: {int(confidence_score * 100)}%</h4>", unsafe_allow_html=True)

