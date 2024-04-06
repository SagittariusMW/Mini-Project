import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
from util import classify, set_background
from accuracy import accur_score

def homepage():
    set_background('bgimg.jpeg')

    # Set title with black color
    st.markdown("<h1 style='color: black;'>CHEST X-RAY EVALUATION</h1>", unsafe_allow_html=True)

    # Set header with black color
    st.markdown("<h3 style='color: black;'>Please upload a chest X-ray image</h3>", unsafe_allow_html=True)

    # upload a file
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    # load classifier
    model = load_model("keras_model.h5", compile=False)

    # load class names
    with open('labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

    # display image
    if file is not None:
        image = Image.open(file).convert('RGB')
        desired_width = 400
        desired_height = 400

        # Resize the image to the desired dimensions
        resized_image = image.resize((desired_width, desired_height))

        # Display the resized image using Streamlit
        st.image(resized_image)

        # classify image
        class_name, confidence_score = classify(image, model, class_names)

        # Write classification with black text
        st.markdown(f"<h2 style='color: black;'> {class_name}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: black;'>Confidence Score: {int(confidence_score * 100)}%</h4>",
                    unsafe_allow_html=True)
def about():
    st.title("About Page")
    st.write("This is the About Page.")

def performance():
    set_background('bgimg.jpeg')

    st.markdown("<h1 style='color: black;'>Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: black;'>Model performance in machine learning is a measurement of how accurate predictions or classifications a model makes on new, unseen data. </h5>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 1px solid black;'>", unsafe_allow_html=True)
    st.write("\n")
    st.markdown("<h3 style='color: black;'> • Epoch</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: black;'>An epoch in machine learning means one complete pass of the training dataset through the algorithm. It specifies the number of epochs or complete passes of the entire training dataset passing through the training process of the algorithm</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: black;'>Epochs : 7</h5>", unsafe_allow_html=True)
    st.write("\n")
    st.markdown("<h3 style='color: black;'> • Batch Size</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: black;'>A batch is a set of samples used in one iteration of training. For example, let's say that you have 80 images and you choose a batch size of 16. This means the data will be split into 80 / 16 = 5 batches. Once all 5 batches have been fed through the model, exactly one epoch will be complete.</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: black;'>Batch Size : 16</h5>", unsafe_allow_html=True)
    st.write("\n")
    st.markdown("<h3 style='color: black;'> • Confusion Matrix</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: black;'>A confusion matrix summarizes how accurate the model's predictions are. You can use this matrix to figure out which classes the model gets confused about.</h5>", unsafe_allow_html=True)
    image = Image.open("confusion matrix1.png").convert('RGB')
    desired_width = 400
    desired_height = 300
    resized_image = image.resize((desired_width, desired_height))
    st.image(resized_image)
    st.write("\n")
    st.markdown("<h3 style='color: black;'> • Accuracy for each class</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: black;'>This metric gives insights into how well the model performs for individual classes within the dataset. It helps identify if the model has biases or performs differently across different classes.</h5>", unsafe_allow_html=True)
    image = Image.open("accuracy_per_class.png").convert('RGB')
    desired_width = 400
    desired_height = 200
    resized_image = image.resize((desired_width, desired_height))
    st.image(resized_image)
    st.write("\n")
    st.markdown("<h3 style='color: black;'> • Accuracy score </h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: black;'>It provides an overall measure of how well the model performs across all classes. It's a commonly used metric for classification tasks but should be interpreted alongside other metrics for a comprehensive understanding of model performance.</h5>", unsafe_allow_html=True)
    accuracy = accur_score()
    st.markdown(f"<h5 style='color: black;'>Accuracy score :  {accuracy}</h5>", unsafe_allow_html=True)

# Add navigation links in the sidebar
page = st.sidebar.selectbox("Go to", ["Home", "About", "Model Performance"])

# Display content based on user selection
if page == "Home":
    homepage()
elif page == "About":
    about()
elif page == "Model Performance":
    performance()
