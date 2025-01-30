import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configure the page
st.set_page_config(
    page_title="PhytoScan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated styling with a fresh color palette
st.markdown("""
    <style>
        .stApp {
            background-color: #2C3E50;
            color: #ECF0F1;
        }
        .stButton > button {
            background-color: #27AE60;
            color: #ECF0F1 !important;
            border-radius: 8px;
            font-weight: bold;
            border: none;
            padding: 0.6rem 1.2rem;
        }
        .css-1d391kg {
            background-color: #34495E;
        }
        h2, h3, h1 {
            color: #ECF0F1 !important;
            font-weight: bold !important;
            text-align: center;
        }
        p, li {
            color: #BDC3C7 !important;
            font-size: 17px !important;
            text-align: center;
        }
        a {
            color: #27AE60 !important;
            font-weight: 500;
        }
        .sidebar-text {
            color: #ECF0F1 !important;
            font-size: 19px;
            font-weight: bold;
            text-align: center;
        }
        .container {
            display: flex;
            justify-content: space-around;
        }
        .box {
            background-color: #34495E;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            width: 30%;
        }
    </style>
""", unsafe_allow_html=True)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("🌾 PhytoScan")
st.sidebar.markdown("<p class='sidebar-text'>AI-Powered Plant Disease Detection</p>", unsafe_allow_html=True)
app_mode = st.sidebar.selectbox("Select page", ["home", "disease recognition"])

# Display plant-themed icons
st.sidebar.markdown("🌿🍃🌱")
st.sidebar.markdown("""
    PhytoScan leverages AI to identify plant diseases early.
    Upload an image for a rapid health assessment!
""")

if app_mode == "home":
    st.markdown("<h1>PhytoScan: Smart Plant Health Monitoring</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='container'>
            <div class='box'>
                <h3>About PhytoScan</h3>
                <p>
                    PhytoScan is an AI-driven tool designed to detect plant diseases efficiently.
                    Upload an image, and our model will analyze and provide an accurate diagnosis.
                </p>
            </div>
            <div class='box'>
                <h3>How It Works</h3>
                <p>
                    1. Upload a plant leaf image<br>
                    2. AI analyzes the image<br>
                    3. Get instant results with a diagnosis
                </p>
            </div>
            <div class='box'>
                <h3>Technologies Used</h3>
                <p>
                    - Streamlit<br>
                    - TensorFlow & Keras<br>
                    - NumPy<br>
                    - Pillow
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif app_mode == "disease recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an Image for Analysis:")
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        if st.button("Analyze Disease"):
            st.balloons()
            st.write("AI Analysis Result")
            result_index = model_prediction(test_image)
            class_names = ['Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
                           'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy', 'Corn - Gray Leaf Spot',
                           'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy', 'Grape - Black Rot',
                           'Grape - Black Measles', 'Grape - Leaf Blight', 'Grape - Healthy', 'Orange - Citrus Greening',
                           'Peach - Bacterial Spot', 'Peach - Healthy', 'Pepper - Bacterial Spot', 'Pepper - Healthy',
                           'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy',
                           'Soybean - Healthy', 'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
                           'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold',
                           'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot',
                           'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy']
            st.success(f"Diagnosis: {class_names[result_index]}")

# Footer
st.markdown("<hr style='margin: 30px 0px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #27AE60;'>PhytoScan © 2025 | AI for Sustainable Farming</p>", unsafe_allow_html=True)