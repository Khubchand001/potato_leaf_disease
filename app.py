import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Potato Leaf Disease Detection", layout="centered")

# Load trained model
model = tf.keras.models.load_model("models/potato_model.h5")
classes = ['Early Blight', 'Late Blight', 'Healthy']

# Custom CSS styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f7f7f7;
    }
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #2d2d2d;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
        color: #666;
    }
    .uploaded-img {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        border: 2px solid #25a244;
        border-radius: 10px;
        padding: 15px;
        background-color: #eafbea;
        font-size: 1.1em;
        color: #1c6c39;
        text-align: center;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Titles
st.markdown('<div class="main-title">🥔 Potato Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image of a potato leaf to identify disease type</div>', unsafe_allow_html=True)

# Prediction function
def predict(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    return classes[np.argmax(pred)], pred

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

# If file is uploaded
if uploaded_file:
    img = Image.open(uploaded_file)

    st.markdown('<div class="uploaded-img">', unsafe_allow_html=True)
    st.image(img, caption='Uploaded Image', use_container_width=True)



    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner('Analyzing the image...'):
        label, confidence = predict(img)

    # Display result
    st.markdown(f'<div class="prediction-box">🩺 Prediction Result: {label}</div>', unsafe_allow_html=True)

    # Show confidence scores
    st.write("📊 **Confidence Scores:**")
    for i, cls in enumerate(classes):
        st.progress(float(confidence[i]))

        st.markdown(f"- **{cls}**: `{confidence[i]*100:.2f}%`")
st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        Made with ❤️ by <strong>Khubchand</strong>
    </div>
""", unsafe_allow_html=True)