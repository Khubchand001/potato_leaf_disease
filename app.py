import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("models/potato_model.h5")
classes = ['Early Blight', 'Late Blight', 'Healthy']

# Page setup
st.set_page_config(page_title="Potato Leaf Disease Detection", layout="centered")

# Prediction function
def predict(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    return classes[np.argmax(pred)], pred

# Custom CSS for responsiveness
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.3em;
        color: #e7e7e7;
    }
    .subtitle {
        text-align: center;
        font-size: 1em;
        color: #555;
        margin-bottom: 2em;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1em;
    }
    .result-box {
        background-color: #eafbea;
        border-left: 6px solid #25a244;
        padding: 1em;
        margin-top: 1em;
        font-size: 1.1em;
        color: #1c6c39;
        text-align: center;
        border-radius: 8px;
    }
    .footer {
        margin-top: 3em;
        text-align: center;
        font-size: 0.9em;
        color: #aaa;
    }
    @media (max-width: 768px) {
        .title { font-size: 2em; }
        .subtitle { font-size: 0.95em; }
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🥔 Potato Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of a potato leaf and let the model detect the disease</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)

    # Show image
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict
    with st.spinner('Analyzing...'):
        label, confidence = predict(img)

    st.markdown(f'<div class="result-box">🩺 Prediction: <strong>{label}</strong></div>', unsafe_allow_html=True)

    # Show confidence
    st.subheader("📊 Confidence Scores:")
    for i, cls in enumerate(classes):
        st.write(f"**{cls}**: {confidence[i]*100:.2f}%")
        st.progress(float(confidence[i]))

# Footer
st.markdown("""
    <div class="footer">
        Made with ❤️ by <strong>Khubchand</strong>
    </div>
""", unsafe_allow_html=True)
