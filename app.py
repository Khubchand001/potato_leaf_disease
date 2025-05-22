import streamlit as st
from PIL import Image
from predict import predict

# Make set_page_config the first Streamlit call
st.set_page_config(page_title="Potato Leaf Disease Detector", layout="centered")

st.title("🥔 Potato Leaf Disease Detector")
st.write("Upload a potato leaf image to identify the disease.")

# File uploader with drag-and-drop
uploaded_file = st.file_uploader("Drag and drop or click to upload an image", type=["jpg", "jpeg", "png"])

<<<<<<< HEAD
if uploaded_file is not None:
    try:
        # Show uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
=======
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
>>>>>>> 174a9cbed1ae6a446599d07c93b4beab5c555265

        # Prediction
        with st.spinner("Analyzing..."):
            label, confidence = predict(image)
            st.success(f"**Prediction:** {label.replace('_', ' ').title()} ({confidence * 100:.2f}% confidence)")

    except Exception as e:
        st.error(f"Error: {e}")
