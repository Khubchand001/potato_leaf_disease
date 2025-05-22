import streamlit as st
from PIL import Image
from predict import predict

# Make set_page_config the first Streamlit call
st.set_page_config(page_title="Potato Leaf Disease Detector", layout="centered")

st.title("🥔 Potato Leaf Disease Detector")
st.write("Upload a potato leaf image to identify the disease.")

# File uploader with drag-and-drop
uploaded_file = st.file_uploader("Drag and drop or click to upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Show uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Prediction
        with st.spinner("Analyzing..."):
            label, confidence = predict(image)
            st.success(f"**Prediction:** {label.replace('_', ' ').title()} ({confidence * 100:.2f}% confidence)")

    except Exception as e:
        st.error(f"Error: {e}")
