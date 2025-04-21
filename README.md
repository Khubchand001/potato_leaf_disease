# 🥔 Potato Leaf Disease Detection

A simple and intuitive web app built with **Streamlit** that detects diseases in potato leaves using a trained deep learning model. This system helps farmers and researchers identify **Early Blight**, **Late Blight**, or **Healthy** potato leaves by uploading an image.

---

## 🚀 Live Demo

🔗 [Click here to try the app](https://your-streamlit-url.streamlit.app)  
*(Replace with your actual Streamlit app URL)*

---

## 🧠 Model Info

The model is a Convolutional Neural Network (CNN) trained on potato leaf disease images with three classes:
- Early Blight
- Late Blight
- Healthy

---

## 🖼️ Features

✅ Upload any image of a potato leaf  
✅ Classify the disease in real-time  
✅ Display prediction with confidence scores  
✅ Mobile-friendly, responsive UI  
✅ Lightweight and fast!

---

## 🛠️ How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/potato-disease-detector.git
cd potato-disease-detector


Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py




├── app.py               # Streamlit app
├── models/
│   └── potato_model.h5  # Trained CNN model
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
