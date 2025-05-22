import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Load model and class label mapping
model_path = "models/potato_model.h5"
label_map_path = "models/label_map.npy"

# Cache the model for efficient reuse
@tf.keras.utils.register_keras_serializable()
def load_model_and_labels():
    model = load_model(model_path)
    label_map = np.load(label_map_path, allow_pickle=True).item()
    labels = {v: k for k, v in label_map.items()}
    return model, labels

model, labels = load_model_and_labels()

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]
    pred_class = np.argmax(preds)
    confidence = preds[pred_class]
    label = labels[pred_class]
    return label, confidence
