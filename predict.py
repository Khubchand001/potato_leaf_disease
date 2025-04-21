import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

classes = ['Early Blight', 'Late Blight', 'Healthy']
model = tf.keras.models.load_model("models/potato_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return classes[class_index]

if __name__ == "__main__":
    img_path = sys.argv[1]
    result = predict_image(img_path)
    print("Prediction:", result)
