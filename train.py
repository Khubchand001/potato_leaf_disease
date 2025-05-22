import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Paths
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'
model_save_path = 'models/potato_leaf_model.h5'
class_index_save_path = 'models/class_indices.npy'
os.makedirs('models', exist_ok=True)

# Parameters
img_size = (250, 250)
batch_size = 32
epochs = 50

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Save class indices for prediction use
np.save(class_index_save_path, train_generator.class_indices)

# Model architecture
def create_model(input_shape=(250, 250, 3), num_classes=3):
    model = Sequential([
        Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='same'),

        Conv2D(164, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='same'),

        Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='same'),

        Flatten(),

        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),

        Dense(num_classes, activation='softmax')
    ])
    return model

# Compile and train
model = create_model(input_shape=(250, 250, 3), num_classes=train_generator.num_classes)
model.compile(optimizer=Adadelta(learning_rate=1.0, rho=0.95), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Save model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Plot training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.grid(True)
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.grid(True)
plt.title('Loss')

plt.tight_layout()
plt.show()
