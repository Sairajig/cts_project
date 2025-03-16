import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  
        layers.Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def main():
    data_dir = 'mri_scan' 
    train_datagen = ImageDataGenerator(
        rescale=1./255,            
        rotation_range=20,         
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2      
    )

    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    model = build_cnn()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Training CNN model with augmented data...")
    history = model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[early_stopping])

    model.save('./models/mri_cnn_model_augmented.h5')
    print("Model saved to './models/mri_cnn_model_augmented.h5'")

    plot_training_history(history)

if __name__ == "__main__":
    main()
