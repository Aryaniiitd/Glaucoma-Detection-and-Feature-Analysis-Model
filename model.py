import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Define directories
image_directory = '/Users/aryansharma/Desktop/ml-project/selected_images'
train_dir = '/Users/aryansharma/Desktop/ml-project/train'
val_dir = '/Users/aryansharma/Desktop/ml-project/validation'
test_dir = '/Users/aryansharma/Desktop/ml-project/test'

# Create directories if they don't exist
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Split data into train, validation, and test sets
def split_data(image_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    train_files, test_val_files = train_test_split(image_files, test_size=(1 - train_ratio), random_state=random_state)
    val_files, test_files = train_test_split(test_val_files, test_size=test_ratio/(test_ratio + val_ratio), random_state=random_state)
    return train_files, val_files, test_files

# Move images to respective directories
def move_images(image_files, src_dir, dst_dir):
    for filename in image_files:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)
        shutil.copy(src, dst)

# List of image file names
image_files = os.listdir(image_directory)
num_images = len(image_files)

# Shuffle the data
image_files = shuffle(image_files, random_state=42)

# Split the data
train_files, val_files, test_files = split_data(image_files)

# Move images to respective directories
move_images(train_files, image_directory, train_dir)
move_images(val_files, image_directory, val_dir)
move_images(test_files, image_directory, test_dir)

# Define image dimensions and batch size
img_height, img_width = 64, 64
batch_size = 32

# Create ImageDataGenerators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Define class weights
class_weights = {0: 1, 1: 10}  # Adjust the weights based on the class distribution

# Define the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Define input shape
input_shape = (img_height, img_width, 3)

# Create the CNN model
model = create_cnn_model(input_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    class_weight=class_weights
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save('/Users/aryansharma/Desktop/ml-project/glaucoma_classifier_model.h5')
