import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize images
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=50, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Make predictions
predictions = model.predict(test_images)

def preprocess_image(image_path):
    # Load the image of the handwritten digit
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not read the image.")
        return

    image = cv2.resize(image, (28, 28))

    image = image.astype('float32') / 255.0

    image = np.expand_dims(image, axis=(0, -1))

    return image

def test_handwritten_digit(custom_image_path):
    custom_image = preprocess_image(custom_image_path)

    predicted_probabilities = model.predict(custom_image)
    predicted_class = np.argmax(predicted_probabilities)

    plt.imshow(cv2.imread(custom_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f"Predicted Digit: {predicted_class}")
    plt.axis('off')
    plt.show()

test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/0MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/1MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/2MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/3MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/4MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/5MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/6MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/7MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/8MSPaint.png")
test_handwritten_digit("D:/School/ECE 470/HW/ECE470Project/Handwritten Numbers/9MSPaint.png")
