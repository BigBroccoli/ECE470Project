import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import cv2

# Load the MNIST dataset without pandas
mnist = fetch_openml('mnist_784', as_frame=False, data_home=None, parser='liac-arff')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

# Normalize the features to scale values between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Initialize the Multi-layer Perceptron Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=42,
                    learning_rate_init=0.1)

# Train the classifier
mlp.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = mlp.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display some of the test images and their predicted labels
plt.figure(figsize=(14, 7))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}")
    plt.axis('off')
plt.show()

def test_handwritten_digit(image_path):
    # Load the image of the handwritten digit
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not read the image.")
        return

    inverted_image = 255 - image

    # Resize the image to match the input size of the classifier (28x28)
    resized_image = cv2.resize(inverted_image, (28, 28))

    # Normalize the pixel values
    normalized_image = resized_image / 255.0

    # Flatten the image to make it compatible with the classifier
    flattened_image = normalized_image.flatten()

    # Predict the digit using the trained classifier
    predicted_digit = mlp.predict([flattened_image])

    print(f"Predicted Digit: {predicted_digit[0]}")

    # Display the image and the predicted digit
    plt.imshow(resized_image, cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit[0]}")
    plt.suptitle(image_path)
    plt.axis('off')
    plt.show()

# Test a handwritten digit image
test_handwritten_digit("D:/School/ECE 470/Project/Handwritten Numbers/7Stackoverflow.jpg")