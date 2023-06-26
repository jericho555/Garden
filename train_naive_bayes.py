# import necessary packages
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import os
import joblib


def extract_color_histogram(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the color histogram
    hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # Normalize the histogram
    cv2.normalize(hist, hist)

    return hist.flatten()


def extract_edges(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    return edges.flatten()


# Set the path to the folder containing the labeled images
dataset_path = 'new_dataset'

# Initialize empty lists for features and labels
features = []
labels = []

# Iterate over the labeled image folders
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)

    # Iterate over the image files in the class folder
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)

        # Load the image
        image = cv2.imread(image_path)

        # Extract features from the image
        color_hist = extract_color_histogram(image)
        edges = extract_edges(image)

        # Add the features and label to the lists
        features.append(np.concatenate((color_hist, edges)))
        labels.append(class_folder)  # Assuming the folder name represents the class/label

# Convert the feature and label lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a Naïve Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'output/naive_bayes_model.pkl')

# Generate predictions on the test set
y_pred = model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a classification report
report = classification_report(y_test, y_pred)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot the training curve
plt.figure(figsize=(8, 6))
plt.plot(model.theta_.T)
plt.legend(model.classes_)
plt.title("Training Curve")
plt.xlabel("Features")
plt.ylabel("Class Probability")
plt.show()
