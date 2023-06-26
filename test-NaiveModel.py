# import necessary packages
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt


def extract_edges(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    return edges.flatten()


def extract_color_histogram(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the color histogram
    hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # Normalize the histogram
    cv2.normalize(hist, hist)

    return hist.flatten()


def draw_label(image, label, bbox):
    # Define the font properties
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL # cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1

    # Determine the text size
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Calculate the position for the label text
    label_position = (int(bbox[0]), int(bbox[1]) - 10)

    # Draw the bounding box rectangle
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    # Draw the label text background
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1]) - 10 - text_height),
                  (int(bbox[0]) + text_width, int(bbox[1]) - 10), (0, 255, 0), -1)

    # Draw the label text
    cv2.putText(image, label, label_position, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    return image


# Set the path to the trained model file
model_path = 'output/naive_bayes_model.pkl'

# Set the path to the image to be detected and identified
image_path = 'new_dataset/banana/banana_images_27.png'

# Load the trained model
model = joblib.load(model_path)

# Load the image
image = cv2.imread(image_path)

# Extract features from the image
color_hist = extract_color_histogram(image)
edges = extract_edges(image)

# Combine the features into a single feature vector
features = np.concatenate((color_hist, edges)).reshape(1, -1)

# Make predictions using the trained model
predictions = model.predict(features)

# Get the detected label
label = predictions[0]
print(label)

# Define the bounding box coordinates
bbox = (20, 20, 200, 200)  # Example coordinates, adjust as needed

# Draw the label on the image
image_with_label = draw_label(image.copy(), label, bbox)

# Create a figure with multiple subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

# Display the image with the detected label
axs[1].imshow(cv2.cvtColor(image_with_label, cv2.COLOR_BGR2RGB))
axs[1].set_title('Image with Label')
axs[1].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
