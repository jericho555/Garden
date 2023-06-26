# import necessary packages
import cv2
import numpy as np
from matplotlib import pyplot as plt


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


# Load the cropped image
image_path = 'new_dataset/banana/banana_images_1719.png'
image = cv2.imread(image_path)

# Extract color histogram
hist_features = extract_color_histogram(image)

# Extract edges
edge_features = extract_edges(image)

# Display the original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Display the color histogram
plt.subplot(1, 3, 2)
plt.plot(hist_features)
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')

# Display the edges
plt.subplot(1, 3, 3)
plt.imshow(edge_features.reshape(image.shape[:2]), cmap='gray')
plt.title('Edges')
plt.axis('off')

# Show the plots
plt.tight_layout()
plt.show()
