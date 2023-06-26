# import necessary packages
import os
import cv2

# Define the input and output directories
input_folder = "data"  # Specify the path to the input folder containing the images
output_folder = "resized"  # Specify the path to the output folder to save the resized images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Resize and save each image
for i, file in enumerate(image_files):
    # Read the image
    image_path = os.path.join(input_folder, file)
    image = cv2.imread(image_path)

    # Resize the image
    resized_image = cv2.resize(image, (900, 1200))

    # Save the resized image
    output_path = os.path.join(output_folder, "frame{}.jpg".format(i))
    cv2.imwrite(output_path, resized_image)

    print("Resized and saved:", output_path)
