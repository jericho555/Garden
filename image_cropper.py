import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", type=str, default="resized", help="path to input image folder")
args = vars(ap.parse_args())

# Create the output folder if it doesn't exist
output_folder = "banan_images"
os.makedirs(output_folder, exist_ok=True)

# Get a list of image files in the input folder
image_files = os.listdir(args["folder"])

# Loop over each image filesssssss
for image_file in image_files:
    # Load the image
    image_path = os.path.join(args["folder"], image_file)
    image = cv2.imread(image_path)

    # Create a window for capturing coordinates
    cv2.namedWindow("Capture Coordinates")
    cv2.imshow("Capture Coordinates", image)
    coordinates = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Capture the top-left coordinate
            coordinates.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            # Capture the bottom-right coordinate
            coordinates.append((x, y))
            cv2.destroyAllWindows()

    cv2.setMouseCallback("Capture Coordinates", mouse_callback)

    while True:
        key = cv2.waitKey(1)

        # Break the loop if coordinates are captured
        if len(coordinates) == 2:
            break

        # Press 's' key to skip to the next image
        if key == ord("s"):
            coordinates = []  # Clear coordinates list
            break

        # Press 'Esc' key to exit the program
        if key == 27:
            cv2.destroyAllWindows()
            exit(0)

    # Skip to the next image if coordinates are not captured
    if not coordinates:
        continue

    # Define the coordinates for cropping
    startX, startY = coordinates[0]
    endX, endY = coordinates[1]

    # Crop the specified region from the image
    cropped = image[startY:endY, startX:endX]

    # Generate a counter-based filename for the cropped image
    counter = image_file.split(".")[0]  # Extract the counter from the filename
    output_filename = f"cropped_{counter}.jpg"

    # Save the cropped image to the output folder
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, cropped)

    # Display the cropped image
    cv2.imshow("Cropped Cassava", cropped)
    cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
