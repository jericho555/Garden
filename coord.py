# import necessary packages
import cv2

# Variables to store the coordinates
top_left = None
bottom_right = None
cropping = False


def mouse_click(event, x, y, flags, param):
    global top_left, bottom_right, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        top_left = (x, y)
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        bottom_right = (x, y)
        cropping = False
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imshow("Image", image)

        print("Top-Left Coordinates: ({}, {})".format(top_left[0], top_left[1]))
        print("Bottom-Right Coordinates: ({}, {})".format(bottom_right[0], bottom_right[1]))


# Load the image
image = cv2.imread("resized/frame4.jpg")

# Create a window and set the callback function for mouse events
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_click)

# Display the image
cv2.imshow("Image", image)

# Wait for the user to define the region of interest
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        image = cv2.imread("resized/frame4.jpg")
        cv2.imshow("Image", image)
        top_left = None
        bottom_right = None
    elif key == ord("c"):
        if top_left is not None and bottom_right is not None:
            # Ensure the coordinates are within image bounds
            x1 = max(0, min(top_left[0], bottom_right[0]))
            y1 = max(0, min(top_left[1], bottom_right[1]))
            x2 = min(image.shape[1], max(top_left[0], bottom_right[0]))
            y2 = min(image.shape[0], max(top_left[1], bottom_right[1]))

            cropped_image = image[y1:y2, x1:x2]
            cv2.imshow("Cropped Image", cropped_image)
            cv2.imwrite("cropped_image.jpg", cropped_image)
    elif key == 27:
        break

# Clean up
cv2.destroyAllWindows()
