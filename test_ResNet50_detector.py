# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, default="new_dataset",
                help="path to out input directory of images")
ap.add_argument("-m", "--model", type=str, default="output/crop_detector.model",
                help="path to pre-trained model")
args = vars(ap.parse_args())

# load our serialized Crop detection model from disk
print("[INFO] loading Crop detection model network...")
model = load_model(args["model"])

# define the class labels
class_names = ['banana', 'cassava', 'grass', 'maize']  # Add your class labels here

# define the label colors (BGR format)
label_colors = {
    0: (0, 0, 255),  # Red
    1: (0, 255, 0),  # Green
    2: (255, 0, 0),  # Blue
    3: (255, 255, 255),  # White
    # Add more colors for additional classes
}

# grab all image paths in the input directory and randomly sample them
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:50]

# initialize our list of results
results = []

# loop over our sampled image paths
for p in imagePaths:
    # load our original input image
    orig = cv2.imread(p)

    # pre-process our image by converting it from BGR to RGB channel
    # ordering (since our Keras model was trained on RGB ordering),
    # resize it to 224x224 pixels, and then scale the pixel intensities
    # to the range [0, 1]
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0

    # add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    # make predictions and assess confidence level on the input image
    pred = model.predict(image)
    predicted_index = np.argmax(pred)
    predicted_label = class_names[predicted_index]
    confidence = pred[0][predicted_index]

    # display confidence level to terminal
    text = " {:.2f}%".format(confidence * 100)

    # class predictions
    label = predicted_label + str(text)
    color = label_colors[predicted_index]

    # resize our original input (so we can better visualize it) and
    # then draw the label on the image
    orig = cv2.resize(orig, (224, 224))
    cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)

    # add the output image to our list of results
    results.append(orig)

# create a montage using 128x128 "tiles" with 10 rows and 5 columns
montage = build_montages(results, (128, 128), (10, 5))[0]

# save show the output montage
cv2.imwrite("Crop-predictions.png", montage)
cv2.imshow("Crop Detection Results", montage)
cv2.waitKey(0)
