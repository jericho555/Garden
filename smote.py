# import necessary packages
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from imutils import paths
import numpy as np
import argparse
from PIL import Image
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="resized",
                help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    # label = os.path.basename(imagePath).split("_")[0]
    label = imagePath.split(os.path.sep)[-2]

    # Load the image and convert it to a numpy array
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    # Add the image and its label to the respective lists
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# Convert the image and label lists to numpy arrays
# print(data)
data = np.array(data)
labels = np.array(labels)

# Reshape the image array to a 2D array.
images = data.reshape(len(data), -1)

# Use SMOTE to oversampled the minority class.
print("[INFO] Oversampling data using SMOTE ...")

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(images, labels)

# Reshape the oversampled image array back to a 4D array.
X_resampled = X_resampled.reshape(len(X_resampled), 512, 512, 3)

# Define the directory to save the oversampled images
oversample_dir = "new_dataset"

# Loop through each oversampled image and save it to the directory
for i in range(len(X_resampled)):
    class_dir = os.path.join(oversample_dir, y_resampled[i])
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    img_path = os.path.join(class_dir, y_resampled[i] + "_" + str(i) + ".png")
    img = X_resampled[i]
    img = img.astype('uint8')
    img = Image.fromarray(img)
    img.save(img_path)

print("[INFO] Process Complete!")
