# File containing all utility functions

#Imports
import cv2, os
from PIL import Image

# Function that loads the image from the file into a ndarray of shape (H x W x C)
def loadImage(image_path):
    return cv2.imread(image_path)

# Function that removes every file from the given directory
def clearDirectory(directory_path):
    for fileName in os.listdir(directory_path):
        os.remove(os.path.join(directory_path, fileName))

# Function that converts images from a specified directory into a GIF
def createGif(imagePath):
    imgs = []
    for f in os.listdir(imagePath):
        img = Image.open(os.path.join(imagePath, f))
        imgs.append(img)

    image = imgs[0]
    image.save(fp=r"C:\Users\tremb\Dev\flask-pytorch-object-recognition\static\outputs\sequence.gif",
        format='GIF', append_images=imgs, save_all=True, duration=400, loop=0)