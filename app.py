# Main application file

# Imports
import os, secrets
from flask import Flask, request, render_template, flash, get_flashed_messages
from detector import Detector
from custom_utils import clearDirectory, createGif

secret = secrets.token_urlsafe(32)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["UPLOAD_FOLDER"] = r"C:\Users\tremb\Dev\flask-pytorch-object-recognition\uploads"
app.secret_key = secret

STATIC_OUTPUT_FOLDER = r"C:\Users\tremb\Dev\flask-pytorch-object-recognition\static\outputs"
OUTPUT_FOLDER = r"C:\Users\tremb\Dev\flask-pytorch-object-recognition\outputs"

detector = Detector()

# The home page route
@app.route('/')
def home():
    # Empty all the directories storing the temporary data
    clearDirectory(app.config["UPLOAD_FOLDER"])
    clearDirectory(STATIC_OUTPUT_FOLDER)
    clearDirectory(OUTPUT_FOLDER)
    return render_template("index.html")

# The detection route that acts as API
@app.route("/detect", methods=["POST"])
def upload_files():
    if request.method == "POST":
        # Get the uploaded files
        files = request.files.getlist("file")
        # Check number of files
        if (len(files) != 3):
            flash("You must select a full image sequence (3 images)", "error")
            # Re-render home page
            return render_template("index.html", message=get_flashed_messages()[0])
        else:
            outputImages = []
            nbOfDetections = 0

            for file in files:
                # Save image 
                imagePath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(imagePath)
                outputImages.append("outputs/"+file.filename)
                # Get detections from image
                boxes, classes = detector.get_prediction(imagePath, 0.70) # Confidence score threshold set to .70 or 70%
                # Increment the number of detections
                nbOfDetections += classes.count("person") + classes.count("vehicle")
                # Draw detections on image
                detector.create_visualization(imagePath, file.filename, boxes, classes, 2, 2, .7)
            # Create and save a GIF of the image sequence in the upload directory
            createGif(app.config["UPLOAD_FOLDER"])
            # Render the result page with the outputImages and nbOfDetection route parameters
            return render_template("result.html", outputImages=outputImages, nbOfDetections=nbOfDetections)
    return render_template("index.html")

# Launches local web app when file is ran
if __name__ == "__main__":
    app.run()