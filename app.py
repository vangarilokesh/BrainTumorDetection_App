from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from keras.models import model_from_json
from PIL import Image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='sparse_categorical_crossentropy',
                     optimizer="adam",
                     metrics=['accuracy'])
print("Loaded model from disk")
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = Image.open(img_path)
    # resizing the image to resizing 50x50 and converting it to numpy array
    x = np.array(img.resize((50, 50)).convert('L'))
    x = x.reshape(1, 50, 50, 1)
    # predicting using the model
    pred = model.predict(x)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, loaded_model)
        index = np.where(preds == np.amax(preds))[1][0]
        arr = {0: 'No', 1:'Yes'}
        print(preds[0][index])
        result = arr[index]
        percentage=(str)(np.amax(preds)*100)
        percentage=percentage[:5]
        return result+"\nAccuracy: "+percentage
    return None


if __name__ == '__main__':
    app.run(debug=True)
