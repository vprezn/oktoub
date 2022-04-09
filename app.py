from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from keras.models import load_model
import numpy as np
import cv2

#Initialize the flask App
app = Flask(__name__)
# cors = CORS(app, resources={r"/api/": {"origins": ""}})


model = load_model('keras_model.h5')


# CORS Headers
# @app.after_request
# def after_request(response):
#     header = response.headers
#     header['Access-Control-Allow-Origin'] = '*'
#     header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
#     header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
#     return response



@app.route('/')
def home():
    return "Ahmad"


@app.route('/predict')
def predict():
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # preprocessing
    image = cv2.imread('2 - 02 - تمساح.png')
    res = cv2.resize(image, dsize=(224, 224))

    #turn the image into a numpy array
    image_array = np.asarray(res)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    prediction = model.predict(data).argmax()
    
    return prediction

