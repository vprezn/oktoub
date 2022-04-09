from flask import Flask, request, jsonify
# from flask_cors import CORS

from keras.models import load_model
import numpy as np
import cv2
import json
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


@app.route('/predict',methods=['POST'])
def predict():

    r = request.get_json()
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    with open('data.json', 'w') as f:
        json.dump(r, f)
    # preprocessing

    r = np.asarray(r, dtype = np.uint8)
    img = cv2.imdecode(r, cv2.IMREAD_COLOR)
    # res = cv2.resize(img, dsize=(224, 224))

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    constant= cv2.copyMakeBorder(rect,20,20,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])
    res = cv2.resize(constant, dsize=(224, 224))

    #turn the image into a numpy array
    image_array = np.asarray(res)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    prediction = model.predict(data).argmax()
    
    return jsonify({
        "success":True,
        "pred":str(prediction)
    })

# def preprocessing(data):
    
