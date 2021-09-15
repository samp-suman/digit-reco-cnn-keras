from flask import Flask, render_template, request
from imageio import imsave, imread
from datetime import datetime
# import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import re
import base64
import scipy.misc
modal = tf.keras.models.load_model('pkl/keras_modal.h5')

app = Flask(__name__)

response = None
x = None

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    now = str(datetime.now()).replace(':', '')


    # get data from drawing canvas and save as image
    img_data=request.get_data()
    if img_data:
        parseImage(img_data)
    user_input = request.args.get('user-input')


    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.jpg', pilmode='L')
    # read parsed image back in 8-bit, black and white mode (L)
    # plt.imsave(open('output/img-{a}.jpg'.format(a=now), 'wb'), x)
    x = np.invert(x)
    x = resize(x, (28, 28))
    
    out = modal.predict(x.reshape(1,28,28))
    response = np.argmax(out, axis=1)
    return str(response[0])

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.jpg', 'wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == '__main__':
    app.debug = True
    app.run()
