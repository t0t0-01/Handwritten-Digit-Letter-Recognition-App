from flask import Flask
from flask import request
import random
import io
import base64 
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import predict_letter_from_image
from main import load_model



rf_model = load_model()

app = Flask(__name__)


def analyze():
    nb = random.randint(0, 100)
    return str(nb)


def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    temp = Image.open(io.BytesIO(imgdata))
    final = cv2.cvtColor(np.array(temp), cv2.COLOR_BGR2RGB)
    return final
    


@app.route('/')
def main():
    return analyze()


@app.route("/upload-img", methods=["POST"])
def upload():
    if request.method == 'POST':
        string = request.json["base64"]
        img_array = stringToImage(string)
        img = img_array[1450:2130, :, :]
        
        plt.imshow(img)
        plt.show()
        
        if np.mean(img) != 255:
            cv2.imwrite("./temp/letter.png", img)
            y = predict_letter_from_image(rf_model, "./temp")
            return str(y[0])
        
        return ""

app.run("0.0.0.0", port=5000)