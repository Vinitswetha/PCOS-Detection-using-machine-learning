from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil


app = Flask(__name__, template_folder="template")

reg = pickle.load(open("model.pkl", "rb"))




# 

@app.route("/pcos")
def pcos():
     return render_template("userlog.html")

@app.route("/about")
def hello_wor():
     return render_template("about.html")


@app.route("/")
def hello_worl():
     return render_template("main.html")



@app.route("/choose")
def hello_wo():
     return render_template("choose.html")

@app.route("/remedy")
def hello_wr():
     return render_template("sol.html")

@app.route("/test")
def hello1():
     return render_template("test.html")



@app.route("/predict", methods=["POST"])
def home():
    data1 = float(request.form["b"])
    data2 = float(request.form["b"])
    data3 = float(request.form["c"])
    data4 = float(request.form["d"])
    d5 = float(request.form["e"])
    d6 = float(request.form["f"])
    d7 = float(request.form["g"])
    d8 = float(request.form["h"])
    d9 = float(request.form["i"])
    d10 = float(request.form["j"])
    d11 = float(request.form["k"])
    d12 = float(request.form["l"])
    d13 = float(request.form["m"])
    d14 = float(request.form["n"])
    d15 = float(request.form["o"])
    d16 = float(request.form["p"])
    d17 = float(request.form["q"])
    d18 = float(request.form["r"])
    d19 = float(request.form["s"])
    d20 = float(request.form["t"])
    d21 = float(request.form["u"])
    d22 = float(request.form["v"])
    d23 = float(request.form["w"])
    d24 = float(request.form["x"])
    d25= float(request.form["y"])
    d26 = float(request.form["z"])
    d27 = float(request.form["za"])
   
   

    arr = np.array(
        [
            [
                data1,
                data2,
                data3,
                data4,
                d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,
                d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27
                            ]
        ]
    )
    pred = reg.predict(arr)
    print(pred)
    return render_template("index.html", data=pred,d14=int(d14))

reg1 = pickle.load(open("model1.pkl", "rb"))

@app.route("/test1")
def hello2():
     return render_template("test2.html")


@app.route("/predict", methods=["POST"])
def hom():
    data1 = float(request.form["b"])
    data2 = float(request.form["b"])
    data3 = float(request.form["c"])
    data4 = float(request.form["d"])
    d5 = float(request.form["e"])
    d6 = float(request.form["f"])
    d7 = float(request.form["g"])
    d8 = float(request.form["h"])
    d9 = float(request.form["i"])
    d10 = float(request.form["j"])
    d11 = float(request.form["k"])
    d12 = float(request.form["l"])
    d13 = float(request.form["m"])
    d14 = float(request.form["n"])
    d15 = float(request.form["o"])
    d16 = float(request.form["p"])
    d17 = float(request.form["q"])
    d18 = float(request.form["r"])
    d19 = float(request.form["s"])
    d20 = float(request.form["t"])
    d21 = float(request.form["u"])
    d22 = float(request.form["v"])
    d23 = float(request.form["w"])
    d24 = float(request.form["x"])
    d25= float(request.form["y"])
    d26 = float(request.form["z"])
    d27 = float(request.form["za"])
    d28 = float(request.form["zb"])
    d29 = float(request.form["zc"])
    d30 = float(request.form["zd"])
    d31 = float(request.form["ze"])
    d32 = float(request.form["zf"])
    d33 = float(request.form["zg"])
    d34 = float(request.form["zh"])
    d35 = float(request.form["zi"])
    d36 = float(request.form["zj"])
    d37 = float(request.form["zk"])
   
   

    arr = np.array(
        [
            [
                data1,
                data2,
                data3,
                data4,
                d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,
                d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,
                d28,d29,d30,d31,d32,d34,d35,d36,d37
                            ]
        ]
    )
    pred1 = reg1.predict(arr)
    print(pred1)
    return render_template("index.html", data=pred1)

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("static/test/"+fileName, dst)
        image = cv2.imread("static/test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)

        # Apply thresholding to create a binary mask
        _, binary_mask = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the original image (optional)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Count the number of cells
        # cell_count = len(contours)
        # print(cell_count)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'PCOS_DETECTION1-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        
        str_label=" "
        accuracy=""
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            

            if np.argmax(model_out) == 0:
                str_label = "Normal"
                print("The predicted image of the Normal is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the Normal is with a accuracy of {:.2f}%".format(model_out[0]*100)
                
                
            elif np.argmax(model_out) == 1:
                str_label  = "PCOS"
                print("The predicted image of the PCOS is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the PCOS is with a accuracy of {:.2f}%".format(model_out[1]*100)


        return render_template('userlog.html', status=str_label,accuracy=accuracy ,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg")
        
    return render_template('userlog.html')


if __name__ == "__main__":
    app.run(debug=True)
