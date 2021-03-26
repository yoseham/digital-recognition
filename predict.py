from sklearn.externals import joblib
import pickle
import threading
import json
from keras.models import load_model

from keras.applications.imagenet_utils import decode_predictions
import os
from PIL import Image
import numpy as np
from ELM import ELM
from SemiNBC import AODE
import sys
# import keras.backend.tensorflow_backend as tb

def load_models():
    models = []
    for modelname in modelNames:
        print(modelname)
        if modelname in ['ELM','SemiNBC']:
            pickle_in = open('models/'+modelname+'.pickle','rb')
            models.append(pickle.load(pickle_in))
        else:
            model = load_model('models/'+modelname+'.h5')
            models.append(model)
    return models


def get_features(array):
    
    h, w = array.shape
    data = []
    for x in range(0, int(w/4)):
        offset_y = x * 4
        temp = []
        for y in range(0,int(h/4)):
            offset_x = y * 4
            temp.append(sum(sum(array[0+offset_y:4+offset_y,0+offset_x:4+offset_x])))
        data.append(temp)
    return np.array(data)

def check():
    file_names = os.listdir('mnist/')
    print(file_names)
    for image_name in file_names:
        if image_name in finish:
            continue
        else:
            result = predict(image_name)
            print(result)
            with open('predicts/'+image_name.replace('.png','.json'),'w') as f:
                json.dump(result,f)
            finish.append(image_name)
    global timer
    timer = threading.Timer(3, check)
    timer.start()

def predict(img_name):
    img_name = 'mnist/'+img_name
    print(img_name)
    img = Image.open(img_name)
    dict = {}
    for i in range(len(modelNames)):
        model = models[i]
        modelname = modelNames[i]
        if modelname in ['ELM','SemiNBC']:
            image = img.point(lambda x:1 if x > 20 else 0)
            data = np.array(image)
            data = np.array([get_features(data)])
            X = data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
            y = model.evaluate(X)
            # yp= np.round(y[0]/sum(y[0]),4)
            dict[modelname]={}
            for i in range(len(y[0])):
                dict[modelname][str(i)] = str(np.round(y[0][i],6))
        else:
            data = np.array(img)
            data = np.array([data/255.0])
            X = data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
            if modelname == 'CNN':
                X = X.reshape(X.shape[0],28,28,1).astype('float32')
            predict = model.predict(X)
            dict[modelname]={}
            for i in range(len(predict[0])):
                dict[modelname][str(i)] = str(predict[0][i])

    return dict

if __name__ == '__main__':
    file_names = os.listdir('mnist/')
    print(file_names)
    global modelNames
    modelNames = ['CNN','MLP','Softmax','ELM']
    global models
    models = load_models()
    global finish
    finish = []
    check()





