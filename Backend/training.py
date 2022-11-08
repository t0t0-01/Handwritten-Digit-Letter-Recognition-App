import os 
import re
import numpy as np
import pandas as pd
from features import zoning
from features import profiles
from features import per_Pixel
from features import intersections
from features import getHistograms
from features import invariantMoments
from features import divisionPoints
from pre_processing import get_bounding_box

def train(split = False):

    if split:
        splitDataset()
    
    labels_train = np.genfromtxt("./split/labelstrain.csv", dtype=str, delimiter=',')
    ys_train = np.genfromtxt("./split/ystrain.csv", dtype=str, delimiter=',')
    labels_test = np.genfromtxt("./split/labelstest.csv", dtype=str, delimiter=',')
    ys_test = np.genfromtxt("./split/ystest.csv", dtype=str, delimiter=',')

    y_test = []
    y_train = []

    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []

    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    c6 = []
    c7 = []

    directory = 'dataset'
    for filename in os.scandir(directory):
        if filename.is_file():
            if filename.name in labels_train:
                image = get_bounding_box(directory, filename.name, '.out')
                x1.append(invariantMoments(image))
                x2.append(intersections(image))
                x3.append(per_Pixel(image))
                x4.append(profiles(image))
                x5.append(getHistograms(image))
                x6.append(zoning(image))
                arr = []
                x7.append(divisionPoints(image,0,0,[0,0],arr,0,5))
                y_train.append(ys_train[np.where(labels_train == filename.name)])
                print (filename.name, ' Done')
            elif filename.name in labels_test:
                image = get_bounding_box(directory, filename.name, '.out')
                c1.append(invariantMoments(image))
                c2.append(intersections(image))
                c3.append(per_Pixel(image))
                c4.append(profiles(image)) 
                c5.append(getHistograms(image))
                c6.append(zoning(image))
                arr = []
                c7.append(divisionPoints(image,0,0,[0,0],arr,0,5))
                y_test.append(ys_test[np.where(labels_test == filename.name)])
                print (filename.name, ' Done')


    x_train = np.concatenate((x2, x3, x4, x7), axis=1)
    x_test = np.concatenate((c2, c3, c4, c7), axis=1)
    np.savetxt('./data/xtrain.csv', x_train, fmt = '%d', delimiter=",") 
    np.savetxt('./data/ytrain.csv', y_train, fmt = '%s', delimiter=",")
    np.savetxt('./data/xtest.csv', x_test, fmt = '%d', delimiter=",")
    np.savetxt('./data/ytest.csv', y_test, fmt = '%s', delimiter=",")

def splitDataset():
    df = pd.read_csv('./dataset/english.csv')
    arr = df.to_numpy()
    np.random.shuffle(arr) 

    ys_train, ys_test, blankArr = np.split(arr, [2557,3410]) #Divide 75-25

    label_train = ys_train[:, 0]
    labels_train = []
    label_test = ys_test[:, 0]
    labels_test = []

    for i in label_train:
        labels_train.append(re.sub(r'.', '', i, count=4))

    for i in label_test:
        labels_test.append(re.sub(r'.', '', i, count=4))        

    ys_train = ys_train[:,1]
    ys_test = ys_test[:,1]

    np.savetxt('./split/labelstrain.csv', labels_train, fmt = '%s', delimiter=",") 
    np.savetxt('./split/ystrain.csv', ys_train, fmt = '%s', delimiter=",")
    np.savetxt('./split/labelstest.csv', labels_test, fmt = '%s', delimiter=",")
    np.savetxt('./split/ystest.csv', ys_test, fmt = '%s', delimiter=",")


def getData(sp=False, tr = False):
    if tr:
        train(sp)

    x_train = np.loadtxt("./data/xtrain.csv", delimiter=",")
    y_train = np.genfromtxt("./data/ytrain.csv", dtype=str, delimiter=',') 
    x_test = np.loadtxt("./data/xtest.csv", delimiter=",")
    y_test = np.genfromtxt("./data/ytest.csv", dtype=str, delimiter=',') 

    return x_train, y_train, x_test, y_test