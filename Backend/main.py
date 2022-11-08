import joblib
import numpy as np
import pandas as pd
from sklearn import svm
#from Testing import test
from features import zoning
from training import getData
from features import profiles
from features import per_Pixel
from collections import Counter
from features import intersections
from features import getHistograms
from features import invariantMoments
from features import divisionPoints
from pre_processing import get_bounding_box 
from sklearn.ensemble import RandomForestClassifier



def load_model():
    """
    Loads the trained model
    

    Returns
    -------
    rf_model : Object
        The loaded model.

    """
    
    """
    getData takes 2 parameters:
    sp: to resplit the data (True) or to leave the same split (False)
    tr: to retrain the data (True) or to just get the pre-trained features (train = get feature vectors))
    """

    
    x_train, y_train, x_test, y_test = getData(sp=0, tr=0)
    

    """sci-kit learn random forest"""
    rf_model = joblib.load('./models/rf_model')
    
    return rf_model




def predict_letter_from_image(rf_model, temp_dir):
    """
    Takes as input a model and generates a prediction from an image called
    'letter.png' in temp_dir

    Parameters
    ----------
    rf_model : Object
        The model.
    temp_dir : TYPE
        Directory where 'letter.png' is located

    Returns
    -------
    y_pred : str
        Predicted letter.

    """
    


    """
    getData takes 2 parameters:
    sp: to resplit the data (True) or to leave the same split (False)
    tr: to retrain the data (True) or to just get the pre-trained features (train = get feature vectors))
    """
    
    x_train, y_train, x_test, y_test = getData(sp=0, tr=0)
    
    """
    Pass in 'True' at the end of test() to refit model
    Don't pass anything or pass in 'False' to use saved models
    """
    #test(x_train, y_train, x_test, y_test, 0)
    
    ##################################################
    #### PUT THE IMAGE WE WANT TO PREDICT ON HERE ####
    ##################################################
    im = image = get_bounding_box(temp_dir, 'letter.png', "./temp")
    ##################################################
    
    arr = []
    a1 = invariantMoments(im)
    a2 = intersections(im)
    a3 = per_Pixel(im)
    a4 = profiles(im)
    a5 = getHistograms(im)
    a6 = zoning(im)
    a7 = divisionPoints(im,0,0,[0,0],arr,0,5)
    x_pred = np.concatenate((a2, a3, a4, a7))
    
    """sci-kit learn random forest"""
    y_pred = rf_model.predict(x_pred.reshape(1, -1))
    
    return y_pred


