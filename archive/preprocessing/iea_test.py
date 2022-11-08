import cv2
import numpy as np
import pandas as pd
import os
import io
import base64 
from PIL import Image
import matplotlib.pyplot as plt
import math
import itertools


def remove_contained(boxes):
    """
    Takes a list of boxes and returns a new list with boxes that are included 
    in other boxes removed

    Parameters
    ----------
    boxes : list
        Original list of bounding boxes.

    Returns
    -------
    updated : list
        Updated list without subboxes.

    """
    updated = []
    for x1, y1, w1, h1 in boxes:
        add = True
        for x2, y2, w2, h2 in boxes:
            #check if first is contained in second
            if x1 > x2 and x1+w1 < x2+w2 and y1>y2 and y1+h1 < y2+h2:
                add = False
        if add == True:
            updated.append((x1,y1,w1,h1))
    
    return updated
    

def merge_near(boxes, threshold=15):
    """
    Takes a list of bounding boxes and merges those that are near each other
    within a certain threshold

    Parameters
    ----------
    boxes : list
        Original list of bounding boxes.

    Returns
    -------
    list
        Updated list with merged boxes.

    """

    prev = boxes.copy()                 #returned value
    changed = True
    merged_boxes = set({})
    
    while changed == True:              #as long as changes have been made, loop through the list prev
        changed = False
        updated = []

        for pos, (x1, y1, w1, h1), in enumerate(prev):      
            if changed == False:
                if pos != len(prev) - 1:
                    for (x2,y2,w2,h2) in prev[pos+1:]:
                        if changed == False:                            
                            
                            if (np.abs(x2 + w2 - x1) < threshold or np.abs(x2 -x1 -w1) < threshold) and changed == False:
                                updated_x = min(x1,x2)
                                updated_y = min(y1,y2)
                                updated_w = max(x1+w1, x2+w2) - updated_x   #final point - start point
                                updated_h = max(y1+h1, y2+h2) - updated_y
                                
                                updated.append((updated_x, updated_y, updated_w, updated_h))
                                
                                merged_boxes.add((x1,y1,w1,h1))
                                merged_boxes.add((x2,y2,w2,h2))
                                
                                
                                #add all other boxes to updated for rerun
                                for box in boxes:
                                    if box not in merged_boxes:
                                        updated.append(box)
                                
                                changed = True
                                prev = updated
                            
                            elif changed == True and (x1,y1,w1,h1) not in merged_boxes:
                                updated.append((x1,y1,w1,h1))
                        
                        else:
                            break
                                
            else:
                break
                
                        
                    
                        
                     
                        
                if changed == True:
                    break
                

                
                
            
    return prev if prev else boxes
            
            

                    
                

def get_bounding_box(input_dir, fname, output_directory, show=False):
    #needs updating to work on multiple letters per image
    
    """
    Extracts the letter from an image by getting bounding boxes and resizes 
    the photo to 40x30x3 (RGB color stays)

    Parameters
    ----------
    input_dir : string
        Directory that contains the file to be analyzed (excluding the name of the file).
    fname : string
        The name of the file to be analyzed.
    output_directory : string
        Directory where the resulting file should be saved.
    show : boolean, optional
        Displays the image using OpenCV if True. The default is False.

    Returns
    -------
    None.

    """
    
    image = cv2.imread(f"{input_dir}/{fname}")
    original = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    
    bounding_rects = remove_contained([cv2.boundingRect(c) for c in cnts])
    bounding_rects = merge_near(bounding_rects)
    
    
    vertical_difference = 0
    x,y,w,h = bounding_rects[0]

    if len(bounding_rects) > 1:
        x1, y1, w1, h1 = bounding_rects[0]
        x2, y2, w2, h2 = bounding_rects[1]
        vertical_difference = np.abs(bounding_rects[0][1] - bounding_rects[1][1])
        if vertical_difference <= 240 and x2 < x1 + w1:  
            y = min(y1, y2)
            h += vertical_difference
            
            if min(x1, x2) == x2:
                x = x2
                w += np.abs(x1-x2)
            elif x2 + w2 >= x + w:
                w += np.abs((x2+w2) - (x+w))
                
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
    ROI = original[y:y+h, x:x+w]
    ROI = cv2.resize(ROI, (40, 30), interpolation = cv2.INTER_CUBIC)[:,:,0]
    cv2.imwrite(f'{output_directory}/{fname}', ROI)
    
    if show:
        cv2.imshow('image', image)
        cv2.waitKey()
    
    

    
def split_word(input_dir, fname, output_directory, show=False):
    image = cv2.imread(f"{input_dir}/{fname}")
    original = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    ROI_number = 0


    bounding_rects = [cv2.boundingRect(c) for c in cnts]
    bounding_rects = remove_contained(bounding_rects)
    bounding_rects = merge_near(bounding_rects)

    
    for bounding_rect in bounding_rects:
        x,y,w,h = bounding_rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite(f'{output_directory}/{fname}', ROI)
        ROI_number += 1
    

        if show:
            cv2.imshow('image', image)
            cv2.waitKey()
    
    
    """
    vertical_difference = 0
    x,y,w,h = bounding_rects[0]

    if len(bounding_rects) > 1:
        x1, y1, w1, h1 = bounding_rects[0]
        x2, y2, w2, h2 = bounding_rects[1]
        vertical_difference = np.abs(bounding_rects[0][1] - bounding_rects[1][1])
        if vertical_difference <= 240 and x2 < x1 + w1:  
            y = min(y1, y2)
            h += vertical_difference
            
            if min(x1, x2) == x2:
                x = x2
                w += np.abs(x1-x2)
            elif x2 + w2 >= x + w:
                w += np.abs((x2+w2) - (x+w))
                
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
    ROI = original[y:y+h, x:x+w]
    ROI = cv2.resize(ROI, (40, 30), interpolation = cv2.INTER_CUBIC)[:,:,0]
    """
    
    
    
def get_skeleton(input_dir, fname, output_directory, show=False):
    #Create an image with text on it
    img = cv2.imread(f"{input_dir}/{fname}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img1 = img.copy()
    
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape,dtype='uint8')
    
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        erode = cv2.erode(img1,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()

        

    cv2.imwrite(f'{output_directory}/{fname}', cv2.bitwise_not(thin))
    
    if show:
        cv2.imshow('thinned',thin)
        cv2.waitKey()


def extract_features(img):
    horizontal_half = img.shape[0] // 2
    vertical_half = img.shape[1] // 2
    non_zero = np.count_nonzero(img)
    
    
    per_pixels_above_horizontal = np.count_nonzero(img[:horizontal_half, :]) / non_zero * 100
    per_pixels_left = np.count_nonzero(img[:, :vertical_half]) / non_zero * 100
    
    
    return (per_pixels_above_horizontal, per_pixels_left)
    
    

    







def get_bounding_and_skeletons():
    """
    Generates bounding boxes and skeletons for all the images in the dataset.
    Outputs are in "dataset/img_boxes" and "dataset/img_skeletons"

    Returns
    -------
    None.

    """

    l = list(os.listdir("./dataset/Img"))
    print("Boxes started")
    for pos, e in enumerate(l):
        get_bounding_box("./dataset/Img", e, r'.\dataset\img_boxes')
        
        if pos % 100 == 0:
            print(f"{pos/len(l) * 100}% done")
        
        
    print("Boxes done")
    
    
        
        
        
    l = list(os.listdir(r".\dataset\img_boxes"))
    print("Skeletons started")
    for pos, e in enumerate(l):
        get_skeleton("./dataset/img_boxes", e, "./dataset/img_skeletons")
        
        if pos%100 == 0:
            print(f"{pos/len(l)*100}% done")


def main():
    #test stringToImage
    test = "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"
    plt.imshow(stringToImage(test))
    plt.show()
    
    
    get_bounding_and_skeletons()
    
    #get_skeleton(r"C:\Users\Anton\Desktop", "237-536x354.jpg", r"C:\Users\Anton\Desktop\github")
    
    #try features
    #image = cv2.imread('./test word.png')
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # print(extract_features(thresh))
    
    
#main()
#split_word("./", "test_letter.png", "./r", show=True)
get_skeleton("./", "img005-012.png", "./", show=True)