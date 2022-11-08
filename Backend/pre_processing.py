import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    

def merge_near(boxes, threshold=250):
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
    t, imbw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
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
    
    plt.imshow(image[:,:,::-1])
    plt.show()
    
    ROI = original[y:y+h, x:x+w]
    imbw = imbw[y:y+h, x:x+w]
    ROI = cv2.resize(ROI, (40, 30), interpolation = cv2.INTER_CUBIC)[:,:,0]
    imbw = cv2.resize(imbw, (40, 30), interpolation = cv2.INTER_CUBIC)
    
    plt.imshow(imbw,  cmap="gray")
    plt.show()

    
    cv2.imwrite(f'{output_directory}/{fname}', imbw)
    
    
    if show:
        cv2.imshow('image', imbw)
        cv2.waitKey()
    
    return imbw

#get_bounding_box(r"C:\Users\Anton\Desktop","test.png", r"C:\Users\Anton\Desktop", show=True)