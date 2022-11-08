import numpy as np
from pre_processing import get_bounding_box
import matplotlib.pyplot  as plt
import cv2

"""
Histograms: 
    get the number of pixels in each row and in each column
    form bins where each bin contains 5 rows or 5 columns to reduce sensitivity to noise
"""
def getHistograms(image):
    timage = image.T
    hor = []
    ver = []

    for i in image:
        hor.append(len(i)-np.count_nonzero(i))
    for i in timage:
        ver.append(len(i)-np.count_nonzero(i))

    # Dividing them into bins of 5 rows/column each
    # In order to reduce sensitivity to noise
    hor = formBins(hor)
    ver = formBins(ver) 
    return np.concatenate((hor,ver))

def formBins(mat):
    ret = []
    sum = 0
    for i in range (0, len(mat)):
        if i%5==0 and i!=0:
            ret.append(sum)
            sum = 0
        elif i==len(mat)-1:
            sum += mat[i]
            ret.append(sum)
            break
        sum += mat[i]
    return ret


"""
Invariant Moments:
    7 statistical computations that are generally stable and character dependent 
    where regardless of tilting or not, they have the same moments
"""
def getMu(p, q, image): 
    # black pixels are represented by 0
    black = np.argwhere(image == 0)
    
    # number of black pixels
    total = len(black)

    x = black[:,0]
    y = black[:,1]

    x_bar = np.sum(x) / total
    y_bar = np.sum(y) / total

    numerator = np.sum(np.power((x-x_bar),p) * np.power((y-y_bar),q))
    denominator = (np.sum(np.power((x-x_bar),2)) + np.sum(np.power((y-y_bar),2))) ** (((p + q) / 2) + 1)

    return numerator/denominator

def invariantMoments(image): 
    mu20 = getMu(2,0,image)
    mu02 = getMu(0,2,image)
    mu11 = getMu(1,1,image)
    mu30 = getMu(3,0,image)
    mu12 = getMu(1,2,image)
    mu21 = getMu(2,1,image)
    mu03 = getMu(0,3,image)

    phi1 = mu20 + mu02

    phi2 = (mu20 - mu02)**2 + 4 * (mu11 ** 2)
    
    phi3 = (mu30 - 3*mu12)**2 + (3*mu21 - mu03)**2

    phi4 = (mu30 + mu12)**2 + (mu21 + mu03)**2

    phi5 = (mu30-(3*mu12)) * (mu30 + mu12) * ((mu30 + mu12)**2 - (3*(mu21 + mu03)**2)) + \
         (3*mu21 - 3*mu03) * (mu21 + mu03) * (3*(mu30 + mu12)**2 - (mu21 + mu03)**2) 
            
    phi6 = (mu20 - mu02) * ((mu30 + mu12)**2 - (mu21 + mu03)**2) + \
            4*mu11 * (mu30 + mu12) * (mu21 + mu03)

    phi7 = ((3*mu21)-mu03) * (mu30 + mu12) * ((mu30 + mu12)**2 - (3*(mu21 + mu03)**2)) - \
            (mu30 - (3*mu12)) * (mu21 + mu03) * ((3*(mu30 + mu12)**2) - (mu21 + mu03)**2)

    return [phi1, phi2, phi3, phi4, phi5, phi6, phi7]


"""
Profiles:
    Return an array containing the location of the first black pixels for all 4 sides
    Then computes their derivative to find if the character is smooth or not, and if
    it is thick or not, and if it is tall or not.
"""
def profiles(image):

    raw_left = []
    raw_right = []
    raw_top = []
    raw_bottom = []

    for i in range(30):
        row = image[i]
        black = np.where(row == 0)

        if any(black[0]):
            raw_left.append(black[0][0])
            raw_right.append(black[0][-1])
        else:
            # raw does not contain any black pixels
            raw_left.append(39)
            raw_right.append(0)


    for i in range(40):
        column = image[:,i]
        black = np.where(column == 0)

        if any(black[0]):
            raw_top.append(black[0][0])
            raw_bottom.append(black[0][-1])
        else:
            raw_top.append(29)
            raw_bottom.append(0)

    df_left = np.gradient(raw_left)
    df_right = np.gradient(raw_right)
    df_top = np.gradient(raw_top)
    df_bottom = np.gradient(raw_bottom)

    hor_dist = []
    vert_dist = []

    for i in [0,4,9,14,19,24,29]:
        hor_dist.append(raw_right[i] - raw_left[i])

    for i in [0,4,9,14,19,24,29,34,39]:
        vert_dist.append(raw_bottom[i] - raw_top[i])

    hor_dist = [a/40 for a in hor_dist] 
    vert_dist = [a/30 for a in vert_dist]

    ret = np.concatenate((raw_left, raw_right, raw_top, raw_bottom, df_left, df_right, df_top, df_bottom, hor_dist, vert_dist))

    return ret


"""
Pixel Percentage:
    Returns the ratio of pixels in the top half vs bottom half of the image and in
    the left half vs the right half
"""
def per_Pixel(img):
    horizontal_half = img.shape[0] // 2
    vertical_half = img.shape[1] // 2
    non_zero = np.count_nonzero(img)
    
    
    per_pixels_above_horizontal = np.count_nonzero(img[:horizontal_half, :]) / non_zero * 100
    per_pixels_left = np.count_nonzero(img[:, :vertical_half]) / non_zero * 100
    
    
    return [per_pixels_above_horizontal, per_pixels_left]

"""
Intersections:
    Returns the number of intersection between the character (black pixels) and three pre-determined
    columns and three pre-determined row.
"""
def intersections(image):
    hor_indices = [4,14,24]
    ver_indices = [10,20,30]

    hor_inter = []
    ver_inter = []

    for i in hor_indices:
        row = image[i]
        black = np.where(row == 0)[0]
        intersections = 0

        if any(black):
            new_point = True

            for pix in row:
                if new_point and pix==0:  # the first black pixel in a sequence
                    intersections = intersections + 1
                    new_point = False
                
                if (not new_point) and pix != 0: # reaching the first white pixel after a sequence of black
                    new_point = True
         
        hor_inter.append(intersections)


    for i in ver_indices:
        column = image[:,i]     
        black = np.where(column == 0)[0]
        intersections = 0

        if any(black):
            new_point = True

            for pix in column:
                if new_point and pix==0:  # the first black pixel in a sequence
                    intersections = intersections + 1
                    new_point = False
                
                if (not new_point) and pix != 0: # reaching the first white pixel after a sequence of black
                    new_point = True
         
        ver_inter.append(intersections)
    ret = np.concatenate((hor_inter, ver_inter))
    return ret


def zoning(image):
    rows = image.shape[0]
    columns = image.shape[1]
    zone_height = int(rows/3)
    zone_width = int(columns/4)

    zone11 = image[np.ix_(np.arange(0, zone_height), np.arange(0, zone_width))]
    zone12 = image[np.ix_(np.arange(0, zone_height), np.arange(zone_width, 2*zone_width))]
    zone13 = image[np.ix_(np.arange(0, zone_height), np.arange(2*zone_width, 3*zone_width))]
    zone14 = image[np.ix_(np.arange(0, zone_height), np.arange(3*zone_width, 4*zone_width))]

    zone21 = image[np.ix_(np.arange(zone_height, 2*zone_height), np.arange(0, zone_width))]
    zone22 = image[np.ix_(np.arange(zone_height, 2*zone_height), np.arange(zone_width, 2*zone_width))]
    zone23 = image[np.ix_(np.arange(zone_height, 2*zone_height), np.arange(2*zone_width, 3*zone_width))]
    zone24 = image[np.ix_(np.arange(zone_height, 2*zone_height), np.arange(3*zone_width, 4*zone_width))]

    zone31 = image[np.ix_(np.arange(2*zone_height, 3*zone_height), np.arange(0, zone_width))]
    zone32 = image[np.ix_(np.arange(2*zone_height, 3*zone_height), np.arange(zone_width, 2*zone_width))]
    zone33 = image[np.ix_(np.arange(2*zone_height, 3*zone_height), np.arange(2*zone_width, 3*zone_width))]
    zone34 = image[np.ix_(np.arange(2*zone_height, 3*zone_height), np.arange(3*zone_width, 4*zone_width))]

    # black pixels in the whole image
    total_pix = len(np.where(image == 0)[0])
    
    # black pixels in each zone
    zones_pix = []
    zones_pix.append(len(np.where(zone11 == 0)[0]))
    zones_pix.append(len(np.where(zone12 == 0)[0]))
    zones_pix.append(len(np.where(zone13 == 0)[0]))
    zones_pix.append(len(np.where(zone14 == 0)[0]))
    zones_pix.append(len(np.where(zone21 == 0)[0]))
    zones_pix.append(len(np.where(zone22 == 0)[0]))
    zones_pix.append(len(np.where(zone23 == 0)[0]))
    zones_pix.append(len(np.where(zone24 == 0)[0]))
    zones_pix.append(len(np.where(zone31 == 0)[0]))
    zones_pix.append(len(np.where(zone32 == 0)[0]))
    zones_pix.append(len(np.where(zone33 == 0)[0]))
    zones_pix.append(len(np.where(zone34 == 0)[0]))

    # ratio of black pixel in each zone versus total black pixels in image
    zones_versus_whole = [x / total_pix for x in zones_pix]

    # percentage of black pixels in each individual zone
    per_zone_density = [y / (zone_height*zone_width) for y in zones_pix]

    ret = np.concatenate((zones_versus_whole, per_zone_density))

    return ret


def recursion(arr, im, x, y, fx, fy):
        imm = im[x:fx, y:fy]
        retw = 0
        reth = 0
        w = np.shape(imm)[0]
        h = np.shape(imm)[1]
        if np.size(imm)<10:
            return [0,0]  
        elif np.size(imm)==np.count_nonzero(imm): #only white
            return [0,0]  
        elif (np.size(imm)-np.count_nonzero(imm))==1: #one black
            return [0,0]
        else:
            h2 = int(h/2)
            w2 = int(w/2)
            retw = getRet(arr, imm, w, w2, h, h2, True)
            if retw<1:
                return [0,0]
            reth = getRet(arr, imm, h, h2, w, w2, False)  
            if reth<1:
                return [0,0]      
        reth += y 
        retw += x      
        # if np.shape(arr)[0]<45:         
        arr.append(recursion(arr, im, retw, y, fx, reth)) 
        # if np.shape(arr)[0]<45:
        arr.append(recursion(arr, im, x, reth, retw, fy)) 
        # if np.shape(arr)[0]<45: 
        arr.append(recursion(arr, im, retw, reth, fx, fy))
        # if np.shape(arr)[0]<45: 
        arr.append(recursion(arr, im, x, y, retw, reth)) 
        return [reth, retw]

def getRet(arr, imm, a, a2, b, b2, wh):
    ret = 0
    # if np.shape(arr)[0]>=45:
    #     return 0
    if a>2:
        r = 0
        addOrsub = 5 #0 add, 1 sub, 5 nothing 
        count=0
        while True:
            if(wh):
                im1 = imm[0:a2, 0:b]
                im2 = imm[a2:a, 0:b]
            else:
                im1 = imm[0:b, 0:a2]
                im2 = imm[0:b, a2:a]
            s1 = np.size(im1)-np.count_nonzero(im1)
            s2 = np.size(im2)-np.count_nonzero(im2)
            if s1==0 and s2==0:
                return 0  
            if s1==s2:
                return a2
            else:
                if s1<s2:
                    r1 = s1/s2
                    if r<=r1:
                        r=r1
                        count+=1
                        a2+=1
                        addOrsub = 1
                    else:
                        count=0
                        if addOrsub==1:
                            addOrsub = 5
                            return a2-1
                        elif addOrsub==0:
                            addOrsub = 5
                            return a2+1
                else:  
                    r1 = s2/s1
                    if r<=r1:
                        r=r1
                        a2-=1
                        count+=1
                        addOrsub = 0
                    else:
                        count=0
                        if addOrsub==1:
                            addOrsub = 5
                            return a2-1
                        elif addOrsub==0:
                            addOrsub = 5
                            return a2+1
                if count>=a:
                    count = 0
                    if addOrsub==1:
                        addOrsub = 5
                        return a2-1
                    elif addOrsub==0:
                        addOrsub = 5
                        return a2+1
    else:
        return 0 


def divisionPoints(im, x0, y0, prev, dp, level, max_level):

    #plt.imshow(im)

    if level == max_level:
        return

    else:
        rows = np.shape(im)[0]
        columns = np.shape(im)[1]
        
        pix_hor = []
        pix_ver = []

        for c in range(columns):
            column = im[:,c]
            pix_hor.append(len(np.where(column == 0)[0]))
            
        for r in range(rows):
            row = im[r,:]
            pix_ver.append(len(np.where(row == 0)[0]))

        if not(np.any(pix_ver)) and not (np.any(pix_hor)): #white sub-image\
            center = prev
            dp.append(center[0])
            dp.append(center[1])
            divisionPoints(im, x0, y0, center, dp, level+1, max_level)
            divisionPoints(im, x0, y0, center, dp, level+1, max_level)
            divisionPoints(im, x0, y0, center, dp, level+1, max_level)
            divisionPoints(im, x0, y0, center, dp, level+1, max_level)


        else:
            x_mid = arrayMiddle(pix_ver)
            y_mid = arrayMiddle(pix_hor)

            im11 = im[0:x_mid, 0:y_mid]
            im12 = im[0:x_mid, y_mid:]
            im21 = im[x_mid:, 0:y_mid]
            im22 = im[x_mid:, y_mid:]

            center = [x_mid + x0, y_mid + y0]  
            dp.append(center[0])
            dp.append(center[1])
            divisionPoints(im11, x0, y0, center, dp, level+1, max_level)
            divisionPoints(im12, x0, y_mid + y0, center, dp, level+1, max_level)
            divisionPoints(im21, x_mid + x0, y0, center, dp, level+1, max_level)
            divisionPoints(im22, x_mid + x0, y_mid + y0, center, dp, level+1, max_level)

    return dp


def arrayMiddle(arr):
    start = 0
    end = len(arr)
    mid = int(end/2)
    
    left = sum(arr[start:mid])
    right = sum(arr[mid:end])
    
    if left == right:
        return mid
    elif left < right:
        ratio = left/right
    else:
        ratio = right/left
 
    while True: 
        if left<right:
            new_mid = mid+1
        else:
            new_mid = mid-1
        
        new_left = sum(arr[start:new_mid])
        new_right = sum(arr[new_mid:end])
        
        if new_left == new_right:
            return mid
        elif new_left < new_right:
            new_ratio = new_left/new_right
        else:
            new_ratio = new_right/new_left 
            
        if new_ratio <= ratio:
            return mid
        else:
            left = new_left
            right = new_right
            mid = new_mid
            ratio = new_ratio


"""
im = get_bounding_box('.', 'm.png', '.')
arr = []
arr = divisionPoints(im,0,0,[0,0],arr,0,6)
print(arr)
arr = np.array(arr)
x = arr[:,0] * -1
y = arr[:,1] 

plt.scatter(y,x, marker="o", color="black", s=5)
plt.show()"""
