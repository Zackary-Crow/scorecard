
#########
import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import combinations
import os
import base64
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.neuralNetwork import numeralRecognition

# constants
xval = 400
model = numeralRecognition.CNN()
model.load_state_dict(torch.load("src/neuralNetwork/scorecardCNN.pt",map_location=torch.device('cpu')))

# resize image to height 2048
def resizeImage(img):
        if img.shape[0] <= 2048:
              return img
        
        width = int(img.shape[1] * round(2048 / img.shape[0], 3))
        dim = (width, 2048)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        
        return img

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()

def findCorners(img):
    # threshold black and white
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Create black and white image based on adaptive threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

    # Median filter clears small details
    img = cv2.medianBlur(img, 11)

    # Add black border in case that page is touching an image border
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    edges = cv2.Canny(img, 200, 250)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    con = np.zeros_like(img)
    # Loop over the contours.
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points
        if len(corners) == 4:
            return c
            break
    # cv2.drawContours(con, c, -1, (0, 255, 255), 3)
    # cv2.drawContours(con, corners, -1, (0, 255, 0), 10)

    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv2.circle(img,(int(x),int(y)),50,(36,255,12),-1)

def straightenImage(img, c):
    epsilon = 0.02 * cv2.arcLength(c, True)
    corners = cv2.approxPolyDP(c, epsilon, True)
    x1, y1 = corners[1].ravel()
    x2, y2 = corners[2].ravel()

    angle = math.atan2( y2 - y1, x2 - x1 ) * ( 180 / math.pi )

    img = rotateImage(img, angle)

    # rotate corners by calculated angle about center of image
    for corner in corners:
        x = corner[0][0] - img.shape[1] / 2
        y = corner[0][1] - img.shape[0] / 2

        corner[0][0] = x * math.cos(math.radians(-angle)) - y * math.sin(math.radians(-angle)) + img.shape[1] / 2
        corner[0][1] = y * math.cos(math.radians(-angle)) + x * math.sin(math.radians(-angle)) + img.shape[0] / 2
        # cv2.circle(img,(int(corner[0][0]),int(corner[0][1])),50,(36,255,12),-1)
    return img, corners

def centerImage(img, corners):
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())

    pts = order_points(corners)

    (tl, tr, br, bl) = pts

    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(destination_corners))
    # Perspective transform using homography.
    fimg = cv2.warpPerspective(img, M, (maxWidth,maxHeight), flags=cv2.INTER_LINEAR)

    return fimg, maxHeight

def findContours(img):
    gimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    #ret,th = cv2.threshold(gimg, 150, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret,th = cv2.threshold(gimg, 150, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    detected_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, vertical_kernel, iterations=10)
    # temp = cv2.erode(th, vertical_kernel, iterations=1)
    # detected_lines = cv2.dilate(detected_lines, vertical_kernel, iterations=5)
    closing = cv2.morphologyEx(detected_lines, cv2.MORPH_CLOSE, vertical_kernel, iterations=10)

    # plt.imshow(closing)
    # plt.show()

    #remove lines that are straight and long
    cnts, hierarchy = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)

    return img, cnts

def findSections(img, cnts, maxHeight):
    section = []
    for c in cnts:
        if(cv2.arcLength(c,True) > maxHeight*.5):
            section.append(c[0,0,0])

    section = sorted(section)
    section.append(img.shape[1])

    return img, section

def zeroRowCount(img):
    count = 0

    for i in range (28):
        for j in range (28):
            if img[i][j] != 0:
                break
            if j == 27:
                count += 1

    return count

def computeSection(left,right,maxHeight,img):

    pts = [ [left,0],[left,maxHeight],[right,maxHeight],[right,0] ]
    newBound = [ [0,0],[0,2000],[1000,2000],[1000,0] ]
    M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(newBound))
    # Perspective transform using homography.
    sectionImg = cv2.warpPerspective(img, M, (1000,2000), flags=cv2.INTER_LINEAR)
    # plt.imshow(sectionImg)
    # plt.show()
    filterImg = np.copy(sectionImg)
    graySection = cv2.cvtColor(filterImg,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(graySection, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(filterImg, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    filterImg = np.invert(filterImg)
    result = cv2.morphologyEx(filterImg, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    result = np.invert(result)

    res = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    _,filterth = cv2.threshold(res, 200, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO_INV)
    gray = cv2.cvtColor(sectionImg,cv2.COLOR_BGR2GRAY)
    _,sectionth = cv2.threshold(gray, 200, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO_INV)

    prossImg = filterth[np.r_[600:900,1100:1200,1500:1600],xval:950]
    actualImg = sectionth[np.r_[600:900,1100:1200,1500:1600],xval:950]

    contours, hierarchy = cv2.findContours(prossImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list = []
    grouped = [None]
    sortcontours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    for c in contours:
        if( cv2.contourArea(c) > cv2.arcLength(c,True)):
            x,y,w,h = cv2.boundingRect(c)
            list.append((c,y,y+h))
    # print(f"list size {len(list)}")
    index = 0
    # for l in list:
        # print(str(l[1])+" "+str(l[2]))
    for i, j in enumerate(list[:-1]):
        if grouped[index] == None:
            grouped[index] = [j[0]]
        if (j[1] <= list[i+1][2] and j[1] >= list[i+1][1]) or (j[2] <= list[i+1][2] and j[2] >= list[i+1][1]):
            grouped[index].append(list[i+1][0])
            # print(f"{list[i+1][1]},{list[i+1][2]} added to {j[1]},{j[2]} with index {index}")
        else:
            grouped.append(None)
            index += 1
    if grouped[index] == None:
        grouped[index] = [list[-1][0]]
    grouped.reverse()
    return grouped,actualImg

def findNumbers(group,img):
#finds n many numbers inside of box
    group = sorted(group, key=lambda c: cv2.boundingRect(c)[0])
    group.reverse()
    # print(f"group size {len(group)}")
    # print(cv2.boundingRect(group[0])[0])
    if( cv2.boundingRect(group[0])[0] < (600-xval)):
        return            
    val = len(group)
    for i, j in enumerate(group[:-1]):
        # print(f"{cv2.boundingRect(j)[0]} - {cv2.boundingRect(group[i+1])[0] + cv2.boundingRect(group[i+1])[2]} = {cv2.boundingRect(j)[0] - (cv2.boundingRect(group[i+1])[0] + cv2.boundingRect(group[i+1])[2])}")
        if (cv2.boundingRect(j)[0] - (cv2.boundingRect(group[i+1])[0] + cv2.boundingRect(group[i+1])[2])) > 30:
            val = i+1
            break
    # print(f"val {val} group size {len(group)}")
    group = group[0:val]
    group.reverse()
    i=0
    groupArr = []
    for c in group:
        # print(f"contour area: {cv2.contourArea(c)}\ncontour perimeter: {cv2.arcLength(c,True)}\nheight: {cv2.boundingRect(c)[3]}")
        if cv2.contourArea(c) < cv2.arcLength(c,True) or (cv2.contourArea(c) <= 200 and cv2.boundingRect(c)[3] < cv2.arcLength(c,True)/3):
            continue
        x,y,w,h = cv2.boundingRect(c)
        finalimg = img[y:y+h,x:x+w]
        finalimg = np.pad(finalimg,pad_width=5,mode='constant',constant_values=0)
        #finalimg = np.invert(finalimg)
        # print(f"contour area: {cv2.contourArea(c)}\ncontour perimeter: {cv2.arcLength(c,True)}")
        # plt.imshow(finalimg)
        # plt.show()
        #makes image MNIST size
        finalimg = cv2.resize(finalimg,(28,28))
        finalimg = cv2.erode(finalimg, np.ones((2, 2), np.uint8), iterations=1)
        (thresh, finalimg) = cv2.threshold(finalimg, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if zeroRowCount(finalimg) > 21:
            continue
        groupArr.append(finalimg)
    if(len(groupArr) == 0):
        return
    return groupArr

def makePrediction(img,model):
    transform = transforms.ToTensor()
    # Transform
    input = transform(img)
    # unsqueeze batch dimension, in case you are dealing with a single image
    input = input.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(input)
    pred = torch.argmax(output, 1)
    return pred.item()

def docFind(imgData):
    bimg = base64.b64decode(imgData); 
    npimg = np.fromstring(bimg, dtype=np.uint8); 
    img = cv2.imdecode(npimg, 1)

    img = resizeImage(img)
    c = findCorners(img)
    # print(c.dtype)
    if c is not None:
        if(cv2.contourArea(c) > (img.shape[0]*img.shape[1])*0.3):
            return True, c , img
    return False, c, img

###########################################

# img = cv2.imread("src/python/darkLandscapeScan.jpg")
def fullProcess(img):

    img = resizeImage(img)
    # plt.imshow(img)
    # plt.show()
    c = findCorners(img)
    # plt.imshow(img)
    # plt.show()
    img, corners = straightenImage(img, c)
    # plt.imshow(img)
    # plt.show()
    img, maxHeight = centerImage(img, corners)
    # plt.imshow(img)
    # plt.show()
    img, cnts = findContours(img)
    # plt.imshow(img)
    # plt.show()
    img, section = findSections(img, cnts,maxHeight)
    # plt.imshow(img)
    # plt.show()
    predictions = []
    riderArr = []
    for i,j in enumerate(section[:-1]):
        print(f"section {i}")
        groups, sectionImg = computeSection(section[i],section[i+1],maxHeight,img)
        groupArr = []
        for k,group in enumerate(groups):
            arr = findNumbers(group,sectionImg)
            if arr == None:
                continue
            print(f"group {k}")
            item = []
            for digit in arr:
                val = makePrediction(digit,model)
                item.append(val)
            if len(item) == 2:
                groupArr.append(item[0]+(item[1]/10))
            else:
                groupArr.append(sum(d * 10**i for i, d in enumerate(item[::-1])))
            print(groupArr)
        riderArr.append(groupArr)
        
    return riderArr
        


###########################################

    # plt.imshow(img)
    # plt.show()
