
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

def padToSquare(img):
    (x, y) = img.shape
    if x > y:
        padding = int((x - y) / 2)
        return cv2.copyMakeBorder(img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, None, value = 0)
    else:
        padding = int((y - x) / 2)
        return cv2.copyMakeBorder(img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
# resize image to height 2048
def resizeImage(img):
        if img.shape[0] <= 1024:
              return img
        
        width = int(img.shape[1] * round(1024 / img.shape[0], 3))
        dim = (width, 1024)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        
        return img

def deNoise(img):
    return cv2.fastNlMeansDenoising(img, h=7)

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

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = {}
    for i, line in enumerate(lines):
        label = labels[i]
        if label not in segmented:
            segmented[label] = []
        segmented[label].append(line)
    return list(segmented.values())

def findCorners(img):
    orgimg = img.copy()
    timg = img.copy()
    # threshold black and white
    img = cv2.medianBlur(img, 21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 10)

    # plt.imshow(closed)
    # plt.show()

    conEdge, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    edgeimg = np.zeros_like(closed)
    #edgeimg = cv2.cvtColor(edgeimg, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(edgeimg, conEdge, -1, 255, 3) 
    
    # plt.imshow(edgeimg)
    # plt.show()
    dst = edgeimg.copy()

    lines = cv2.HoughLines(dst, 2, np.pi / 360, int(img.shape[0]/4))
    
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        n = 1000
        x1 = int(x0 + n * (-b))
        y1 = int(y0 + n * (a))
        x2 = int(x0 - n * (-b))
        y2 = int(y0 - n * (a))

        cv2.line(timg, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    
    # plt.imshow(cdst)
    # plt.show()

    segmented = segment_by_angle_kmeans(lines)
    def intersection(line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]


    def segmented_intersections(lines):
        """Finds the intersections between groups of lines."""

        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i+1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(intersection(line1, line2)) 

        return intersections
    
    intersections = segmented_intersections(segmented)

    pts = np.array(intersections)[:,0]
    pts = np.vstack(pts[:])
    pts = np.float32(pts)


    # run kmeans on the coords
    _, labels, centers = cv2.kmeans(pts, 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    #labels = labels.reshape(-1)  # transpose to row vec
    
    centers = centers.astype(int)

    for c in centers:
        cv2.circle(timg,c,25,(0,255,0),-1)
    
    return orgimg, centers, timg


def straightenImage(img, corners):
    corners = sorted(np.concatenate(corners).tolist())
    corners = order_points(corners)
    x1, y1 = corners[3]
    x2, y2 = corners[2]

    angle = math.atan2( y2 - y1, x2 - x1 ) * ( 180 / math.pi )

    img = rotateImage(img, angle)

    # rotate corners by calculated angle about center of image
    for corner in corners:
        x = corner[0] - img.shape[1] / 2
        y = corner[1] - img.shape[0] / 2

        corner[0] = x * math.cos(math.radians(-angle)) - y * math.sin(math.radians(-angle)) + img.shape[1] / 2
        corner[1] = y * math.cos(math.radians(-angle)) + x * math.sin(math.radians(-angle)) + img.shape[0] / 2
        cv2.circle(img,(int(corner[0]),int(corner[1])),50,(36,255,12),-1)
    return img, corners


def centerImage(img, corners):
    # Sorting the corners and converting them to desired shape.
    corners = order_points(corners)

    pts = corners

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
    b = .1
    x,y = gimg.shape[0],gimg.shape[1]
    bor_x,bor_y = int(x*b),int(y*b)
    ret,th = cv2.threshold(gimg, 150, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    th = th[bor_x:x-bor_x,bor_y:y-bor_y]
    th = np.pad(th,pad_width=((bor_x,bor_x),(bor_y,bor_y)),mode='constant',constant_values=0)
    
    #ret,th = cv2.threshold(gimg, 150, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    

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

def findSections(img, cnts):
    width = img.shape[1]
    maxHeight = img.shape[0]
    
    expected_sections = [0.214 * width, 0.395 * width, 0.576 * width, 0.758 * width]
    section = [0]
    for c in cnts:
        if(cv2.arcLength(c,True) > maxHeight*.5 and c[0,0,0] != 0):
            section_candidate = c[0,0,0]
            for i, ex in enumerate(expected_sections):
                if (section_candidate < ex + 30 and section_candidate > ex - 30):
                    section.append(section_candidate)
                    break
            if (len(expected_sections) == 0):
                break
    
    section = sorted(section)
    expected_sections = [0, 0.214 * width, 0.395 * width, 0.576 * width, 0.758 * width]

    if (len(section) != 5):
        for i, ex in enumerate(expected_sections):
            if (i < len(section)):
                if (abs(section[i] - ex) >= 30):
                    section.insert(i, ex)
            else:
                section.insert(i, ex)


    section = section[:5]
    section.append(section[-1] + (section[4] - section[1]) / 3)

    #section.append(maxWidth)

    return img, section

# def findSections(img, cnts, maxHeight):
#     section = [0]
#     for c in cnts:
#         if(cv2.arcLength(c,True) > maxHeight*.5):
#             section.append(c[0,0,0])

#     section = sorted(section)
#     section.append(img.shape[1])

#     return img, section

def zeroRowCount(img):
    count = 0

    for i in range (28):
        for j in range (28):
            if img[i][j] != 0:
                break
            if j == 27:
                count += 1

    return count

def computeSectionBlue(left,right,maxHeight,img):

    originImg = img.copy()

    pts = [ [left,0],[left,maxHeight],[right,maxHeight],[right,0] ]
    newBound = [ [0,0],[0,2000],[1000,2000],[1000,0] ]
    M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(newBound))
    # Perspective transform using homography.
    sectionImg = cv2.warpPerspective(img, M, (1000,2000), flags=cv2.INTER_LINEAR)
    # plt.imshow(sectionImg)
    # plt.show()
    filterImg = np.copy(sectionImg)

    hsv = cv2.cvtColor(filterImg, cv2.COLOR_RGB2HSV)
    # plt.imshow(hsv)
    # plt.show()
    # define range of blue color in HSV
    lower_blue = np.array([100,5,90])
    upper_blue = np.array([150,100,170])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    filter_output = cv2.bitwise_and(filterImg,filterImg, mask= mask)

    # plt.imshow(hsv)
    # plt.show()


    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(filterImg, [c], -1, (255,255,255), 2)

    # plt.imshow(filterImg)
    # plt.show()

    # Repair image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morphOP = filter_output.copy()
    # morphOP = cv2.morphologyEx(morphOP, cv2.MORPH_OPEN, kernel)
    morphOP = cv2.morphologyEx(morphOP, cv2.MORPH_CLOSE, kernel, iterations=2)

    res = cv2.cvtColor(morphOP,cv2.COLOR_BGR2GRAY)
    _,filterth = cv2.threshold(res, 200, 255, cv2.THRESH_OTSU)
    # gray = cv2.cvtColor(sectionImg,cv2.COLOR_BGR2GRAY)
    # _,sectionth = cv2.threshold(gray, 200, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO_INV)

    prossImg = filterth[np.r_[600:900,1100:1200,1500:1600],xval:950]
    # actualImg = sectionth[np.r_[600:900,1100:1200,1500:1600],xval:950]
    

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
    if grouped[index] == None and list != None:
        grouped[index] = [list[-1][0]]
    grouped.reverse()
    return grouped,prossImg

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
    filter_output = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    

    cnts = cv2.findContours(filter_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    if grouped[index] == None and list != None:
        grouped[index] = [list[-1][0]]
    grouped.reverse()
    return grouped,actualImg

def findNumbers(glist,img):
#finds n many numbers inside of box
    fullSize = []
    width = img.shape[1]
    height = img.shape[0]
    glist = sorted(glist, key=lambda c: cv2.boundingRect(c)[0])
    glist.reverse()
    #print(cv2.boundingRect(glist[0])[0])
    #plt.imshow(img)
    #plt.show()
    if( cv2.boundingRect(glist[0])[0] < (600-xval)):
        return None, None
    val = len(glist)

    #print(cv2.boundingRect(glist[val-1])[0])
    #print(cv2.boundingRect(glist[val-1])[1])

    for i, j in enumerate(glist[:-1]):
        #print(cv2.boundingRect(glist[0])[0])
        #print(f"{cv2.boundingRect(j)[0]} - {cv2.boundingRect(glist[i+1])[0] + cv2.boundingRect(glist[i+1])[2]} = {cv2.boundingRect(j)[0] - (cv2.boundingRect(glist[i+1])[0] + cv2.boundingRect(glist[i+1])[2])}")
        #if (cv2.boundingRect(j)[0] - (cv2.boundingRect(glist[i+1])[0] + cv2.boundingRect(glist[i+1])[2])) > 30:
            #val = i+1
            #break
        
        if (cv2.boundingRect(j)[1] < height * 0.36):
            if (cv2.boundingRect(j)[0] < width * 0.61):
                val = i+1
                break
        elif (cv2.boundingRect(j)[1] < height * 0.60):
            if (cv2.boundingRect(j)[0] < width * 0.63):
                val = i+1
                break
        else:
            if (cv2.boundingRect(j)[0] < width * 0.47):
                val = i+1
                break

        #print(cv2.boundingRect(j)[0])
        #print(cv2.boundingRect(j)[1])
        #print()

    #print(f"val {val} glist size {len(glist)}")
    glist = glist[0:val]
    glist.reverse()
    i=0
    groupArr = []
    for c in glist:
        #print(f"contour area: {cv2.contourArea(c)}\ncontour perimeter: {cv2.arcLength(c,True)}\nheight: {cv2.boundingRect(c)[3]}")
        if cv2.contourArea(c) < cv2.arcLength(c,True) or (cv2.contourArea(c) <= 200 and cv2.boundingRect(c)[3] < cv2.arcLength(c,True)/3):
            continue
        x,y,w,h = cv2.boundingRect(c)
        finalimg = img[y:y+h,x:x+w]
        
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        #print(cx)
        #print(cy)
        #print("\n")
        if (cy < height * 0.36):
            if (cx < width * 0.61):
                continue
        elif (cy < height * 0.60):
            if (cx < width * 0.63):
                continue
        else:
            if (cx < width * 0.47):
                continue
        
        finalimg = np.pad(finalimg,pad_width=5,mode='constant',constant_values=0)

        finalimg = padToSquare(finalimg)
        
        fullSize.append(finalimg)
        #finalimg = np.invert(finalimg)
        # print(f"contour area: {cv2.contourArea(c)}\ncontour perimeter: {cv2.arcLength(c,True)}")

        #makes image MNIST size
        finalimg = cv2.resize(finalimg,(28,28))
        #plt.imshow(finalimg)
        #plt.show()
        finalimg = cv2.erode(finalimg, np.ones((2, 2), np.uint8), iterations=1)
        #plt.imshow(finalimg)
        #plt.show()
        (thresh, finalimg) = cv2.threshold(finalimg, 1, 255, cv2.THRESH_BINARY)
        #plt.imshow(finalimg)
        #plt.show()
        if zeroRowCount(finalimg) > 21:
            continue
        #plt.imshow(finalimg)
        #plt.show()
        groupArr.append(finalimg)
    if len(groupArr) == 0:
        return None, None
    return groupArr, fullSize
            # cv2.imwrite('python/generated/img'+str(i)+'.png',finalimg)
            # i+=1

# def findNumbers(group,img):
# #finds n many numbers inside of box



def makePrediction(img,model):
    transform = transforms.ToTensor()
    # Transform
    input = transform(img)
    

    input = input.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(input)
    val = torch.argmax(output)
    prob = torch.nn.functional.softmax(output, 1)
    conf,_ = torch.max(prob,1)
    return conf.item()*100, val.item()
###########################################

# img = cv2.imread("src/python/darkLandscapeScan.jpg")
def fullProcess(img,blueink=False):

    debugImg = []
    riderArr = []
    predictions = []
    try:
        img = resizeImage(img)
        img = deNoise(img)
        debugImg.append(img)
        # plt.imshow(img)
        # plt.show()
        img, corners, dimg = findCorners(img)
        debugImg.append(dimg)
        # plt.imshow(img)
        # plt.show()
        #img, corners = straightenImage(img, c)
        #debugImg.append(img)
        # plt.imshow(img)
        # plt.show()
        img, maxHeight = centerImage(img, corners)
        debugImg.append(img)
        # plt.imshow(img)
        # plt.show()
        img, cnts = findContours(img)
        debugImg.append(img)
        # plt.imshow(img)
        # plt.show()
        img, section = findSections(img, cnts)
        debugImg.append(img)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img)
        # plt.show()
        
        for i,j in enumerate(section[:-1]):
            # print(f"section {i}")
            if(blueink):
                groups, sectionImg = computeSectionBlue(section[i],section[i+1],maxHeight,img)
            else:
                groups, sectionImg = computeSection(section[i],section[i+1],maxHeight,img)

            debugImg.append(sectionImg)
            # plt.imshow(sectionImg)
            # plt.show()
            groupArr = []
            for k,group in enumerate(groups):
                arr,fullDebug = findNumbers(group,sectionImg)
                if arr == None:
                    continue
                item = []
                for digit, fullDigit in zip(arr,fullDebug):
                    debugImg.append(fullDigit)
                    conf, val = makePrediction(digit,model)
                    print(val, conf)
                    item.append([val,conf])
                if len(item) > 3:
                    continue
                if len(item) == 2:
                    if(item[0][0] == 1 and item[1][0] == 0):
                        groupArr.append({"value":10,"confidence":min(item[0][1],item[1][1])})
                    else:    
                        groupArr.append({"value":item[0][0]+(item[1][0]/10),"confidence":min(item[0][1],item[1][1])})
                else:
                    groupArr.append({"value":sum(d[0] * 10**i for i, d in enumerate(item[::-1])),"confidence":min(d[1] for d in item)})
                print(groupArr[-1])
                # print(groupArr)
            riderArr.append(groupArr)
        
        riderArr = fillArr(riderArr.copy())
        print(riderArr)
        return riderArr, debugImg
    except Exception as e:
        print(e)
        riderArr = fillArr(riderArr.copy())
        return riderArr, debugImg
            
def fillArr(temp): 
    for group in temp:
        if(len(group) > 7):
            break
        while(len(group) <= 7):
            group.append({"value":'',"confidence":100})

    while(len(temp) < 5):
        gTemp = []
        for i in range(7):
            gTemp.append({"value":'',"confidence":100})
        temp.append(gTemp)

    return temp

    ###########################################

        # plt.imshow(img)
        # plt.show()
