import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import base64
def docFind(imgData):
    bimg = base64.b64decode(imgData); 
    npimg = np.fromstring(bimg, dtype=np.uint8); 
    img = cv2.imdecode(npimg, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 2)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 150)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))


    # Blank canvas.
    con = np.zeros_like(img)
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)



    closing = cv2.morphologyEx(con, cv2.MORPH_CLOSE, (25,25),iterations=5)
    # plt.imshow(closing, cmap = "binary")
    # plt.show()

    # Blank canvas.
    con = np.zeros_like(img)
    # Loop over the contours.
    for c in page:
    # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
    # If our approximated contour has four points
        if len(corners) == 4:
            break
    if(cv2.contourArea(c) > (img.shape[0]*img.shape[1])*0.3):
        return True
    else:
        return False

def warpDocument(imgData):

    newBound = [[0,0],[1000,0],[0,500],[1000,500]]

    bimg = base64.b64decode(imgData); 
    npimg = np.fromstring(bimg, dtype=np.uint8); 
    img = cv2.imdecode(npimg, 1)
    # img = cv2.imread("./goodScan.jpg")


    # plt.imshow(img)
    # plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 2)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 150)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))

    # Blank canvas.
    con = np.zeros_like(img)
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

    closing = cv2.morphologyEx(con, cv2.MORPH_CLOSE, (25,25),iterations=5)
    # plt.imshow(closing, cmap = "binary")
    # plt.show()

    # Blank canvas.
    con = np.zeros_like(img)
    # Loop over the contours.
    for c in page:
    # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
    # If our approximated contour has four points
        if len(corners) == 4:
            break
    cv2.drawContours(con, c, -1, (0, 255, 255), 3)
    cv2.drawContours(con, corners, -1, (0, 255, 0), 10)

    # plt.imshow(con)
    # plt.show()
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())


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

    # print (pts)
    # print(destination_corners)
    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(destination_corners))
    # Perspective transform using homography.
    fimg = cv2.warpPerspective(img, M, (maxWidth,maxHeight), flags=cv2.INTER_LINEAR)

    _, im_arr = cv2.imencode('.png', fimg)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = str(base64.b64encode(im_bytes)).split("'")[1]

    return im_b64

#TODO fix warpDocument
def scanData(fimg):


    plt.imshow(fimg)
    plt.show()


    #transform the tilted picture to the straightened rider
    # M = cv2.getPerspectiveTransform(np.float32(corners),np.float32(newBound))
    # fimg = cv2.warpPerspective(img,M,(500,1000))
    # plt.imshow(fimg)
    # plt.show()

    gimg = cv2.cvtColor(fimg.copy(),cv2.COLOR_BGR2GRAY)

    ret,th = cv2.threshold(gimg, 150, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    detected_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, vertical_kernel, iterations=10)
    # temp = cv2.erode(th, vertical_kernel, iterations=1)
    detected_lines = cv2.dilate(detected_lines, (1,25), iterations=5)
    #
    #remove lines that are straight and long
    cnts, hierarchy = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plt.imshow(detected_lines)
    plt.show()

    section = [0]
    for c in cnts:
        #print(cv2.arcLength(c,False))
        #print(cv2.contourArea(c))
        if(cv2.contourArea(c) < maxHeight*8):
            section.append(c[0,0,0])

    section = sorted(section)

    # cnts, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for c in cnts:
    #     if cv2.arcLength(c,False) > 0:
    #         cv2.drawContours(temp,[c],-1,(0,0,0),2)

    # plt.imshow(temp)
    # plt.show()

    #turns grays into either fully white or fully black

    #detect write-on lines
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    # detected_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # #remove lines that are straight and long
    # cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(th, [c], -1, (0,0,0), 2)
    # #readd part of line if a number was inside it
    # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    # result = cv2.morphologyEx(th, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    #top left point of boxes for section A of rider form

    for t in range(len(section)-1):

        pts = [ [section[t],0],[section[t],maxHeight],[section[t+1],maxHeight],[section[t+1],0] ]

        newBound = [ [0,0],[0,2000],[1000,2000],[1000,0] ]

        M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(newBound))
        # Perspective transform using homography.
        finalimg = cv2.warpPerspective(fimg, M, (1000,2000), flags=cv2.INTER_LINEAR)
        plt.imshow(finalimg)
        plt.show()

        # timg = fimg[0:maxHeight,0:section[0]]
        # plt.imshow(timg)
        # plt.show()

        coordinates = [[750,620],[750,655],[750,710],[670,1140],[670,1190]]
        boxSize = [200,50]

        res = cv2.cvtColor(finalimg,cv2.COLOR_BGR2GRAY)
        res = np.invert(res)
        ret,th = cv2.threshold(res, 50, 0, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)


        i = 0
        #goes box by box
        for p in coordinates:
            mask = np.zeros(th.shape[:2],dtype=th.dtype)
            r = np.array([[p[0],p[1]],[p[0]+boxSize[0],p[1]],[p[0]+boxSize[0],p[1]+boxSize[1]],[p[0],p[1]+boxSize[1]]])
            cv2.rectangle(mask,r[0],r[2],(255,255,255),-1)
            #save to temp image and leave only data that is inside
            temp = cv2.bitwise_and(th,th,mask=mask)
            #uses contours with external fill to seperate digits 
            contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sortcontours = sorted(contours, key=lambda c:cv2.boundingRect(c)[0])

            #finds n many numbers inside of box
            for c in sortcontours:
                if cv2.contourArea(c) > 200:
                    x,y,w,h = cv2.boundingRect(c)
                    finalimg = th[y:y+h,x:x+w]
                    finalimg = np.pad(finalimg,pad_width=5,mode='constant',constant_values=0)
                    #finalimg = np.invert(finalimg)
                    #makes image MNIST size
                    finalimg = cv2.resize(finalimg,(28,28))
                    # plt.imshow(finalimg , cmap = 'binary')
                    # plt.show()
                    #invert because otherwise text is white and background is black
                    
                    # cv2.imwrite('./generated/img'+str(i)+'.png',finalimg)
                    i+=1