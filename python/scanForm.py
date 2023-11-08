import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math

#manual points for first rider
#corners = [[1200,2600],[1200,1660],[4000,2600],[4000,1660]]
#straighted allignment 
newBound = [[0,0],[1000,0],[0,500],[1000,500]]

xval = 400

#read in image as grayscale
#img = cv2.imread("testingDocument.png")
#img = cv2.imread("tracedSample1.jpg")
#img = cv2.imread("BlackAdditionSample.png")
img = cv2.imread("darkLandscapeScan.jpg")
# plt.imshow(img)
# plt.show()


#TODO: Straighten entire document
# Repeated Closing operation to remove text from the document.
#kernel = np.ones((11,11),np.uint8)
#img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

# plt.imshow(img)
# plt.show()

# gray_img = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
# blur_img = cv2.GaussianBlur(gray_img,(11,11),2)
# edged_img = cv2.Canny(blur_img,50,200)

# plt.imshow(edged_img)
# plt.show()


# cnts,_ = cv2.findContours(edged_img.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
# for c in cnts:
#     peri = cv2.arcLength(c,True)
#     approx = cv2.approxPolyDP(c,0.02*peri,True)
#     if len(approx) == 4: 
#       doc = approx
#       break

# p=[]
# for d in doc:
#     tuple_point = tuple(d[0])
#     #cv2.circle(img,tuple_point,3,(0,0,255),4)
#     p.append(tuple_point)

# corners = p
gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# plt.imshow(th)
# plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
gray = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel,iterations=10)


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

foundCorners = np.copy(img)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(foundCorners,(int(x),int(y)),50,(36,255,12),-1)

# plt.imshow(foundCorners)
# plt.show()



# plt.imshow(img)
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

def zeroRowCount(img):
    count = 0

    for i in range (28):
        for j in range (28):
            if img[i][j] != 0:
                break
            if j == 27:
                count += 1

    return count

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

# plt.imshow(fimg)
# plt.show()

# plt.imshow(fimg)
# plt.show()
#transform the tilted picture to the straightened rider
# M = cv2.getPerspectiveTransform(np.float32(corners),np.float32(newBound))
# fimg = cv2.warpPerspective(img,M,(500,1000))
# plt.imshow(fimg)
# plt.show()

gimg = cv2.cvtColor(fimg.copy(),cv2.COLOR_BGR2GRAY)

ret,th = cv2.threshold(gimg, 150, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
detected_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, vertical_kernel, iterations=10)
# temp = cv2.erode(th, vertical_kernel, iterations=1)
# detected_lines = cv2.dilate(detected_lines, vertical_kernel, iterations=5)
closing = cv2.morphologyEx(detected_lines, cv2.MORPH_CLOSE, vertical_kernel, iterations=10)

plt.imshow(closing)
plt.show()

#remove lines that are straight and long
cnts, hierarchy = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# plt.imshow(detected_lines)
# plt.show()

section = [0]
for c in cnts:
    #print(cv2.arcLength(c,False))
    #print(cv2.contourArea(c))
    if(cv2.arcLength(c,True) > maxHeight*.5):
        section.append(c[0,0,0])

section = sorted(section)
section.append(maxWidth)
print(section)

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
def findNumbers(group,img):
#finds n many numbers inside of box
    for glist in group:
        glist = sorted(glist, key=lambda c: cv2.boundingRect(c)[0])
        glist.reverse()
        print(cv2.boundingRect(glist[0])[0])
        if( cv2.boundingRect(glist[0])[0] < (600-xval)+100):
            continue            
        val = len(glist)
        for i, j in enumerate(glist[:-1]):
            print(f"{cv2.boundingRect(j)[0]} - {cv2.boundingRect(glist[i+1])[0] + cv2.boundingRect(glist[i+1])[2]} = {cv2.boundingRect(j)[0] - (cv2.boundingRect(glist[i+1])[0] + cv2.boundingRect(glist[i+1])[2])}")
            if (cv2.boundingRect(j)[0] - (cv2.boundingRect(glist[i+1])[0] + cv2.boundingRect(glist[i+1])[2])) > 30:
                val = i+1
                break
        print(f"val {val} glist size {len(glist)}")
        glist = glist[0:val]
        glist.reverse()
        i=0
        for c in glist:
            if cv2.contourArea(c) < cv2.arcLength(c,True):
                continue
            x,y,w,h = cv2.boundingRect(c)
            finalimg = img[y:y+h,x:x+w]
            finalimg = np.pad(finalimg,pad_width=5,mode='constant',constant_values=0)
            #finalimg = np.invert(finalimg)
            plt.imshow(finalimg)
            plt.show()
            #makes image MNIST size
            finalimg = cv2.resize(finalimg,(28,28))
            finalimg = cv2.erode(finalimg, np.ones((2, 2), np.uint8), iterations=1)
            (thresh, finalimg) = cv2.threshold(finalimg, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if zeroRowCount(finalimg) > 21:
                continue
            # plt.imshow(finalimg)
            # plt.show()
            cv2.imwrite('python/generated/img'+str(i)+'.png',finalimg)
            i+=1
            

for t in range(len(section)-1):

    pts = [ [section[t],0],[section[t],maxHeight],[section[t+1],maxHeight],[section[t+1],0] ]

    newBound = [ [0,0],[0,2000],[1000,2000],[1000,0] ]

    M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(newBound))
    # Perspective transform using homography.
    finalimg = cv2.warpPerspective(fimg, M, (1000,2000), flags=cv2.INTER_LINEAR)
    plt.imshow(finalimg)
    plt.show()

    testimg = np.copy(finalimg)

    gray = cv2.cvtColor(testimg,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(testimg, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    testimg = np.invert(testimg)
    result = cv2.morphologyEx(testimg, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    result = np.invert(result)

    plt.imshow(result)
    plt.show()
    # timg = fimg[0:maxHeight,0:section[0]]
    # plt.imshow(timg)
    # plt.show()

    #coordinates = [[750,620],[750,655],[750,710],[670,1140],[670,1190]]
    #boxSize = [200,50]
    hist = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    # hist = np.invert(hist)
    ret,th = cv2.threshold(hist, 200, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO_INV)
    # print(ret)
    # plt.imshow(th)
    # plt.show()

    # histgram = cv2.calcHist([th], [0], None, [256], [1, 256])

    # plt.figure()
    # plt.title("Grayscale Histogram")
    # plt.xlabel("Bins")
    # plt.ylabel("# of Pixels")
    # plt.plot(histgram)
    # plt.xlim([1,256])
    # plt.show()

    #th = np.invert(th)

    # ret,th = cv2.threshold(th, 100, 255, cv2.THRESH_TOZERO_INV)
    # otsu,t = cv2.threshold(th, 100, 255, cv2.THRESH_OTSU)
    # print("otsu")
    # print(otsu)

    
    th[:,0:500] = 0
    th[np.r_[0:500,1700:2000],:] = 0

    plt.imshow(th)
    plt.show()

    # res = cv2.cvtColor(finalimg,cv2.COLOR_BGR2GRAY)
    # #res = np.invert(res)
    # contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # thcon = np.copy(th)
    # for c in contours:
    #     if(cv2.arcLength(c,True) > cv2.contourArea(c) and cv2.arcLength(c,True) > 200):
    #     #thcon = cv2.drawContours(thcon,contours, -1, (0,255,255),3)
    #         x,y,w,h = cv2.boundingRect(c)
    #         thcon = cv2.rectangle(thcon,(x,y),(x+w,y+h),255,5)
    #         #temp = th[y:y+h,x:x+w]
    # plt.imshow(thcon)
    # plt.show()
    




    # i = 0
    #goes box by box
    
    # mask = np.zeros(th.shape[:2],dtype=th.dtype)
    # r = np.array([[p[0],p[1]],[p[0]+boxSize[0],p[1]],[p[0]+boxSize[0],p[1]+boxSize[1]],[p[0],p[1]+boxSize[1]]])
    # cv2.rectangle(mask,r[0],r[2],(255,255,255),-1)
    # #save to temp image and leave only data that is inside
    # temp = cv2.bitwise_and(th,th,mask=mask)
    #uses contours with external fill to seperate digits 
    test = cv2.cvtColor(finalimg,cv2.COLOR_BGR2GRAY)
    ret,rth = cv2.threshold(test, 200, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO_INV)
    plt.imshow(rth)
    plt.show()

    temp = th[np.r_[600:900,1100:1200,1500:1600],xval:950]
    real = rth[np.r_[600:900,1100:1200,1500:1600],xval:950]
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list = []
    grouped = [None]
    sortcontours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for c in contours:
        if( cv2.contourArea(c) > cv2.arcLength(c,True)):
            x,y,w,h = cv2.boundingRect(c)
            list.append((c,y,y+h))
    print(f"list size {len(list)}")
    index = 0
    for l in list:
        print(str(l[1])+" "+str(l[2]))
    for i, j in enumerate(list[:-1]):
        if grouped[index] == None:
            grouped[index] = [j[0]]
        if (j[1] <= list[i+1][2] and j[1] >= list[i+1][1]) or (j[2] <= list[i+1][2] and j[2] >= list[i+1][1]):
            grouped[index].append(list[i+1][0])
            print(f"{list[i+1][1]},{list[i+1][2]} added to {j[1]},{j[2]} with index {index}")
        else:
            grouped.append(None)
            index += 1
    if grouped[index] == None:
        grouped[index] = [list[-1][0]]
    # print(grouped)

    #sortcontours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    grouped.reverse()
    findNumbers(grouped,img=real)

