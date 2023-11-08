import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import combinations
import math

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

def zeroRowCount(img):
    count = 0

    for i in range (28):
        for j in range (28):
            if img[i][j] != 0:
                break
            if j == 27:
                count += 1

    return count

# read input file
img = cv2.imread("python/niceScan.jpg")

# resize
img = resizeImage(img)

original_img = img

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
      break
cv2.drawContours(con, c, -1, (0, 255, 255), 3)
cv2.drawContours(con, corners, -1, (0, 255, 0), 10)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img,(int(x),int(y)),50,(36,255,12),-1)

img = original_img

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
    cv2.circle(img,(int(corner[0][0]),int(corner[0][1])),50,(36,255,12),-1)

plt.imshow(img)
plt.show()

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

plt.imshow(fimg)
plt.show()


gimg = cv2.cvtColor(fimg.copy(), cv2.COLOR_BGR2GRAY)

#ret,th = cv2.threshold(gimg, 150, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret,th = cv2.threshold(gimg, 150, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
detected_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, vertical_kernel, iterations=10)
# temp = cv2.erode(th, vertical_kernel, iterations=1)
detected_lines = cv2.dilate(detected_lines, (1,25), iterations=5)

#remove lines that are straight and long
cnts, hierarchy = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(fimg, cnts, -1, (0, 255, 0), 3)

plt.imshow(fimg)
plt.show()

section = [0]
for c in cnts:
    #print(cv2.arcLength(c,False))
    #print(cv2.contourArea(c))
    if(cv2.contourArea(c) < maxHeight*8):
        section.append(c[0,0,0])

section = sorted(section)

section.pop()

avg_section_distance = 0

# removes duplicates and calculates average section distance
for x in range(len(section) - 1):
    if x != 0:
        if section[x] > section[x+1] - 100:
            section.pop(x)
        else:
            avg_section_distance += section[x+1] - section[x]

# adds final section
avg_section_distance /= len(section) - 2
section.append(section[-1] + avg_section_distance)














##################################
# v STILL HAVE YET TO GO THROUGH v














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

                finalimg = cv2.erode(finalimg, np.ones((2, 2), np.uint8), iterations=1)

                (thresh, finalimg) = cv2.threshold(finalimg, 100, 255, cv2.THRESH_BINARY)

                if zeroRowCount(finalimg) > 21:
                    continue

                plt.imshow(finalimg, cmap = 'binary_r')
                plt.show()
                
                cv2.imwrite('python/generated/img'+str(i)+'.png',finalimg)
                i+=1
        

