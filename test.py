import numpy as np
import cv2 as cv
import utils

##########################################
path = "1.jpg"
widthImg = 700
heightImg = 700
##########################################

img = cv.imread(path)

#    Preprocessing     ###################
img = cv.resize(img,(widthImg,heightImg))
imgContour = img.copy()
imgMarkContour = img.copy()
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv.Canny(imgBlur,10,50)

#   Finding all Countours   ##############
contours, hierarchy = cv.findContours(imgCanny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
cv.drawContours(imgContour,contours,-1,(0,255,0),10)

#   Find Rectangles     ##################
rectContour = utils.rectContour(contours)
markContourPoints = utils.getCornerPoints(rectContour[0])
gradePoints = utils.getCornerPoints(rectContour[1])

if markContourPoints.size != 0 and gradePoints.size != 0:
    cv.drawContours(imgMarkContour,markContourPoints,-1,(0,0,255),20)
    cv.drawContours(imgMarkContour,gradePoints,-1,(255,0,0),20)

    markContourPoints = utils.reorder(markContourPoints)  # Reorder
    gradePoints = utils.reorder(gradePoints)  # Reorder
    pt1 = np.float32(markContourPoints)
    imgPoint = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv.getPerspectiveTransform(pt1, imgPoint)
    imgWrapColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))

#   Wraping The Image   ##################


#   Stacked Images      ##################
imgBlank = np.zeros_like(img)
imgArray = ([img,imgGray,imgBlur,imgCanny],[imgContour,imgMarkContour,imgWrapColored,imgBlank])
imgStacked = utils.stackImages(imgArray,0.5)

cv.imshow('Stacked',imgStacked)
cv.waitKey(0)