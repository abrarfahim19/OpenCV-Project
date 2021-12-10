import numpy as np
import cv2 as cv

##########################################
import utils

path = "1.jpg"
widthImg = 700
heightImg = 700
##########################################

img = cv.imread(path)
img = cv.resize(img,(widthImg,heightImg))

imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv.Canny(imgBlur,10,50)

imgArray = ([img,imgGray,imgBlur,imgCanny])
imgStacked = utils.stackImages(imgArray,0.5)

cv.imshow('Stacked',imgStacked)
cv.waitKey(0)