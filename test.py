import numpy as np
import cv2 as cv
import utils

##########################################
path = "test.jpg"
widthImg = 700
heightImg = 700
question = 5
choice = 5
answer = [1, 2, 0, 1, 4]
##########################################

img = cv.imread(path)

#    Preprocessing     ###################
img = cv.resize(img,(widthImg,heightImg))
imgContour = img.copy()
imgMarkContour = img.copy()
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv.Canny(imgBlur,10,50)

#   Finding all Contours   ##############
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

#   Wraping the Answer    ################
pt1 = np.float32(markContourPoints)
imgPoint = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix = cv.getPerspectiveTransform(pt1, imgPoint)
imgWrapColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))

#   Wraping the Mark    ##################
pt2 = np.float32(gradePoints)
imgGradePoint = np.float32([[0, 0], [315, 0], [0, 200], [315, 200]])
matrixGrade = cv.getPerspectiveTransform(pt2, imgPoint)
imgGradeWrapColored = cv.warpPerspective(img, matrixGrade, (300, 100))
#cv.imshow('Grade', gradeWrapColored)

#   Threshold of the answer ##############
imgWrapGrey = cv.cvtColor(imgWrapColored,cv.COLOR_BGR2GRAY)
imgThresh = cv.threshold(imgWrapGrey,165,255,cv.THRESH_BINARY_INV)[1]

#   Slicing into Box    ##################
boxes = utils.splitBoxes(imgThresh)
#cv.imshow('last',boxes[24])

#   Getting Pixel Value From Box    ######
pixelValue = np.zeros((question, choice))
col = 0
row = 0
for image in boxes:
    totalPixel = cv.countNonZero(image)
    pixelValue[row][col] = totalPixel
    col += 1
    if col == choice:
        row += 1
        col = 0
print(pixelValue)

#   Extracting Result from the OMR  ######
omrSelection = []
for each in range(0,question):
    array = pixelValue[each]
    omr = np.where(array == np.amax(array))
    omrSelection.append(omr[0][0])
print(omrSelection)

#   Grading     ##########################
grading = []
for each in range(0, question):
    if answer[each] == omrSelection[each]:
        grading.append(1)
    else:
        grading.append(0)
#print(grading)
grade = (sum(grading)/question)*100
print(grade)

#   Marking Answer      ##################
imgResult = imgWrapColored.copy()
imgResult = utils.showAnswers(imgResult, omrSelection, grading, answer, question, choice)

imgRawDrawing = np.zeros_like(imgWrapColored)
imgRawDrawing = utils.showAnswers(imgRawDrawing, omrSelection, grading, answer, question, choice)

invMatrix = cv.getPerspectiveTransform(imgPoint, pt1)
imgInvRaw = cv.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

imgFinal = img.copy()
imgFinal = cv.addWeighted(imgFinal,1,imgInvRaw,1,0)

imgRawGrade = np.zeros_like(imgGradeWrapColored)
cv.putText(imgRawGrade,str(int(grade))+"%",(50,100),cv.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
cv.imshow("Grade",imgRawGrade)

cv.imshow('Final Output',imgFinal)
#   Stacked Images      ##################
imgBlank = np.zeros_like(img)

imgArray = ([img, imgGray, imgBlur, imgCanny],
            [imgContour, imgMarkContour, imgWrapColored, imgThresh],
            [imgResult, imgRawDrawing, imgInvRaw, imgFinal])
imgStacked = utils.stackImages(imgArray, 0.3)

cv.imshow('Stacked',imgStacked)
cv.waitKey(0)