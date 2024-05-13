import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
#listdir is used to list all the images that are present in our folder
# print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
    # this above line will enter all the names of people into a list, it removes the.jpg
# print(classNames)

#we created a function to find encodings
def findEncodings(images):
    encodeList= []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        #in this loop, we will pick each picture

    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList= f.readlines()
        nameList= []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')



encodeListknown = findEncodings(images)
print('Encoding Complete, Length= ', len(encodeListknown))

cap = cv2.VideoCapture(0)
while True:
    success, img= cap.read()
    #imgSmall is the frame captured from the video of our video cam
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    #here we are finding location of faces in current frame of videocam, and then encoding it
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall,facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        #we are using Zip because we want these two items in one loop
        matches = face_recognition.compare_faces(encodeListknown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
        #here we are comparing the encodings of the images in our folder with the imageframe taken from our webcam
        # print(faceDis)
        #printing face distance will tell us that the face in front of webcam is closest to which of the images from our database
        #now we will try to find best match based on these distances from our list
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1= faceLoc
            y1, x2, y2, x1= y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

