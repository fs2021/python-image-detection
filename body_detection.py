import cv2
import numpy as np
import os


#path for face cascade, path where it is stored specific classifier
# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')


#check all files in images dir:
rootdir = 'images'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg"):
            #print (filepath)
            detected_bodies = 0
            img = cv2.imread(filepath, -1) # -1 unchanged, 0- gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            bodies = body_cascade.detectMultiScale(gray, 1.3,3)
            upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.3, 3)
            faces = face_cascade.detectMultiScale(gray, 1.3,5)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w, y+h), (255,0,0),3)
                detected_bodies += 1 

            for (x,y,w,h) in bodies:
                cv2.rectangle(img,(x,y),(x+w, y+h), (255,0,0),2)
                detected_bodies += 1 


            for (x,y,w,h) in upper_bodies:
                cv2.rectangle(img,(x,y),(x+w, y+h), (255,0,0),2)
                detected_bodies += 1 
                
            print(f'Detected {detected_bodies} in {filepath}')


#imgfile_name = 'images/im03.jpg'
#img = cv2.imread(imgfile_name, -1) # -1 unchanged, 0- gray
#img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) 
#draw rectangle over image
#img = cv2.rectangle(img, (300, 300), (200, 200), (134, 56, 132), 6)



cv2.imshow('Image', img)

cv2.waitKey(0) #time to display, 0 - wait for key
cv2.destroyAllWindows()

# print(img.shape) #2d array for gray, 3d arr for color rgb


# # video from camera 0
# cap = cv2.VideoCapture(0) 

# # from file 
# #cap = cv2.VideoCapture('video.mp4')

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3,5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w, y+h), (255,0,0),3)
#     #region of interest for eyes:
#         roi_gray = gray[y:y+w,x:x+w]
#         roi_color = frame[y:y+h,x:x+w]

#         eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)

#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

#     cv2.imshow('frame',frame)

#     cv2.imshow('Frame', frame)#color
#     #cv2.imshow('Gray', gray)
#     #ord(char) takes a char as a parameter and returns its ASCII value.
#     if (cv2.waitKey(1)==ord('e')):  #press key e to escape
#         break
# cap.release()
# cap.destroyAllWindows()