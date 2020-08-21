import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('test.jpg')
con_pic = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(con_pic, 1.1, 4)

for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 3)

cv2.imshow('FaceDetect', img)
cv2.waitKey()