import cv2
from matplotlib import pyplot as plt
img = cv2.imread('car2.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cars = car_cascade.detectMultiScale(gray, 1.1, 1)
for (x, y, w, h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 

plt.figure(figsize=(10,20))
plt.imshow(img)
img = cv2.imread('car2.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cars = car_cascade.detectMultiScale(gray, 1.1, 1)
for (x, y, w, h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 

plt.figure(figsize=(10,20))
plt.imshow(img)
face_cascade = cv2.CascadeClassifier('cars.xml')
img = cv2.imread('car4.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cars = face_cascade.detectMultiScale(gray, 1.1, 1)
for (x, y, w, h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 
plt.figure(figsize=(10,20))

plt.imshow(img)