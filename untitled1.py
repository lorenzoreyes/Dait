import cv2

img = cv2.imread('topa.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(1020,1800))


_, result = cv2.threshold(img,20,255,cv2.THRESH_BINARY)
adapt = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41,10)
cv2.imshow("original",img)
cv2.imshow("result",result)