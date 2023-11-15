import cv2

img = cv2.imread('topa.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.resize(img,(520,900))


_, result = cv2.threshold(img,85,255,cv2.THRESH_BINARY)
adapt = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49,10)

#cv2.imshow("original",img)
#cv2.imshow("result",result)

cv2.imwrite('topeado.jpeg', result) 
cv2.imwrite('topeadapt.jpeg', adapt) 