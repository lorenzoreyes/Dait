import cv2, pytesseract
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('tres288.jpeg', cv2.IMREAD_UNCHANGED)

blur = cv2.GaussianBlur(image, (5,5), 0)

canny = cv2.Canny(blur, threshold1=15,threshold2=80)

# Define the kernel for dilation
kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed

# Dilate the edges
dilated = cv2.dilate(canny, kernel, iterations=1)

# Find contours in the dilated image
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define the minimum area ratio for partially detected shapes (adjust as needed)
min_area_ratio = 0.7  # 70% of the total area