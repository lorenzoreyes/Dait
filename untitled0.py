import cv2
import numpy as np

# Load the transparent image
image = cv2.imread('object.png', cv2.IMREAD_UNCHANGED)

# Extract alpha channel
alpha_channel = image[:, :, 3]

# Find contours of the object
contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the image for plotting
image_with_contours = np.copy(image)

# Draw contours on the image
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Contours in green color

# Display the image with contours
cv2.imshow('Object with Contours', image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
