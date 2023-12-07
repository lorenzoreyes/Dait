# Split the image into its channels
b, g, r, alpha = cv2.split(input_image)

# Threshold the alpha channel to find the non-transparent region
_, mask = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a black background for drawing contours
contour_image = np.zeros_like(input_image)

# Draw contours around the object
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Convert BGR image to RGB for displaying with matplotlib
contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)