import cv2 
import numpy as np 
from time import sleep
from scipy import ndimage 

def straighten_document(image,width, height):    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)    
    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    # Get the largest contour (assumed to be the document boundary)
    largest_contour = max(contours, key=cv2.contourArea)    
    # Get the corners of the largest contour
    perimeter = cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)    
    # Ensure the contour has 4 corners (a rectangle)
    if len(corners) == 4:
        # Order the corners clockwise starting from the top-left
        corners = np.array([corner[0] for corner in corners])
        ordered_corners = np.zeros((4, 2), dtype=np.float32)
        
        # Calculate the sum and difference for ordering
        sum_coords = corners.sum(axis=1)
        ordered_corners[0] = corners[np.argmin(sum_coords)]
        ordered_corners[2] = corners[np.argmax(sum_coords)]
        
        diff_coords = np.diff(corners, axis=1)
        ordered_corners[1] = corners[np.argmin(diff_coords)]
        ordered_corners[3] = corners[np.argmax(diff_coords)]
        
        # Define the destination points for perspective transformation (A4 size)
        #width = 210  # A4 width in millimeters
        #height = 297  # A4 height in millimeters
        dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        
        # Calculate the perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
        
        # Apply the perspective transformation
        warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
        
        return warped_image
    
    else:
        print("Could not detect a proper document boundary.")
        sleep(1)
        color = [255, 255, 255]
        top, bottom, left, right = [150]*4
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        image = straighten_document(image,width, height)
        return image
    
def straighten_image(image):
    # Split the image into its channels
    b, g, r, alpha = cv2.split(image)
    width, height = image.shape[1],image.shape[0]
    # Threshold the alpha channel to find the non-transparent region
    _, mask = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a black background for drawing contours
    #contour_image = np.zeros_like(image)
    # Draw contours around the object
    #cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 4)
    # Convert BGR image to RGB for displaying with matplotlib
    #contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    # Get the largest contour assuming it represents the document boundary
    largest_contour = max(contours, key=cv2.contourArea)
    # Get the approximate polygon of the contour
    perimeter = cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

    # Rearrange the corners of the approximated polygon
    pts = np.float32([approx_polygon[0], approx_polygon[1], approx_polygon[2], approx_polygon[3]])


    # Define the corresponding corners in the output image
    dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # Calculate the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(pts, dst_pts)

    # Apply the perspective transformation to get the straightened image
    straightened_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
    return straightened_image
    

def ocr(gray):
    # Apply edge detection (Canny edge detection)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Perform line detection using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Calculate the angle of the detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)

    # Calculate the median angle
    median_angle = np.median(angles)

    print(f"Detected angle: {median_angle} degrees")
    rotated = ndimage.rotate(gray, median_angle-(2*median_angle),reshape=True)
    return rotated 

'''
for i in range(len(photos)):
    image = cv2.imread(photos[i], cv2.IMREAD_UNCHANGED)
    # Split the image into its channels
    b, g, r, alpha = cv2.split(input_image)
    # Threshold the alpha channel to find the non-transparent region
    _, mask = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a black background for drawing contours
    contour_image = np.zeros_like(input_image)
    # Draw contours around the object
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 4)
    # Convert BGR image to RGB for displaying with matplotlib
    contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{photos[i][:-4]}contours.jpeg',contour_image_rgb)

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image
    
def straighten_text_orientation(image_path):
    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection or any other edge detection method
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Calculate the average angle of detected lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        angles.append(angle)

    avg_angle = np.mean(angles)

    # Rotate the image to straighten the text orientation
    rotated_image = rotate_image(image, -avg_angle)

    return rotated_image


    # Split the image into its channels
    b, g, r, alpha = cv2.split(image)
    # Width & Height of the original image
    width, height = image.shape[1],image.shape[0]  
    # Threshold the alpha channel to find the non-transparent region
    _, mask = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour assuming it represents the document boundary
    largest_contour = max(contours, key=cv2.contourArea)
    # Get the approximate polygon of the contour
    perimeter = cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

    # Rearrange the corners of the approximated polygon
    pts = np.float32([approx_polygon[1], approx_polygon[0], approx_polygon[3], approx_polygon[2]])   
    # Define the corresponding corners in the output image
    dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # Calculate the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(pts, dst_pts)

    # Apply the perspective transformation to get the straightened image
    straightened_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
    return straightened_image

def find_document_corners(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection or any other edge detection method
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour assuming it represents the document boundary
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the approximated polygon of the contour (approximating it as a rectangle)
    perimeter = cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

    # Rearrange the corners of the approximated polygon
    approx_polygon = approx_polygon.reshape(-1, 2)
    corners = np.float32([approx_polygon[0], approx_polygon[1], approx_polygon[2], approx_polygon[3]])

    return corners, original_image

def match_to_a4(corners, original_image):
    # Define the corners of an A4 paper (in A4's dimensions: 210mm × 297mm)
    a4_corners = np.float32([[0, 0], [210, 0], [210, 297], [0, 297]])

    # Calculate the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(corners, a4_corners)

    # Apply the perspective transformation to match to A4 dimensions
    a4_image = cv2.warpPerspective(original_image, perspective_matrix, (210, 297))

    return a4_image
'''cv2.approxPolyDP(curve, epsilon, closed)