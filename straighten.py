import cv2 
import numpy as np 
from time import sleep

def straighten_document(image,width, height):    
    # Threshold the image to create a mask for the paper region
    _, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
        return image

'''    
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
        return image'''