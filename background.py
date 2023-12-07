from rembg import remove
import glob, cv2, pytesseract
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 

photos = []

for i in glob.iglob('./fotos/*jpeg'):
    photos.append(i)

pngs = [i.replace('jpeg','png') for i in photos]

# save the photo as pngs & with the background removed
for i in range(1,len(photos)):
    inn = Image.open(photos[i])
    out = remove(inn)
                 #alpha_matting=True,
                 #alpha_matting_foreground_threshold=240,
                 #alpha_matting_background_threshold=10,
                 #alpha_matting_erode_structure_size=10,
                 #alpha_matting_base_size=1000,
                 #)
    out.save(pngs[i])
    
photos = []
for i in glob.iglob('./fotos/*png'):
    photos.append(i)
    

# load them and save them as grayscale to be readable again
for i in range(len(photos)):
    image = cv2.imread(photos[i], cv2.IMREAD_UNCHANGED)
    width, height = image.shape[1],image.shape[0]  # Width & Height of the original image
    # Split the image into its color channels (BGR) and alpha channel (A)
    bgr_image = image[:, :, :3]  # BGR channels
    alpha = image[:, :, 3]  # Alpha channel
    # Find the bounding box of the non-transparent region
    coords = np.argwhere(alpha > 0)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # Adding 1 to include the last row and column
    # Create a mask of transparent pixels
    mask = alpha == 0  # Set mask to True where alpha channel is 0 (transparent)
    # Replace transparent areas with white color
    # Set RGBA values to white where mask is True
    image[mask] = [255, 255, 255, 255]  
    #nomask = alpha != 0
    #black_value = [50, 50, 50,50]
    #image[nomask] = black_value
    # Crop the non-transparent region
    image = bgr_image[x0:x1, y0:y1]
    image = cv2.resize(image,(width,height), interpolation=cv2.INTER_NEAREST)
    # Set non-white and non-transparent pixels to a unique value of black (e.g., [50, 50, 50])
    # Stretch or resize the cropped region to cover the entire canvas
    #stretched_image = cv2.resize(image, (width, height))
    # Increase contrast using histogram equalization
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Denoising using Gaussian blur
    #equalized_image = cv2.equalizeHist(gray_image)
    #enhanced_contrast = cv2.cvtColor(equalized_image , cv2.COLOR_GRAY2BGR)
    # Apply thresholding
    #_, thresholded_image = cv2.threshold(gray_image , threshold_value, 255, cv2.THRESH_BINARY)
    # Adaptive threshold
    # 4k image = cv2.resize(thresholded_image, (3840,2160), interpolation=cv2.INTER_NEAREST)  
    image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,81,15)  
    image = cv2.resize(image,(width*5,height*5), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f'{photos[i][:-4]}WB.png',image)# load them and save them as grayscale to be readable again
    
def straighten_document(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
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
        width = 210  # A4 width in millimeters
        height = 297  # A4 height in millimeters
        dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        
        # Calculate the perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
        
        # Apply the perspective transformation
        warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
        
        return warped_image
    
    else:
        print("Could not detect a proper document boundary.")
        return image