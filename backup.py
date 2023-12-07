from rembg import remove
import glob, cv2, pytesseract
from PIL import Image
import numpy as np 

photos = []

for i in glob.iglob('./fotos/*jpeg'):
    photos.append(i)

pngs = [i.replace('jpeg','png') for i in photos]

# save the photo as pngs & with the background removed
for i in range(1,len(photos)):
    inn = Image.open(photos[i])
    out = remove(inn)
    out.save(pngs[i])
    
photos = []
for i in glob.iglob('./fotos/*png'):
    photos.append(i)



threshold_value = 140

# load them and save them as grayscale to be readable again
for i in range(len(photos)):
    image = cv2.imread(photos[i], cv2.IMREAD_UNCHANGED)
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
    # Crop the non-transparent region
    image = bgr_image[x0:x1, y0:y1]
    # Threshold to create a mask of non-white pixels
    _, mask = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    # Set non-white pixels to black
    black_value = np.array([0, 0, 0])  # Black color value
    image[mask != 255] = 0
    # Increase contrast using histogram equalization
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Denoising using Gaussian blur
    #equalized_image = cv2.equalizeHist(gray_image)
    #enhanced_contrast = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    # Apply thresholding
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'{photos[i][:-4]}WB.png',thresholded_image)# load them and save them as grayscale to be readable again
    