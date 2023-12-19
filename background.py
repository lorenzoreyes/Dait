from rembg import remove
import glob, cv2, pytesseract, imutils
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
#from straighten import straighten_document
from scipy import ndimage

photos = []

for i in glob.iglob('./fotos/*jpeg'):
        photos.append(i)
    
# save the photo as pngs & with the background removed
for i in range(1,len(photos)):
  inn = Image.open(photos[i])
  out = remove(inn)
  photos[i] = photos[i].replace('jpeg','png')
  out.save(photos[i])
  # crop the non-transparent part
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
  # Crop the non-transparent region
  image = bgr_image[x0:x1, y0:y1]
  cv2.imwrite(f'{photos[i]}', image)


for i in range(len(photos)):
    try:
        image = cv2.imread(photos[i], cv2.IMREAD_UNCHANGED)
        width, height = image.shape[1],image.shape[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(gray)
        #kernel = np.ones((5,5),np.uint8)
        #image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        osd = pytesseract.image_to_osd(image,config='--psm 0 -c min_characters_to_try=5',output_type='dict')
        orientation, rotate, confidence = osd['orientation'], osd['rotate'], osd['orientation_conf']
        angle = orientation # rotate if confidence < 0.6 else orientation
        angle = 0 if orientation == rotate else angle
        rotated = imutils.rotate_bound(image, angle)
        osd = pytesseract.image_to_osd(rotated ,config='--psm 0 -c min_characters_to_try=5',output_type='dict')
        orientation, rotate, confidence = osd['orientation'], osd['rotate'], osd['orientation_conf']
        angle = rotate if confidence < 0.6 else orientation
        angle = 0 if orientation == rotate else angle
        rotated = imutils.rotate_bound(rotated , angle)
        #gray = cv2.adaptiveThreshold(rotated, 165, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,71,12)  
        image = cv2.resize(rotated,(width*2,height*2), interpolation=cv2.INTER_LINEAR_EXACT)
        cv2.imwrite(f'{photos[i][:-4]}WB{angle}.png',image)
        print(photos[i], orientation, rotate, confidence)
    except:
        print(f'{photos[i]} Cannot read OSD')
        
images = []

for i in glob.iglob('./fotos/*WB*'):
        images.append(i)
        
for i in images:
    straight = straighten(i)
    cv2.imwrite(f'{i[:-4]}Straight.png',straight)


        
def straighten(image):
    image = cv2.imread(image, 0)  # Assuming it's a grayscale image
    width, height = image.shape[0],image.shape[1]
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Approximate the contour to get the simplified shape
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    # Ensure the contour is a rectangle
    x, y, w, h = cv2.boundingRect(approx)
    # Define the four corner points of the rectangle
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    # Define the target rectangle for perspective transformation
    target_width, target_height = width, int(width*0.75)# Define your desired size
    target_corners = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
    # Perform perspective transformation
    homography_matrix, _ = cv2.findHomography(corners, target_corners)
    result = cv2.warpPerspective(image, homography_matrix, (target_width, target_height))
    return result