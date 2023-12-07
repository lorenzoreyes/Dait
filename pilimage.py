import cv2, pytesseract
import numpy as np
from PIL import Image

# Load the image
image = Image.open("topa.jpeg")
text = pytesseract.image_to_string(image)

cv2.imwrite('topeado.jpeg', result) 

# function to apply and check changes on the go
def preview(name,effect):
    cv2.imwrite(f'{name}.jpeg', effect)
    img = Image.open(f'{name}.jpeg')
    return img


print(text)
