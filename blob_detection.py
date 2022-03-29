import cv2
from cv2 import erode
from matplotlib import image
import numpy as np
from pytesseract import image_to_string
def blackW(image):
    _config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789'
    img = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    (thresh, blackAndWhiteImage) = cv2.threshold(img, 129, 255, cv2.THRESH_BINARY)
    cv2.imshow('black_and_white',blackAndWhiteImage)
    return blackAndWhiteImage
# kernel = np.ones((2,2), np.uint8)  
# img_erosion = erode(blackAndWhiteImage, kernel, iterations=1)  
# cv2.imshow('erode',img_erosion)
# text=image_to_string(img_erosion,config=_config)
# print(text)
