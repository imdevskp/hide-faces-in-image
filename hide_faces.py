# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:17:03 2020

@author: imdevskp
"""

import sys

import matplotlib.pyplot as plt
from matplotlib import patches

from skimage import data

from skimage.feature import Cascade
from skimage.filters import gaussian
   
def save_image(image):
    plt.imshow(image)
    plt.axis('off')
    save_img_name = 'face_hidden.jpg'
    plt.savefig(save_img_name, dpi=300, bbox_inches='tight', pad_inches=0)
    print('Image saved as', save_img_name)

def getFace(d):
    ''' Extracts the face rectangle from the image using the
    coordinates of the detected.'''
    
    # X and Y starting points of the face rectangle
    x, y = d['r'], d['c']
    
    # The width and height of the face rectangle
    width, height = d['r'] + d['width'], d['c'] + d['height']
    
    # Extract the detected face
    face= image[x:width, y:height]
    return face

def mergeBlurryFace(original, gaussian_image):
    # X and Y starting points of the face rectangle
    x, y = d['r'], d['c']
    
    # The width and height of the face rectangle
    width, height = d['r'] + d['width'], d['c'] + d['height']
    
    original[x:width, y:height] = gaussian_image
    return original

# read image
image = plt.imread(sys.argv[1])
image= image.copy()

# Load the trained file from the module root.
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade.
detector = Cascade(trained_file)

# Detect the faces
detected = detector.detect_multi_scale(img=image,
                                       scale_factor=1.2, step_ratio=1,
                                       min_size=(50, 50), max_size=(200, 200))
# No. of faces detected
print('No. of faces detected :', len(detected))

# For each detected face
for d in detected:
    # Obtain the face cropped from detected coordinates
    face = getFace(d)
    
    # Apply gaussian filter to extracted face
    gaussian_face = gaussian(face, multichannel=True, sigma = 10)
    
    # Merge this blurry face to our final image and show it
    resulting_image = mergeBlurryFace(image, gaussian_face)
    
save_image(resulting_image)