# import the cv2 as well as numpy library
import cv2
import numpy as np
# Create a function that takes two images’ paths as a parameter
def calculate_psnr(firstImage, secondImage):
   # Compute the difference between corresponding pixels
   diff = np.subtract(firstImage, secondImage)
   # Get the square of the difference
   squared_diff = np.square(diff)

   # Compute the mean squared error
   mse = np.mean(squared_diff)

   # Compute the PSNR
   max_pixel = 255
   psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
   return psnr

# Resize images to a common size
rHeight = 256
rWidth = 256

# Read the original and distorted images
firstI = cv2.imread('niqe-master/test_imgs/00734_00836_refine_3.png')
secondI = cv2.imread('niqe-master/test_imgs/00734.png')

# Check if images are loaded successfully
if firstI is None or secondI is None:
    print("Failed to load one or both images.")
else:
   # Resize images for first image
    firstI = cv2.resize(firstI, (rWidth, rHeight))
   # Resize the details for second image
    secondI = cv2.resize(secondI, (rWidth, rHeight))
    
   # Call the above function and perform the calculation
    psnr_score = calculate_psnr(firstI, secondI)
   # Display the result
    print("PSNR:", psnr_score)