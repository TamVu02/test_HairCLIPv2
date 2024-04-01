
import numpy as np
from PIL import Image
from torchvision import transforms

# image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# Create a function that takes two images’ paths as a parameter
def calculate_psnr(src_tensor, ref_tensor):

   # Get the square of the difference
   squared_diff = np.square(diff)

   # Compute the mean squared error
   mse = np.mean(squared_diff)

   # Compute the PSNR
   max_pixel = 255
   psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
   return psnr

src_image = Image.open('niqe-master/test_imgs/00734.png').convert('RGB')
ref_image = Image.open('niqe-master/test_imgs/00734_00836_refine_3.png').convert('RGB')

if src_image is None or ref_image is None:
   print("Failed to load one or both images.")
else:   
   # Call the above function and perform the calculation
   psnr_score = calculate_psnr(src_image, ref_image)
   # Display the result
   print("PSNR:", psnr_score)
