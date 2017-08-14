import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Read in an image
image = mpimg.imread('signs_vehicles_xygrad.png')

# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    sobel_y_abs = np.abs(sobel_y)
    sobel_x_abs = np.abs(sobel_x)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    orientation = np.arctan2(sobel_y_abs, sobel_x_abs)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(grayscale)
    orientation_min, orientation_max = thresh
    mask[(orientation_min <= mask) & (mask <= orientation_max)] = 1
    # 6) Return this mask as your binary_output image
    return mask

# Run the function
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)