import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Read in an image
image = mpimg.imread('signs_vehicles_xygrad.png')

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    d_x = cv2.Sobel(grayscale, cv2.CV_64F,1,0, ksize=sobel_kernel)
    d_y = cv2.Sobel(grayscale, cv2.CV_64F,0,1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    d_mag = np.sqrt(d_x**2 + d_y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    mag_standardized = (255 * d_mag / np.max(d_mag)).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    mag_min, mag_max = mag_thresh
    mask = np.zeros_like(grayscale)
    mask[(mag_min <= mag_standardized) & (mag_standardized <= mag_max)] = 1
    # 6) Return this mask as your binary_output image
    return mask

# Run the function
mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
