import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, camera_matrix, dist_coeff):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undistorted = cv2.undistort(img, camera_matrix, dist_coeff)
    # 2) Convert to grayscale
    grayscale = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    detection_status, corners = cv2.findChessboardCorners(
        grayscale,
        (nx, ny)
    )
    # 4) If corners found:
    if detection_status != 0:
            # Parse image size:
            (H, W) = undistorted.shape[:2]
            # a) draw corners
            cv2.drawChessboardCorners(
                undistorted,
                (nx, ny),
                corners,
                detection_status
            )
            # b) define 4 source points
            src = np.float32(
                [
                    corners[         0, 0],
                    corners[      nx-1, 0],
                    corners[ny*nx - nx, 0],
                    corners[ ny*nx - 1, 0]
                ]
            )
            # c) define 4 destination points
            dst = np.float32(
                [
                    [int(0.1*W), int(0.1*H)],
                    [int(0.9*W), int(0.1*H)],
                    [int(0.1*W), int(0.9*H)],
                    [int(0.9*W), int(0.9*H)]
                ]
            )
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            M = cv2.getPerspectiveTransform(
                src,
                dst
            )
            # e) use cv2.warpPerspective() to warp your image to a top-down view
            warped = cv2.warpPerspective(
                undistorted,
                M,
                (W, H),
                cv2.INTER_LINEAR
            )
    #delete the next two lines
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
