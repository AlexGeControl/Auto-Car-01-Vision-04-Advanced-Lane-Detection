# Set up session:
import numpy as np
import cv2

def get_ROI_mask(
    image_size,
    points
):
    """ Generate ROI mask

    Args:
        image_size (2-element tuple, (W, H)): image size
        points (
            4-element tuple,
            (
                top-left,
                top-right,
                bottom-right,
                bottom-left
            )
        ): points for ROI definition
    """
    # Image dimensions:
    (W, H) = image_size
    # ROI definition:
    (top_left, top_right, bottom_right, bottom_left) = points

    # Generate mask:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask = cv2.fillPoly(
        # Black canvas:
        mask,
        # Quadrilateral region mask:
        np.array(
            [
                [
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left
                ]
            ],
            dtype=np.int
        ),
        # White for further bitwise operation:
        1
    )

    return mask

def get_channel_mask(
    image,
    thresholds
):
    """ Generate mask based on channel component values

    Args:
        image (numpy 2-d array): input image with selected channel component
        channel_idx (int): channel index
        thresholds (2-element tuple): min & max values for thresholding
    """
    # Image dimensions:
    H, W = image.shape

    # Generate mask:
    mask = np.zeros((H, W), dtype=np.uint8)
    channel_min, channel_max = thresholds
    mask[
        (channel_min <= image) & (image <= channel_max)
    ] = 1

    return mask

def get_gradient_mask(
    grayscale,
    type,
    kernel_size,
    scaling_mask,
    thresholds
):
    """ Generate mask based on gradient calculations

    Args:
        grayscale (numpy 2-d array): grayscale input image.
        type (str): one of 'x', 'y', 'mag' or 'ori'
        kernel_size (int): kernel size of Sobel operator.
        scaling_mask (numpy 2-d array): mask for gradient magnitude scaling
        thresholds (2-element tuple): min & max values for gradient thresholding
    """
    # equalize hist
    grayscale = cv2.equalizeHist(grayscale)

    # Calculate gradient components:
    grad_x = np.abs(
        cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, kernel_size)
    )
    grad_y = np.abs(
        cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, kernel_size)
    )

    # Final gradient calculation:
    if type == 'x':
        grad = grad_x
    elif type == 'y':
        grad = grad_y
    elif type == 'mag':
        grad = np.sqrt(grad_x**2 + grad_y**2)
    elif type == 'ori':
        grad = np.arctan2(grad_y, grad_x)
    else:
        grad = np.zeros_like(grayscale, dtype=np.uint8)

    # Keep only ROI:
    grad[
        np.logical_not(scaling_mask)
    ] = 0

    # Scaling:
    grad = (255 * grad / np.max(grad)).astype(np.uint8)

    # Generate mask:
    mask = np.zeros_like(grad)
    grad_min, grad_max = thresholds
    mask[
        (grad_min <= grad) & (grad <= grad_max)
    ] = 1

    return mask

def auto_canny(grayscale, sigma=0.01):
    """ Set lower and upper thresholds using auto Canny heuristics
    """
    median = np.median(grayscale)

    lower = max(
        0,
        int(
            (1.0 - sigma) * median
        )
    )
    upper = min(
        255,
        int(
            (1.0 + sigma) * median
        )
    )

    return cv2.Canny(grayscale, lower, upper)

def get_canny_mask(
    grayscale,
    ROI_mask
):
    """ Generate mask using auto Canny detector
    """
    # Get gradients:
    grad = auto_canny(grayscale)
    grad[np.logical_not(ROI_mask)] = 0

    # Generate mask:
    mask = np.zeros_like(grad)
    mask[grad > 0] = 1

    return mask
