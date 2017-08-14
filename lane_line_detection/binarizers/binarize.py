# Set up session:
import argparse

import numpy as np
import cv2

from .utils import get_ROI_mask, get_channel_mask, get_gradient_mask, get_canny_mask

# Sklearn transformer interface:
from sklearn.base import TransformerMixin

class Binarizer(TransformerMixin):
    """ Transform input image to thresholded binary image based on gradient magnitude
    """
    def __init__(
        self,
        image_size,
        ROI,
        yellow_lane_hue_thresholds,
        yellow_lane_saturation_thresholds,
        white_lane_saturation_thresholds,
        white_lane_value_thresholds,
        gradient_kernel_size,
        gradient_thresholds,
        morphology_kernel_size
    ):
        # ROI:
        self.ROI = get_ROI_mask(image_size, ROI)

        self.yellow_lane_hue_thresholds = yellow_lane_hue_thresholds
        self.yellow_lane_saturation_thresholds = yellow_lane_saturation_thresholds

        self.white_lane_saturation_thresholds = white_lane_saturation_thresholds
        self.white_lane_value_thresholds = white_lane_value_thresholds

        self.gradient_kernel_size = gradient_kernel_size
        self.gradient_thresholds = gradient_thresholds

        self.morphology_kernel = np.ones(
            (morphology_kernel_size,morphology_kernel_size),
            np.uint8
        )

    def transform(self, X):
        """ Binarize input image
        """
        # Convert to HSV:
        HSV = cv2.cvtColor(
            X, cv2.COLOR_BGR2HSV
        )

        # Split components:
        H, S, V = cv2.split(HSV)

        # Colors:
        yellow_lane_mask = (
            get_channel_mask(H, self.yellow_lane_hue_thresholds) &
            get_channel_mask(S, self.yellow_lane_saturation_thresholds)
        )
        white_lane_mask = (
            get_channel_mask(S, self.white_lane_saturation_thresholds) &
            get_channel_mask(V, self.white_lane_value_thresholds)
        )

        #cv2.imshow("Yellow", 255 * yellow_lane_mask)
        #cv2.waitKey(0)
        #cv2.imshow("White", 255 * white_lane_mask)
        #cv2.waitKey(0)

        # Gradients:
        grayscale = V
        grad_mask = get_gradient_mask(
            grayscale,
            'x',
            self.gradient_kernel_size,
            (self.ROI & (yellow_lane_mask | white_lane_mask)),
            self.gradient_thresholds
        )

        #cv2.imshow("Grad", 255 * grad_mask)
        #cv2.waitKey(0)

        mask = self.ROI & ((yellow_lane_mask | white_lane_mask) | grad_mask)

        #cv2.imshow("Pre-Filter Mask", 255 * mask)
        #cv2.waitKey(0)

        # Morphological filtering:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.morphology_kernel,
            iterations=3
        )

        return mask

    def fit(self, X, y=None):
        """ Do nothing
        """
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == "__main__":
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image filename."
    )
    args = vars(parser.parse_args())

    binarizer = Binarizer(
        [1280, 720],
        [
    		[ 565, 492],
    		[ 715, 492],
    		[1279, 700],
    		[   0, 700]
    	],
        [ 19,  23],
        [ 64, 255],
        [  0,  20],
        [190, 255],
        5,
        [ 32, 128],
        3
    )

    binary = binarizer.transform(
        cv2.imread(args["input"])
    )

    cv2.imshow("Binarized", 255 * binary)
    cv2.waitKey(0)
